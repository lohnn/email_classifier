# Retrain Orchestrator

Native, sandboxed macOS menu-bar app that will operate the full
email-classifier retraining loop (design:
[`../docs/RETRAIN-ORCHESTRATOR-DESIGN.md`](../docs/RETRAIN-ORCHESTRATOR-DESIGN.md)).
Follows the operator decisions locked 2026-10-27/28 (§9) — this is **P1**:
the app skeleton plus the classifier's status-mirror endpoints. Pipeline
execution lands in P2, GGUF merge automation in P3/P4, bucket-mirror in P4.

## P1 — what works already

- Menu bar icon (no Dock icon) with state text; menu **Run now / Settings… /
  Open log / Erase all data… / Quit**
- **Settings window**: trigger threshold (default 10), weekly-cadence toggle
  + interval (default 7 days, requires ≥ 1 new correction), autonomy dial
  (default fully automatic; kept for reversibility), notification category
  switches (failures pinned), Start-with-the-system login-item toggle
- **Classifier connection** (Settings › "Classifier connection"): the base
  URL is an editable setting (host AND port, defaults to
  `http://jarvis:8008`; `classifierURL` in `state.json`), plus a
  **Test connection** button that probes `GET <url>/health` and renders
  exactly what `/health` returned (HTTP code, `status`, per-check truth —
  a 503 or `model <not loaded>` renders as what it is, never smoothed to
  green). Probe happens ONLY on button press — no background poller in P1.
- **Menu-bar indicator dot** next to the state text: gray = not probed,
  green = last probe reachable + healthy, red = last probe failed or
  unhealthy; hover tooltip carries the last probe result + timestamp.
  Driven only by explicit probes, same no-poller rule.
- **Ledger** `state.json` — trigger-policy fields, last-train timestamps,
  corrections baseline, classifier URL; written atomically. Trigger policy
  evaluated by *Run now* (reporting only at P1)
- **Run log** inside the container dir; *Open log* reveals it
- **Erase all data…** — ask-then-wipe of the app's data dir (truly empty
  afterwards; Settings reset to defaults)
- **Login item** via `SMAppService` (macOS 13+): applies immediately,
  survives restarts; NOT wiped by Erase-all-data (it is a launch setting)

Not yet (by phase design): training pipeline execution (P2), the automatic
pre-flight/polling cadence (P2), GGUF merge (P3/P4), GDrive bucket-mirror
(§7, P4).

## Build on the Mac (Command Line Tools only)

One-time prerequisite: `xcode-select --install` (Xcode Command Line Tools,
Swift 5.9+). No full Xcode required.

```bash
cd retrain-orchestrator
./build_app.sh              # swift build → .app assembly → codesign
```

The script is the ONE build loop: run it, and paste any error output back to
the coordinator verbatim — Swift cannot be compiled in the Linux container.

Optional: `RUN_SWIFT_TEST=1 ./build_app.sh` first runs the Swift unit tests
(trigger policy, ledger persistence, erase). Signing: ad-hoc by default;
set `CODESIGN_IDENTITY="Developer ID Application: …"` if you want your
identity baked in.

## Install & run

1. Drag `RetrainOrchestrator.app` into `/Applications`.
2. Open it — it appears in the menu bar (no Dock icon).
3. Notifications ask for permission on first launch; failures are pinned,
   other categories are toggled in Settings.

Minimum macOS: **13.0 (Ventura)** — the login-item service (`SMAppService`)
requires it. If System Settings → Login Items shows the app as allowed in
the background, you may need to approve it there once.

## Where your data lives (and nowhere else)

Everything the app writes sits in the sandbox container dir:

```
~/Library/Containers/com.lohnn.retrain-orchestrator/Data/Library/
└── Application Support/RetrainOrchestrator/
    ├── state.json          # ledger + settings (atomic writes)
    └── orchestrator.log    # the log that "Open log" opens
```

- **Erase all data…** empties exactly that directory (ask-then-wipe; the
  confirm dialog shows the resolved path and "N file(s) / X byte(s) would
  be removed", and fresh defaults are rewritten after the wipe).
- **Uninstall** = drag the app to Trash; delete the container dir above if
  you want zero residue. The login-item registration dissolves when the
  app is removed (macOS cleans orphaned entries; nothing else is global —
  no `/usr/local`, no brew, no daemon files outside the container).
- **Entitlements** (`retrain-orchestrator.entitlements`): `app-sandbox` +
  outbound-network client only (P2's pre-flight `/health` and mirror POST
  need it). No file grants beyond the container.

### If the data dir looks empty (diagnostic of record, 2026-10-28)

The app writes `state.json` + `orchestrator.log` **unconditionally at every
launch** — an empty data dir after a launch is NOT expected and means the
app is writing elsewhere (most likely the ad-hoc signature embedded no
sandbox entitlements, so the app runs unsandboxed and lands in
`~/Library/Application Support/RetrainOrchestrator/`). Capture these facts
and send them back:

```bash
ls -la "$HOME/Library/Containers/com.lohnn.retrain-orchestrator/Data/Library/Application Support/RetrainOrchestrator/" 2>&1
ls -la "$HOME/Library/Application Support/RetrainOrchestrator/" 2>&1
find "$HOME/Library/Containers/com.lohnn.retrain-orchestrator" -type f 2>/dev/null | head -20
codesign -d --entitlements - /Applications/RetrainOrchestrator.app 2>&1
```

## Checklist to verify P1 (paste results back)

1. App launches, menu bar only.
2. Menu contains Run now / Settings… / Open log / Erase all data… / Quit.
3. Settings: change threshold to 42 → Save → Quit → relaunch → still 42.
4. Open log opens a readable `orchestrator.log` inside the container dir.
5. Erase all data → confirm → wipe count in the dialog/log matches; fresh
   default `state.json` rewritten. (Precondition: two expected files exist
   beforehand — see "If the data dir looks empty" above if not.)
6. "Run now" writes a trigger-policy evaluation line to the log (and, if
   notifications are allowed, posts the skip/fire verdict).
7. Settings › Classifier connection: press **Test connection** with the
   default `http://jarvis:8008` (Tailscale up) → menu-bar dot turns green
   and `/health`'s truth (status + checks) renders in the window; change
   the port to something dead → dot turns red and the reachable/probe
   verdict flips.
8. Exit code and paste of any build errors from `./build_app.sh`.

## Repo layout

```
retrain-orchestrator/
├── Package.swift                     # SwiftPM: RetrainKit + app + tests
├── build_app.sh                      # THE Mac-side build (the one loop)
├── retrain-orchestrator.entitlements
├── Sources/RetrainKit/               # pure logic (unit-testable): ledger,
│   │                                 # trigger policy, log, erase
│   ├── AppState.swift                #   state.json model + atomic Ledger
│   ├── TriggerPolicy.swift           #   locked §4 policy, pure evaluation
│   ├── RunLog.swift                  #   container-dir log
│   └── Erase.swift                   #   the wipe primitive
├── Sources/RetrainOrchestrator/      # AppKit surface
│   ├── main.swift                    #   menu-bar entry (accessory app)
│   ├── AppDelegate.swift             #   status item + menu + actions
│   ├── SettingsWindowController.swift#   the Settings window
│   ├── LoginItem.swift               #   SMAppService toggle
│   └── Notifier.swift                #   UNUserNotificationCenter + pinning
└── Tests/RetrainKitTests/            # swift test (opt-in via RUN_SWIFT_TEST=1)
```
