# Design — Native Retraining App + Status Mirror (v2)

**Status:** PROPOSAL v2 — decisions locked in a Q&A with the operator on
2026-10-27/28 (§9 Decision log). No build started. **v2 replaces the
split-brain container design entirely**: the orchestrator is a native,
sandboxed macOS menu-bar application ("the app").

**Companion doc:** [RETRAIN-AND-DEPLOY.md](../email_classifier_brain/RETRAIN-AND-DEPLOY.md)
**Origin:** the operator's wish for a durable, transferable, hands-off
retraining loop — iterated v1 (docker compose split-brain) →
operator pivot to a native always-on app.

---

## 1. Goal

A native macOS app in `/Applications` that, once running (optionally
"start with the system"), operates the full email-classifier retraining
pipeline automatically:

- keeps its own clock and trigger policy (§4),
- trains locally with the Mac's **MPS/Metal** GPU (no container can do this —
  platform constraint of record, see §2),
- ships results to the classifier service **and mirrors them into the GDrive
  model store** so no trainer-side rclone/sync setup is ever needed (§7),
- reports everything: native notifications + log + a **status mirror** on the
  classifier so any future session (or human) can ask *"what's the status of
  the last retraining?"* and get time + status over the network (§6).

**Non-goals:** distributed/multi-host training; CUDA/NVIDIA (none present);
human review gates (operator chose **fully automatic**, §9); touching the
classifier's runtime data (`storage/`) — read-only awareness.

## 2. Platform constraint (why native)

GPU access from Linux containers on **macOS is impossible in 2026, on both
runtimes**: Docker Desktop's Linux VM is blind to Metal (Hypervisor.framework
limitation), and Apple's native `container` runtime has no GPU passthrough
([`apple/container` #1511](https://github.com/apple/container/issues/1511),
open May 2026; earlier [discussion #62](https://github.com/apple/container/discussions/62)).
Mac-native PyTorch acceleration is MPS, host-side only
([Apple: Accelerated PyTorch on Mac](https://developer.apple.com/metal/pytorch/)).

A sandboxed menu-bar app gets the GPU the honest way and dissolves the
container/host bridge completely. The GPU steps are themselves Metal-backed:
`train.py` runs on PyTorch **MPS**; the centroid builder embeds through a
local `llama-server` whose llama.cpp build uses **Metal** by default on
Apple Silicon.

## 3. The app contract

| Property | Behavior |
|---|---|
| Install | drag `.app` into `/Applications`; first run bootstraps its sandboxed runtime (embedded python runtime + torch MPS build, portable `llama-server` binary, data-repo clone) **inside `~/Library/Containers/<app-id>/`** |
| Always running | menu-bar app (no dock icon by default); **"Start with the system"** toggle uses standard login-item services; runs headless (no screen attached is fine — status is queryable over the network, notifications arrive whenever a screen is) |
| Status bar | icon with state (idle / health-checking / training / shipping / last-run time); menu: *Run now*, *Settings…*, *Open log*, *Erase all data…*, *Quit* |
| Settings window | trigger threshold, weekly-cadence toggle + interval, autonomy mode, notification categories — everything §4/§6 promises, operator-editable **as the classifier matures** (explicit requirement) |
| Uninstall | drag `.app` to Trash; the app is **sandboxed**, so its footprint is the container dir `~/Library/Containers/<app-id>/`; **"Erase all data…"** performs the guaranteed wipe. No global pollution: no `/usr/local`, no brew dependency (the `llama-server` binary downloads into the container dir) |
| GPU | PyTorch **MPS** for training; Metal `llama-server` for embedding/centroids; device log line at every step (`mps` verified) |

## 4. Trigger policy (decisions locked)

The app re-runs the pipeline when:

1. **≥ `corrections_threshold` (default 10, Settings-editable)** new confirmed
   corrections since `ledger.last_train.count`, **OR**
2. **weekly cadence**: 7 days since `last_train` **AND ≥ 1 new confirmed
   correction** (a week with fresh data must not be ignored).

**Zero-corrections rule (operator-set):** if no new confirmed corrections
exist at all, the pipeline **does not run** — instead it pushes a
`skipped: no new training data` event to the classifier's status mirror (§6)
and waits for the next cycle. All values live in the **Settings window** (the
operator wants to re-tune them as the classifier matures).

## 5. Pipeline (per retrain run)

```
pre-flight  GET {pi}/health         → unreachable ⇒ push "skipped: classifier
                                      unreachable" event, exit clean (retry
                                      on the next trigger; no retry storm)
1. pull     data repo (git, HTTPS + Keychain-stored PAT)
2. train    venv python train.py             (PyTorch MPS; device logged)
3. embed    llama-server (local, Metal) on the NEW merged GGUF
            → build_label_centroids.py       (validates FINDINGS §16 or aborts)
4. ship GGUF     sha-pinned POST /admin/upload-model-artifact → install
                 (the GGUF merge step itself is the known P4 automation gap —
                 RETRAIN-AND-DEPLOY.md §3)
5. ship vocab    open centroids PR (repo-tracked path) via Keychain PAT
6. deploy        POST /admin/trigger-update  → verify end-state
                 (journal SHA, sweep-state across a cycle)
7. mirror        POST /admin/retrain-events  → event + ledger snapshot (§6)
   notify        native macOS notifications per category + log line
```

**Autonomy: fully automatic (operator decision).** All seven steps run
unattended; every step has a defined failure path (abort the pipeline — never
deploy a half-state — notify + mirror the failure). The Settings dial
(`ask-before-run` → `auto-train + gated ship` → `full automatic`) is kept for
reversibility, defaulting to `full automatic`.

Pre-flight detail: if the classifier is unreachable, the run is skipped (not
deferred-by-retry-storm) — the skip itself is mirrored so the operator can see
*why* nothing ran. A training run while the classifier is down would produce
artifacts nobody can accept; the health check makes "the target must exist"
an explicit precondition.

## 6. Observability (decisions locked)

- **Native notifications**, per-category switches in Settings: run started /
  finished / failed / PR opened / deploy verified / skipped. Failures are
  pinned (delivered even when other categories are muted + highlighted in the
  log window).
- **Log**: always-on, structured, one file per run inside the container dir.
- **Status mirror on the classifier** (the operator's explicit request — so
  the operator, the deploy automation, or a fresh session can later ask
  *"what's the status of the last retraining?"* and get time + status over HTTP):
  `POST /admin/retrain-events` records
  `{event, at, run_id, status, eval, data_sha, artifact_shas, durations,
  error}`, and `GET /admin/retrain-events` serves latest + history. The app is
  the source of truth; the classifier holds the queryable mirror that persists
  when the Mac is offline. Storage side (decided with the operator
  2026-10-28, see Q1): `retrain_events.json` JSONL in the service workdir
  — event-per-line, ~50 newest kept, GET shape fixed as {latest, history}.

## 7. The bucket-mirror rule (operator decision — replaces trainer-side rclone)

**When the trainer pushes a model update to the classifier, the CLASSIFIER
updates the GDrive store.** Concretely: after a successful GGUF install, the
Pi (which now holds the received payload) pushes it into the model bucket
with its own rclone grant:

```
rclone copy llama_model/merged_text_q8_0.gguf gdrive:email-classifier-model/
```

- This is **additive** (`copy`, never `sync`).
- Because the Pi's grant is `drive.file`-scoped, the Pi-created copy is
  immediately visible to the Pi itself — this also *gradually heals* the
  bucket visibility asymmetry (the Oct-27 vortex lesson).
- The trainer (Mac) therefore needs **no rclone and no bucket access at all**.
- Non-fatal on failure: mirrored + logged; the vocabulary file remains
  git-owned (PR-carried), the GGUF remains install-endpoint-delivered.
- The existing `model/` mirror-sync exclusion (#94/#95) is unchanged — the
  bucket-mirror writes target `email-classifier-model/`, never the mirrored
  `model/` dir conflict domain.

## 8. Security posture

- Sandboxed app; container-dir-only footprint; portable binaries inside it.
- GitHub PAT: macOS Keychain.
- Pi: keyless via Tailscale CGNAT bypass (unchanged).
- GDrive: the trainer needs nothing; the Pi's existing `drive.file` grant is
  used for the bucket-mirror (§7) — **no new grants, no widened scopes** (the
  Oct-27 lesson: re-route artifacts instead of widening grants).
- Git-tracked vocabulary PRs: human-visible on GitHub, auto-opened via PAT.
- The expiring-SSH access model (used incidentally during the diagnosis arc)
  is not required by this design.

## 9. Decision log (operator, 2026-10-27/28)

| # | Question | Decision |
|---|---|---|
| Q1 | Trigger | **≥ 10 confirmed corrections OR 7 days (only if ≥ 1 new correction); zero-corrections ⇒ skip + "no new training data" event** (later refined from an initial 5). All values Settings-editable. |
| Q2 | Observability | **Native notifications + always-on log + classifier status mirror** (`/admin/retrain-events`). |
| Q3 | Autonomy | **(c) Fully automatic** — runs unattended on an always-on, usually-headless Mac; notifications informational. Dial kept for reversibility. |
| + | Bucket | **The classifier mirrors received model updates into the GDrive store** — trainer needs no rclone. |
| + | Pre-flight | **`/health` check before every run; unreachable ⇒ skip + event, no retry storm.** |
| + | v1 container design | **Discarded.** Native app replaces the split-brain container; ends the GPU-constraint workaround. |

## 10. Implementation plan

| Phase | Deliverable | Dependencies |
|---|---|---|
| P1 | Swift menu-bar app skeleton (sandbox, menu, Settings, ledger, login-item toggle, notifications) + **classifier `/admin/retrain-events` POST+GET** | Mac-side build (CLT only, no full Xcode); coordinator authors source, operator runs one build script |
| P2 | Runner pipeline unattended: data pull → train (MPS) → centroids (local Metal llama-server) → status mirror + pre-flight | P1 + embedded runtime bootstrap + `llama-server` download |
| P3 | GGUF merge automation (the current gap) — evaluate whether the llama.cpp finetune/merge path is tractable on Metal, else keep manual-merge + pipeline pickup | P2 |
| P4 | Ship path unattended: upload/install → reload → trigger-update → acceptance reads + status mirror + bucket-mirror (§7) | P2/P3 + #95/#96 already live |
| P5 | Hardening: retry policy, skip-storm protection, log rotation, erase-and-quit correctness | P4 |

Build-loop reality: the coordinator cannot compile Swift (Linux container).
The operator runs one build script on the Mac; errors paste back; iterate.
Keep the Swift surface deliberately small (menu + subprocess orchestration).

## 11. Open questions (small, for P1)

1. **DECIDED 2026-10-28 (with the operator):** classifier-side event
   storage = `retrain_events.json` JSONL in the service workdir,
   event-per-line, keep the ~50 newest, GET shape {latest, history}; one
   human-readable `update_history.json` line per recorded event. Not
   SQLite; not update_history.json-only.
2. Whether the SetFit-path quality retrain (no GGUF merge) ships at P2 as a
   standalone "vocabulary refresh + eval progression" mode while P3 is open.
