# Design — Retraining Orchestrator (split-brain: container clock, Mac GPU)

**Status:** PROPOSAL — no build started. Approve → implement per §9.
**Companion doc:** [RETRAIN-AND-DEPLOY.md](../email_classifier_brain/RETRAIN-AND-DEPLOY.md)
**Origin:** user wish — *"a durable and transferable method that I just do
`docker compose up`, and then it keeps track of time for automatic retraining
and runs the flow"* — plus the 2026-10-27 platform-constraint analysis.

---

## 1. Goal

One command (`docker compose up -d`) starts a permanently-running retraining
pipeline for the email classifier that:

- keeps its own clock (schedule + retrain-worthiness triggers),
- tracks state across restarts (a ledger, not a log),
- runs the GPU-heavy training on the Mac host with **native Metal (MPS)** speed,
- ships results (GGUF + centroids → Pi; vocabulary → repo PR) using the exact
  mechanisms proven in the WI-103 arc,
- returns to idle and never requires a human except to approve the PR it opens.

**Non-goals:** distributed/multi-host training; CUDA/NVIDIA (none present);
auto-merge of the retrain PR (human review stays); touch the classifier's
runtime data (`storage/`) — read-only awareness.

## 2. Platform constraint (why split-brain)

GPU access from Linux containers on **macOS is impossible in 2026, on both
runtimes**: Docker Desktop's Linux VM is blind to Metal (Hypervisor.framework
limitation), and Apple's native `container` runtime has no GPU passthrough
([`apple/container` #1511](https://github.com/apple/container/issues/1511),
open May 2026; earlier [discussion #62](https://github.com/apple/container/discussions/62)).
Mac-native PyTorch acceleration is MPS, host-side only
([Apple: Accelerated PyTorch on Mac](https://developer.apple.com/metal/pytorch/)).

Therefore: **everything except the GPU-touching step lives in the container;
the GPU step runs natively on the Mac host** and is bridged over SSH. This is
the same split the Pi already embodies (orchestrator brain + remote actuator)
and it reuses the proven expiring-key circulation pattern.

The host GPU step is itself Metal-accelerated: `llama.cpp` on Apple Silicon
uses Metal by default (brew install), so both `train.py` (PyTorch MPS) and the
centroid embedding pass run on the GPU.

## 3. Architecture

```
┌─────────────────────────────── Mac (studio) ───────────────────────────────┐
│  ┌──────────────────────────────┐     ┌─────────────────────────────────┐  │
│  │ orchestrator container       │     │ host (native macOS, MPS/Metal)  │  │
│  │ (docker compose)             │     │                                 │  │
│  │  • clock / trigger policy    │ SSH │  retrain-runner.sh:             │  │
│  │  • ledger (state.json)       ├────►│   pull data repo                │  │
│  │  • Pi-side verify + ship     │ key │   train.py  (PyTorch MPS)       │  │
│  │  • HTTP → Pi (keyless,.CG)   │     │   llama-server (local, Metal)   │  │
│  │  • artifact dir (bind mount) │◄────┤   build_label_centroids.py      │  │
│  └──────────────┬───────────────┘mount│   → artifacts + manifest        │  │
│                 │                    └─────────────────────────────────┘  │
│        bind dir: ~/jarvis-orchestrator/{state,artifacts,logs}               │
└──────────────┬──────────────────────────────────────────────────────────────┘
               │ Tailscale (trusted CGNAT → keyless admin)
        ┌──────▼──────────────┐
        │ Pi (jarvis :8008)   │  upload/install GGUF → trigger-update → sweep-state
        └─────────────────────┘
```

**The bind-mount channel** is the honest bridge: on macOS the `artifacts/`
directory is a real Mac folder; the host runner drops artifacts + a
`manifest.json` (sha256s) into it; the container picks them up — no network
copy between halves on the same machine. If the orchestrator later runs on a
*non*-Mac Docker host, the SSH/scp flag replaces the mount (.flag in config).

## 4. Components (proposed tree)

```
retrain-orchestrator/
  docker-compose.yml          # one service, bind mounts, restart: unless-stopped
  Dockerfile                  # python:3.13-slim + openssh-client + deps
  orchestrator/
    config.py                 # env + config.yaml: schedule, thresholds, paths, SSH
    trigger.py                # retrain-worthiness policy (§6)
    ledger.py                 # state.json read/append (atomic write + fsync)
    runner_bridge.py          # SSH exec to host runner (expiring key aware)
    shipper.py                # sha-verify artifacts → Pi upload/install → PR body
    verify.py                 # Pi sweep-state + /health + update-log SHA checks
    main.py                   # APScheduler loop + POST /run-now + GET /status
  tests/                      # pytest: trigger policy, ledger, shipper fakes
mac-runner/
  retrain-runner.sh           # the ONLY host-executed script (§5)
  bootstrap.sh                # one-time: host venv, rclone config check, llama.cpp
                              #   (brew install llama.cpp), data-repo clone check
  launchd/                    # optional no-Docker variant (template plist)
docs/RETRAIN-ORCHESTRATOR-DESIGN.md   # this file
```

Deps (container): `apscheduler`, `httpx`, `pyyaml`, `paramiko` **or** system
ssh (prefer system ssh — expiring-key UX identical to the Pi flow). No secrets
in the image; everything via mounted config + key file.

## 5. The host runner (the GPU step — single file, single venv)

`retrain-runner.sh` executes ONLY on the Mac host:

1. `git -C ../email_classifier_data pull` (training data).
2. `venv/bin/python email_classifier_brain/train.py` — PyTorch MPS
   (`device = mps if torch.backends.mps.is_available() else cpu`; log device).
3. Start a local `llama-server` (`llama.cpp`, Metal) serving the **fresh merged
   GGUF** this train produced, then
   `python build_label_centroids.py --endpoint http://127.0.0.1:4503/...`
   (validates FINDINGS §16 dev/test or aborts), stop the server.
4. Write `artifacts/{merged_text_q8_0.gguf?, label_centroids.json,
   MODEL_INFO.json, manifest.json}` — manifest carries sha256 + provenance
   (data-repo SHA, train args, model eval JSON).
5. Append a run report line to `logs/runner.log` (the orchestrator's ledger
   mirrors it).

Bootstrap (`bootstrap.sh`, run once): creates `~/jarvis-orchestrator/venv`,
installs torch (MPS build), llama.cpp, clones/contrôles the data repo, prints
the SSH public key / adds the authorized_keys line handed to it (same
expiry-time pattern as the Pi).

**Trim option:** if the LoRA/GGUF merge isn't yet automatable, phase 1 runner
mode = `--stonly`: train the SetFit path + rebuild centroids against the
**serving** GGUF on the Pi via a port-forward (`ssh -L 4503` or Tailscale
direct) — no GGUF shipping; still a quality-progression retrain. The design
does not depend on that being permanent.

## 6. Trigger policy (the clock)

A retrain is warranted when ANY holds (all thresholds in `config.yaml`):

- **corrections**: ≥ `min_new_confirmed` (default 10) new confirmed corrections
  since `ledger.last_train.corrections_total` (source: classifier
  `/admin/retrain-status` + notifications deltas);
- **calendar**: `cron` fallback (default: none — corrections-driven only, per
  the user's "keeps track of time" = the SCHEDULER keeps time, not a blind
  cadence);
- **manual**: `curl -X POST http://localhost:8080/run-now` (orchestrator's
  local-only endpoint);
- **veto watches** (before firing, abort + log if): classifier `/health` not ok,
  sweep-state vocabulary empty (an already-sick system isn't retrained on top
  of), another job still running (ledger lockfile).

After firing, the orchestrator: waits for runner completion (SSH poll), verifies
manifest shas, opens the **centroids PR** (task documented in
RETRAIN-AND-DEPLOY.md §3.3-3.4), uploads GGUF sha-pinned → install → reload,
fires `trigger-update`, re-reads `sweep-state` across a cycle, and appends the
verdict to the ledger. Human touches remain exactly two: merge the PR, glance
at the report.

## 7. Ledger (`state.json` — append + snapshot)

```
{ "last_train":  { "at", "data_sha", "model_eval": {dev_acc, test_acc}, "artifacts": {...} },
  "corrections_total_at_train": N,
  "deploys":      [ { "at", "sha", "verified": bool, "sweep_state_after": {...} } ],
  "runs":         [ { "started", "finished", "trigger": "manual|policy", "status", "error": null } ],
  "lock":         null | {"since", "reason"} }
```

Write policy: temp file + rename + fsync (the file is the truth across reboots;
loss = re-derive from Pi + git, both self-healing).

## 8. Security posture

- Container holds: the compose stack, an expiring SSH **private key** for
  studio (authorized_keys `expiry-time` line, same pattern as jarvis access),
  and a read-only view of nothing else. No ADMIN_API_KEY needed anywhere
  (Tailscale CGNAT bypass carries it — same as the coordinator).
- Host runner: plain script, no sudo, only touches its own dirs + the data repo.
- Pi: bounded by the same trust we already granted the coordinator (drive.file
  bucket posture stays untouched; nothing requires widening it).
- Revocation = delete one authorized_keys line (host) — mirrored from the
  Proven Pi pattern.

## 9. Implementation plan (on approval)

| Phase | Deliverable | Test |
|---|---|---|
| P1 skeleton | compose + orchestrator + ledger + `/status` | unit tests, compose up, no SSH |
| P2 bridge | runner bootstrap + SSH exec, `--dry-run` mode (runner echoes plan) | manual dry-run on studio |
| P3 st-only live | runner `--stonly` (SetFit retrain + centroids vs served GGUF) | one real run + Pi trigger |
| P4 full LoRA | GGUF merge + upload/install + reload path alignment | one full cycle end-to-end |
| P5 hardening | veto watches, failure replay, docs | chaos drill (kill mid-run) |

Estimated: P1–P3 a session; P4 depends on the LoRA-merge automation gap
(out-of-repo today — see RETRAIN-AND-DEPLOY.md §3).

## 10. Open questions

1. Trigger threshold: 10 corrections a sensible default?
2. Alerts: log-only, or a webhook/notification into the classifier's own UI?
3. May the orchestrator run unattended on schedule, or manual-only until
   confidence builds? (P3 default proposal: manual-only; schedule armed at P4.)
