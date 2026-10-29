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
| Install | drag `.app` into `/Applications`; first run bootstraps its sandboxed runtime (uv-managed python + PyTorch-MPS wheels, portable Metal `llama-server` binary, HTTP training snapshot from the classifier — no git, no PAT) **inside `~/Library/Containers/<app-id>/`** |
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

Superseded 2026-10-29 (§9, "git ownership"): the Mac holds **no GitHub PAT,
no git client, opens no PR** — the classifier (Pi) owns git. The trainer
receives code+data as an HTTP snapshot and returns artifacts over
peer-only HTTP; the Pi does all git via its existing proven paths.

```
pre-flight  GET {pi}/health         → unreachable ⇒ push "skipped: classifier
                                      unreachable" event, exit clean (retry
                                      on the next trigger; no retry storm)
1. snapshot GET /admin/training-snapshot (peer-only) → unpack into the
            container dir: deployed brain code (config.py, trainers) +
            training-data corpus + the validated §16 splits. Training-side
            consistency BY CONSTRUCTION: the Mac trains with the same
            config.py/input formatting the deployment runs.
2. train    venv python  (embedded runtime; PyTorch MPS, device logged)
            LoRA on the EmbeddingGemma text tower. The in-repo trainer is
            a P2 artifact (none exists in-repo today — the Oct-13 finetune
            tooling lived outside the repo).
3. MERGE    human-in-the-loop CHECKPOINT until P3 (Mode A decision, §9):
            the run pauses, mirrors "run finished (merge required)" +
            pinned notification with instructions; the operator merges the
            LoRA into the merged GGUF by hand. P3 automates exactly this
            step. (The merge itself is llama.cpp CPU tooling — proven once,
            Oct 13.)
4. embed    llama-server (local, Metal) on the MERGED GGUF
            → build_label_centroids.py (validates FINDINGS §16 or aborts)
5. ship GGUF   sha-pinned POST /admin/upload-model-artifact — STAGED ONLY
               (activated in P4 together with the vocabulary pair).
6. ship vocab  POST /admin/receive-centroids (peer-only) → the Pi validates
               and commits label_centroids.json on branch vocab/<run_id>,
               pushes it (human-visible on GitHub; main untouched). INERT
               until P4 deploys the GGUF+vocabulary PAIR — activation
               ordering is a P4 design item; P2 never changes the running
               classifier.
7. mirror      POST /admin/retrain-events per step + notifications per
               category + per-run log line
```

**Autonomy: fully automatic (operator decision).** Steps 1–2 and 4–7 run
unattended; step 3 is the deliberate human checkpoint until P3. Every step
has a defined failure path (abort the pipeline — never deploy or stage a
half-state — notify + mirror the failure). The Settings dial
(`ask-before-run` → `auto-train + gated ship` → `full automatic`) is kept for
reversibility, defaulting to `full automatic`. First P2 cut ships behind
**Run now only**; the unattended timer lands as its own commit after the
operator validates one full manual run.

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
- **GitHub: the Mac app holds no PAT, no git client, opens no PR**
  (2026-10-29, the bucket-mirror §7 rule generalized to git). The
  classifier owns all git: training data and code reach the trainer as an
  HTTP snapshot; the vocabulary returns as an HTTP POST that the Pi
  commits and pushes on a `vocab/<run_id>` branch (existing proven Pi push
  path — verified 2026-10-29, no new grants needed).
- Trust root: **Tailscale membership** — the new snapshot/centroids
  endpoints sit behind the same Tailscale/localhost-peer-only wall as the
  model-artifact routes; the Pi stays keyless via the CGNAT bypass
  (unchanged).
- GDrive: the trainer needs nothing; the Pi's existing `drive.file` grant is
  used for the bucket-mirror (§7) — **no new grants, no widened scopes** (the
  Oct-27 lesson: re-route artifacts instead of widening grants).
- Git-tracked vocabulary: human-visible on GitHub as `vocab/<run_id>`
  branches (was: PAT-opened PRs — superseded).
- The expiring-SSH access model (used incidentally during the diagnosis arc)
  is not required by this design.

## 9. Decision log (operator, 2026-10-27/28 + 2026-10-29)

| # | Question | Decision |
|---|---|---|
| Q1 | Trigger | **≥ 10 confirmed corrections OR 7 days (only if ≥ 1 new correction); zero-corrections ⇒ skip + "no new training data" event** (later refined from an initial 5). All values Settings-editable. |
| Q2 | Observability | **Native notifications + always-on log + classifier status mirror** (`/admin/retrain-events`). |
| Q3 | Autonomy | **(c) Fully automatic** — runs unattended on an always-on, usually-headless Mac; notifications informational. Dial kept for reversibility. |
| + | Bucket | **The classifier mirrors received model updates into the GDrive store** — trainer needs no rclone. |
| + | Pre-flight | **`/health` check before every run; unreachable ⇒ skip + event, no retry storm.** |
| + | v1 container design | **Discarded.** Native app replaces the split-brain container; ends the GPU-constraint workaround. |
| + | 2026-10-29 | **Model-id default classifier address — none** (setup-specific); URL starts empty and Settings *is* the onboarding: it auto-opens every launch until the first successful Save. Port 8008 remains placeholder/docs guidance. |
| + | 2026-10-29 | **Bundle id domain `se.lohnn.*`** (was `com.lohnn.*`). Container dir follows the id; migration = fresh container, old dir deletable. |
| + | 2026-10-29 | **§11-Q2 P2 mode = A: full LoRA-on-MPS pipeline** with the GGUF merge as a human-in-the-loop checkpoint until P3 (SetFit quick-mode dropped from P2; SetFit remains the rollback-backend story). Confirmed reasoning: the merge is llama.cpp CPU tooling proven once (Oct 13) — P3 is automation engineering, not a feasibility bet. |
| + | 2026-10-29 | **First-run bootstrap download consented** (~2–3 GB: uv-managed python + PyTorch-MPS wheels + Metal llama.cpp server, all inside the container dir; Erase-all-data removes it all — the "installs itself" contract stands). |
| + | 2026-10-29 | **First P2 cut ships behind Run now only**; the unattended timer is a follow-up commit after the operator validates one full manual run — §9-Q3 full-automatic then governs unchanged. |
| + | 2026-10-29 | **Git ownership (bucket-mirror §7 rule generalized): the Mac holds no PAT, no git client, opens no PR.** The classifier owns git: code+data reach the trainer as `GET /admin/training-snapshot`; vocabulary returns as `POST /admin/receive-centroids` (Pi validates, commits `label_centroids.json` on `vocab/<run_id>`, pushes). Trust root = Tailscale membership; both endpoints peer-only. |
| + | 2026-10-29 | **Menu-bar status dot** (operator report: not visible) — reimplemented in P2 as a composited rendered image; menubar text colors are not honored. |
| + | 2026-10-29 | **P1 runtime acceptance validated; status-bar dot rework + full pipeline deferred to P2** (operator). |

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
2. **DECIDED 2026-10-29 (operator, via §9):** P2 mode = **A** — the full
   LoRA-on-MPS path with the GGUF merge as a human-in-the-loop checkpoint
   until P3. The SetFit quick-mode (B) is dropped from P2; SetFit remains
   the rollback-backend story.
