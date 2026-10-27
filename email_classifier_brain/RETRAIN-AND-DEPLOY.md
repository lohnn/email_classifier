# Retrain & System-Update Runbook

Operational runbook for the jarvis email-classifier deployment: how to retrain
(train → centroids → artifacts), ship a new model (GGUF + vocabulary), and roll
a code update onto the Pi — including the pitfalls that cost real debugging
time in Oct 2026 (see PRs #90–#96).

Audience: a future agent session, or a human, doing any of:
(a) a LoRA/GGUF retrain + rollout, (b) a code deploy, (c) recovering a
misbehaving deployment.

---

## 1. System topology

| Piece | Where | Notes |
|---|---|---|
| Classifier service | Pi `jarvis` (Tailscale; last-known IP `100.79.150.106`), `/home/lohnn/email_classifier`, `email_classifier_brain/` | uvicorn `:8008`, brain venv (py3.14), user `lohnn` |
| systemd units | `email-classifier.service` (runs `run_service.sh`), `email-embedder.service` (static `llama-server`, `:8090`, loopback), `email-embedder-reload.path/.service` (GGUF-change reload) | installed by `bootstrap_llama_cpp.sh` with paths baked in — change all three together (see unit file NOTE) |
| Trainer / dev box | this container (`/workspace/projects/email_classifier`) | has venv + full test suite; **no rclone**; `retrain.sh` needs a rclone-capable box |
| User's Mac | rclone with `gdrive` remote (full Drive scope), web browser for OAuth | the only place that can (re)fill the model bucket |
| Deploy operator workstation | this workspace | keyless admin over Tailscale, clamped rclone remote, expiring-SSH access |

Runtime env on the Pi (`.env`, not in repo): `EMBEDDER_BACKEND=llama_cpp`,
`LLAMA_ENDPOINT=http://127.0.0.1:8090`, `LLAMA_TIMEOUT_SECONDS=180` (Pi A72
needs it; default 30 s aborts long emails — b34ffd8),
`CENTROIDS_PATH=model/label_centroids.json`, `MODEL_DIR=model`,
`TRAINING_DATA_DIR=../email_classifier_data`,
`GDRIVE_REMOTE=gdrive`, `GDRIVE_MODEL_PATH=email-classifier-model`,
`GDRIVE_STORAGE_PATH=email-classifier-storage`, `LLAMA_MODEL_FILE=llama_model/merged_text_q8_0.gguf`.

## 2. Model architecture current state (2026-10)

- **Production backend**: `llama_cpp` — `llama-server` serves a single merged
  Q8 GGUF (`merged_text_q8_0.gguf`, ~300 MB); classification happens per-email
  via the endpoint; the **vocabulary** (category list, UNSURE calibration)
  comes from `model/label_centroids.json` (`get_available_categories()`),
  built by `build_label_centroids.py` against the **exact GGUF being served**.
- **Rollback backend**: `st` (SetFit in-process; reads `model/label_mapping.json`
  … the SetFit artifacts that live in the GDrive bucket). Empty fallback on the
  llama path is a **trap**: `_label_mapping` is empty unless the `st` stack ran.
- **Vocabulary file**: `email_classifier_brain/model/label_centroids.json` is
  **repo-tracked (PR #94)** — do not edit by hand; refresh it via
  `build_label_centroids.py` in the same PR as any model/code change (README
  carries the same rule). **UNSURE gate calibration** ships inside the same file.

## 3. Retrain flow (LoRA/GGUF)

What is scripted: `retrain.sh` (trainer-side: pull data repo → `train.py` →
`rclone sync $MODEL_DIR/ gdrive:email-classifier-model/` → commit+push data).
What is **not** scripted in-repo: the LoRA→GGUF merge step for the llama_cpp
path (the current GGUF provenance lives in `MODEL_INFO.json` and the
`retrain_2026-10-13/` + `retrained_model_2026-10-13/` working dirs; PR #87
notes the llama-seam decision). Sequence for a fresh LoRA rollout:

1. **Data**: training data is a *private git data repo* (sibling clone,
   `../email_classifier_data`). Corrections arrive via the service
   (`/admin/push-training-data`, auto-daily update job, or
   `/admin/check-corrections`). On the trainer box: `./retrain.sh` handles
   pull → train → bucket sync → data push. **This box cannot run it** (no
   rclone) — run it on the Mac or add rclone here first.
2. **Centroids + UNSURE calibration**: serve the new merged GGUF on a local
   `llama-server` (`:4503` lab default) and run
   `uv run python build_label_centroids.py` with the target `--out`. It
   validates against FINDINGS §16 (dev 0.900 / test 0.889) or aborts. The
   buckets it embeds come from the data repo's `Jarvis/` jsonl splits.
3. **Ship the vocabulary**: commit the new
   `email_classifier_brain/model/label_centroids.json` (byte-for-byte, sha
   noted in the PR) **in the same PR as any accompanying code change**. This
   is what #94/#95 made wipe-proof: tracked in git, excluded from the为主
   GDrive syncs.
4. **Ship the GGUF**: `POST /admin/upload-model-artifact?filename=…&sha256_expected=…`
   then `POST /admin/install-model-artifact?staged_name=…` (Tailscale/localhost
   peers only — valid keys from untrusted networks are **rejected** because a
   .pkl/GGUF executes on use; 2 GiB staging cap). **Path alignment rule**: the
   GGUF install target (`LLAMA_MODEL_FILE`), `email-embedder.service`'s `-m`,
   and `email-embedder-reload.path`'s `PathChanged` must all agree — the shipped
   unit currently watches `/home/email_bot/models/…` while the code default is
   `llama_model/…`; **check which is live on the Pi before shipping** and align
   all three (bootstrap script bakes them together). The reload path unit
   restarts the embedder on change; otherwise restart
   `email-embedder.service` manually.
5. **Verify**: `/health` → `model: llama_cpp ok`; one real classification per
   category; `GET /admin/sweep-state` → `trained_categories.count == expected`,
   `centroids_exists=true`, `centroids_entries == category count`.

## 4. Code deploy flow (PR → Pi)

1. PR against `main` → merge (true merges were used for #90–#96; pull handles
   any shape).
2. Trigger from a Tailscale-scoped box (a Tailscale-joined workstation
   qualifies):
   ```
   curl -X POST 'http://100.79.150.106:8008/admin/trigger-update'
   ```
   No `X-API-Key` needed — `api/security.py` trusts the Tailnet CGNAT range.
3. What the wrapper does now (PR #95 semantics — read before editing it):
   - `rclone sync gdrive:email-classifier-model/ → model/ --exclude label_centroids.json`
     (the exclude is load-bearing: the Pi's rclone grant is **drive.file-scoped**
     and its bucket view holds only files it created itself — a plain mirror
     would delete the git-owned vocabulary on every run; this specific
     arrangement was the Oct 27 outage),
   - `git pull`, `pip install -r requirements.txt`,
   - 30 s health verify on `:8008` (success: restart; failure: `git reset --hard`
     rollback),
   - post-verify: "Model config.json missing → sync" repair branch — it
     re-fires on every startup while `config.json` is invisible to the drive.file
     grant; this is known and now harmless to the vocabulary file.
4. **Verify the end state, not the POST**: `journalctl -u email-classifier` →
   `Update verified (Health check passed)` + `Update to <SHA> successful`
   (the SHA is the version certificate); `GET /admin/sweep-state` for the
   category/vocabulary state; `/health` for db+model; dev-side `pytest` before
   merging (the known environmental baseline on the dev box: 13 failures =
   5 `test_app`/`test_update_logic` health + 8 `test_classify` unsure-threshold
   ordering — those need the llama endpoint; zero regressions means byte-stable
   baseline).
5. If a deploy damages the tree: `git checkout -- <file>` restores tracked
   files (`git pull` alone will **not** restore locally-deleted tracked files
   when HEAD is already current — #95's incident).

## 5. Verification battery (copy-paste)

```
curl -s http://100.79.150.106:8008/health          # {"status":"ok","checks":{database,model}}
curl -s http://100.79.150.106:8008/stats           # label counts + read_sweep_* counters
curl -s http://100.79.150.106:8008/admin/sweep-state | python3 -m json.tool
ssh -i /opt/toolbox/ssh/jarvis_ops lohnn@100.79.150.106 'cd /home/lohnn/email_classifier && git status -sb && git log --oneline -3'
journalctl -u email-classifier --since '-30 min' --no-pager | grep -iE "sync|update|error"
```

Healthy `sweep-state` reads: `trained_categories.count == 4`
(`Jarvis/FOCUS,NOISE,REFERENCE,URGENT`), `category_source.centroids_exists=true`,
`resolved_all_mail="[Gmail]/All Mail"`, `discovery_error=null`, counters moving
(`read_sweep_labeled_total` ≤ `READ_SWEEP_BATCH_SIZE` per cycle, default 5 ≈
1440/day). Failures name themselves: empty vocabulary → `category_source`
block; discovery failures → `mailbox.discovery_error` + `mailbox.list_sample`
(LOGICAL and WIRE forms — read both, the wire form is what Gmail saw).

## 6. Access & credentials

- **SSH to Pi**: expiring key in `/opt/toolbox/ssh/jarvis_ops`; authorized_keys
  line carries `expiry-time="20261103235900"` — it **dies Nov 3, 2026**; ask
  the user for a fresh paste (or regenerate: `ssh-keygen -t ed25519 -f
  /opt/toolbox/ssh/jarvis_ops -N ''` and hand over the line with a new
  expiry-time). Connect: `ssh -i /opt/toolbox/ssh/jarvis_ops
  -o UserKnownHostsFile=/opt/toolbox/ssh/known_hosts lohnn@100.79.150.106`.
- **rclone**: container has a clamped remote `gdrive-classifier`
  (`/opt/toolbox/rclone/rclone.conf`, root_folder_id = the model bucket
  `1UpL5xeg…`, copy/copyto only — never sync). Caveat: it rides rclone's
  **shared client_id, which Google retires during 2026** — migrate to an own
  client_id before then. The Pi's remote is scope=**drive.file** (by design:
  minimal blast radius) — it sees only files it created; never widen it to fix
  a sync problem, re-route or exclude instead (that lesson took three days:
  the drive.file vortex of Oct 27).
- **OAuth authorize from a phone**: SSH with `-L 53682:127.0.0.1:53682`, then
  `rclone authorize drive --auth-no-open-browser` on the target machine; open
  the printed URL in the phone browser; paste the `config_token` back.

## 7. Bucket & mirror ownership map

| File/dir | Owner | Mirror behavior |
|---|---|---|
| `email_classifier_brain/model/label_centroids.json` | **git** (tracked) | excluded from both GDrive syncs (#95); bucket copy exists as belt, md5-matched |
| `model/*` (SetFit-era artifacts) | GDrive bucket (Mac-owned uploads) | sync will replace/delete locally per Pi's 2-file drive.file view — do not store anything there that code needs |
| `llama_model/merged_text_q8_0.gguf` (or aligned path — verify on Pi) | admin upload/install | untracked; reload path unit may auto-restart embedder |
| `storage/` (SQLite DB) | Pi + `gdrive:email-classifier-storage` | **copy** (restore + backup), not sync — safe for the DB |

Retrain PR checklist: new GGUF uploaded+installed+embedder restarted → new
`label_centroids.json` committed (same PR) → code PR merged → trigger-update →
sweep-state shows new `centroids_entries` + `trained_categories` across the
cycle → `/health` green → one classification sanity test.

---
*Maintained by the deploy operator & `email-brain-ml`. Sources: the
read-mail backfill sweep arc (PRs #90–#96) and lived diagnosis of three
production faults.*
