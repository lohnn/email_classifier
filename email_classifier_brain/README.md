# 📧 Email Classification System (SetFit + Raspberry Pi)

A few-shot email classifier powered by **SetFit** and the
**intfloat/multilingual-e5-small** embedding model. Designed to train on a
workstation and deploy for CPU inference on a **Raspberry Pi 4** (4 GB RAM).

## Features

- **Rich metadata** — Model sees role, sender, mass-mail flag, and attachment
  types, not just text
- **Dynamic categories** — Auto-discovered from `TrainingData/` file paths
- **Nested labels** — Use subdirectories for hierarchical categories (any depth)
- **E5 prefix** — `"passage: "` handled automatically everywhere
- **Raw email parsing** — `predict_raw_email()` extracts headers from
  `email.message.Message` objects
- **Uncertainty detection** — Applies a configurable `UNSURE_CLASSIFICATION`
  Gmail label when the classifier's confidence is low or when the gap between
  the top-two class probabilities is too small

## Project Structure

```
.
├── main.py                # FastAPI entry point: registers routers & scheduler
├── database.py            # SQLite schema & persistence logic
├── imap_client.py         # Gmail/IMAP integration
├── config.py              # Shared configuration & input formatting
├── job_queue.py           # Sequential JobQueue class + module-level singleton
├── train.py               # SetFit training script
├── evaluate.py            # Cross-validation evaluation harness
├── classify.py            # Inference engine
├── api/                   # HTTP API layer
│   ├── models.py          # Pydantic request/response models
│   ├── security.py        # API key authentication
│   └── routes/
│       ├── classification.py  # POST /run, POST /reclassify, GET /labels
│       ├── jobs.py            # GET /jobs/status, POST /jobs/cancel, GET /jobs/history
│       ├── notifications.py   # GET/POST /notifications/*
│       ├── admin.py           # POST /logs/{id}/correction, GET /logs/ambiguous, /admin/*
│       └── health.py          # GET /health, GET /stats
├── jobs/                  # Background job functions
│   ├── classification.py  # classification_job
│   ├── correction.py      # check_corrections_job, force_check_corrections_job
│   ├── reclassify.py      # reclassify_job
│   ├── training_data.py   # add_to_training_data, push_training_data_to_git, backfill_training_data_job
│   └── update.py          # scheduled_update_job, shutdown_server
├── retrain.sh             # Trainer workflow (pull → train → upload)
├── run_service.sh         # Production service wrapper with auto-upgrade
├── setup.sh               # Venv & dependency bootstrap
├── setup_wizard.py        # Interactive configuration
├── requirements.txt       # Python dependencies
├── TrainingData/          # Category examples (.jsonl format)
│   ├── NOISE.jsonl        # → label "NOISE"
│   └── WORK/
│       ├── URGENT.jsonl   # → label "WORK/URGENT"
│       └── FOCUS.jsonl    # → label "WORK/FOCUS"
└── model/                 # Trained model artifacts (Synced via rclone)
    ├── model.safetensors
    ├── model_head.pkl
    ├── label_mapping.json
    ├── MODEL_INFO.json
    └── MODEL_EVAL.json
├── storage/               # Persistent data
│   └── email_history.db   # SQLite logs and notification state
```

## Configuration

The easiest way to configure the system is to use the interactive setup script:

```bash
./setup.sh
```

This will:

1. Create a Python virtual environment.
2. Install all necessary dependencies.
3. Guide you through a setup wizard to configure your `.env` file, training data
   repository, and model syncing.

Alternatively, you can copy `.env.example` to `.env` manually and set your
values:

```bash
cp .env.example .env
# Then edit .env
```

### Environment Variables

| Variable                     | Description                                           |
| :--------------------------- | :---------------------------------------------------- |
| `MY_EMAIL`                   | Comma-separated list of your email addresses.         |
| `IMAP_SERVER`                | IMAP server (defaults to `imap.gmail.com`).           |
| `IMAP_USER`                  | Email address for authentication.                     |
| `IMAP_PASSWORD`              | App password or standard password.                    |
| `ADMIN_API_KEY`              | Key for protecting admin/write endpoints.             |
| `ENABLE_AUTO_CLASSIFICATION` | Set to `false` to disable the 5-min job.              |
| `ENABLE_RECHECK_JOB`         | Set to `false` to disable correction discovery.       |
| `RECHECK_INTERVAL_HOURS`     | Frequency of re-check job (default: 12).              |
| `ENABLE_RECLASSIFY_JOB`      | Set to `false` to disable periodic re-classification. |
| `RECLASSIFY_INTERVAL_HOURS`  | Frequency of re-classification (default: re-check).   |
| `ENABLE_READ_SWEEP`          | The read-mail backfill sweep — **enabled by default**; set to `false` to disable it. |
| `READ_SWEEP_BATCH_SIZE`      | Max read emails labelled per classification cycle when the sweep is enabled (default: `5`). |
| `VERIFICATION_LABEL`         | Gmail label used for explicit verification.           |
| `UNSURE_LABEL`               | Gmail label applied when confidence is low (default: `UNSURE_CLASSIFICATION`). |
| `UNSURE_CONFIDENCE_THRESHOLD`| Top-class probability below which the unsure label is applied (default: `0.65`). |
| `UNSURE_DELTA_THRESHOLD`     | Min gap between top-two class probabilities; below this the unsure label is applied (default: `0.10`). |
| `MODEL_DIR`                  | Path to trained model artifacts.                      |
| `TRAINING_DATA_DIR`          | Path to JSONL training files.                         |
| `GDRIVE_REMOTE`              | Rclone remote name (e.g. `gdrive`).                   |
| `GDRIVE_MODEL_PATH`          | Google Drive path for model storage.                  |
| `STORAGE_DIR`                | Directory for SQLite database.                        |

The `MY_EMAIL` list is used to determine your **role** in each email:

- `Direct` — any of your addresses is in the "To" field.
- `CC` — any is in the "CC" field.
- `Hidden` — BCC or mailing list.

## Training Data Format

Each category has one **JSONL** (JSON Lines) file in `TrainingData/`. The label
is derived from the file's path relative to `TrainingData/`, with `.jsonl`
stripped:

- `TrainingData/NOISE.jsonl` → label **NOISE**
- `TrainingData/WORK/URGENT.jsonl` → label **WORK/URGENT**
- `TrainingData/A/B/C.jsonl` → label **A/B/C** (arbitrary depth)

Subdirectories create hierarchical labels separated by `/`.

Each line in the `.jsonl` file is a single JSON object:

```json
{
  "subject": "Server is down",
  "body": "All services are offline! Engineers have been paged.",
  "from": "ops-alert@company.com",
  "to": "me@company.com",
  "cc": "cto@company.com",
  "mass_mail": false,
  "attachment_types": ["PDF"]
}
```

| Field              | Type     | Description                                    |
| ------------------ | -------- | ---------------------------------------------- |
| `subject`          | string   | Email subject line                             |
| `body`             | string   | Email body text                                |
| `from`             | string   | Sender address                                 |
| `to`               | string   | Recipient(s)                                   |
| `cc`               | string   | CC'd addresses                                 |
| `mass_mail`        | bool     | `true` if List-Unsubscribe header present      |
| `attachment_types` | string[] | File extensions, e.g. `["PDF", "DOCX", "ICS"]` |

One email must appear under exactly **one** label file. Corrections are
moves (`jobs/training_data.py` removes the `(subject, body)` copy from the
previous label's file when it writes the new one), and `train.py` resolves
any legacy cross-file duplicates to their **latest** version with a loud
printed warning — a stale double-labelled corpus cannot silently train.

## Model Input Format

The structured string sent to the model looks like:

```
passage: Role: Direct | Mass Mail: No | Attachment Types: [PDF] | From: ops@company.com | To: me@company.com | Subject: Server is down | Body: All services offline...
```

### Context-window cap (body truncation)

`intfloat/multilingual-e5-small` accepts at most **512 tokens**; the tokenizer
truncates silently from the end, which would silently eat body text on long
newsletters and reply chains. To prevent that silent accuracy leak,
`config.format_model_input()` caps the body itself before building the string:

- The cap is **`config.MAX_BODY_CHARS` (1280 chars ≈ 320 tokens)** — the
  512-token window minus the worst measured metadata prefix (161 tokens,
  measured with the real tokenizer) minus a safety margin.
- Long bodies keep the **first half and the last half** of the window, joined
  by a ` [...] ` marker, so signatures and closing paragraphs survive.
- The cap is purely character-based and deterministic — training and
  inference always produce byte-identical inputs for the same email.

Prevalence is observable, not just logged (a log line is only read when
someone already suspects a problem):

- `GET /stats` returns `model_inputs_total` and `model_inputs_truncated`
  counters for the running process.
- `python evaluate.py` prints a `Body truncation: N/M inputs (x%)` line and
  records a `body_truncation` block in `MODEL_EVAL.json`.
- `train.py` prints the same count when building the dataset.

## Quick Start

The quickest way to get started on any machine (Training or Server) is:

```bash
./setup.sh
```

Follow the prompts to configure your environment. Once finished, you can start
training or run the service as described below.

### Manual Setup (Alternative)

If you prefer to set up manually:

#### 1. Install Dependencies

```bash
# Python
pip install -r requirements.txt
```

#### 2. Configure & Train

```bash
# 1. Create .env from .env.example and configure it
# 2. Edit/add JSON files in TrainingData/
python train.py
```

### 3. Run Inference

```python
from classify import predict_email, predict_raw_email

# With explicit metadata
label = predict_email(
    subject="Server is down!",
    body="All services offline since 14:00.",
    sender="ops@company.com",
    to="me@company.com",
    mass_mail=False,
    attachment_types=["PDF"],
)

# Or parse a raw .eml file
import email
with open("message.eml") as f:
    msg = email.message_from_file(f)
label = predict_raw_email(msg)
```

## Adding a New Category

**Flat label:**

1. Create `TrainingData/BILLING.jsonl` with examples (one JSON per line)
2. Run `python train.py` → label `BILLING`

**Nested label:**

1. Create `TrainingData/WORK/BILLING.jsonl` with examples
2. Run `python train.py` → label `WORK/BILLING`

`classify.py` picks up new labels automatically from `model/label_mapping.json`.

**Via a manual Gmail tag:** if you tag an email with a label the model has
never heard of, the re-check job records it as a *candidate new category*
(visible via `GET /logs/candidate-categories`). Accepting it
(`POST /logs/candidate-categories/{id}/accept`) writes the training file and
records the correction; the category only takes effect for classification
after the next retrain. Discarding it just clears the flag — it never
touches your Gmail labels. Gmail **system artifacts never appear** in the
queue: backslash-prefixed IMAP flags (`\Important`, `\Inbox`, …) and the
auto-labels Gmail applies for your own addresses (`MY_EMAILS`) are filtered
out before recording, so the queue is for genuine new tags only. Tags the
model already knows never wait in the queue — they flow through the normal
correction path (the candidate filter only sees labels OUTSIDE the trained
set).

## Evaluating the Model

Training prints "Training complete" and nothing else — with a tiny dataset
there is no way to tell whether a retrain helped or hurt. `evaluate.py`
answers that with stratified k-fold cross-validation over the current
`TrainingData/` snapshot:

```bash
python evaluate.py            # k=5 (default)
python evaluate.py --folds 3  # fewer folds (faster, noisier)
```

Every example is held out exactly once: k throwaway models are trained on
k-1 folds and validated on the held-out fold. The pooled out-of-fold
predictions produce:

- **Overall accuracy**
- **Per-class precision / recall / F1** (with support)
- **Confusion matrix** (printed as a readable table)
- **Unsure rate** — fraction of held-out predictions that would get the
  `UNSURE_LABEL` (uses the same `is_unsure_classification()` thresholds as
  inference)
- **Body truncation rate** — how many inputs hit the E5 context-window body
  cap during formatting (see "Context-window cap" below)

The report is saved as `MODEL_EVAL.json` next to `MODEL_INFO.json` in the
model directory (`MODEL_DIR`). Note the semantics: cross-validation does
**not** produce or modify the saved model — `MODEL_EVAL.json` is the quality
score of the *current training-data snapshot*, and it records the training
data's git repo/commit so you can compare scores across retrains.

Each category needs at least k examples (one held out per fold). With tiny
datasets the metrics are high-variance — treat them as a signal, not a
guarantee.

## Microservice API

The brain runs a FastAPI microservice for inference, monitoring, and feedback.

### Endpoints

| Method | Endpoint  | Description                  |
| :----- | :-------- | :--------------------------- |
| `GET`  | `/health` | Simple service health check. |

### Protected Endpoints

Requires `X-API-Key` header validated against `ADMIN_API_KEY`.

| Method | Endpoint                         | Description                                          |
| :----- | :------------------------------- | :--------------------------------------------------- |
| `POST` | `/run`                           | Manually trigger the classification job immediately. |
| `POST` | `/reclassify`                    | Re-predict categories for existing logs.             |
| `GET`  | `/labels`                        | List all supported classification categories.        |
| `GET`  | `/stats`                         | Get classification counts per category.              |
| `GET`  | `/jobs/status`                   | Return current job queue state (running + waiting).  |
| `POST` | `/jobs/cancel`                   | Cancel the running job and clear the pending queue.  |
| `GET`  | `/jobs/history`                  | Per-run metadata for completed and in-progress jobs. |
| `GET`  | `/notifications`                 | Get unread classification logs for the UI.           |
| `POST` | `/notifications/ack`             | Mark specific (or all) notifications as read.        |
| `POST` | `/notifications/pop`             | Fetch and mark unread notifications as read.         |
| `GET`  | `/notifications/read`            | Get already-read notifications within a time range.  |
| `POST` | `/logs/{id}/correction`          | Submit a category correction for a logged email. Correction is a **move**: the example is written to the new category's file and removed from the previous one. |
| `GET`  | `/logs/ambiguous`                | Get logs flagged with multiple conflicting labels.   |
| `GET`  | `/logs/candidate-categories`     | Get logs carrying Gmail labels the model has never trained on (candidate new categories). |
| `POST` | `/logs/candidate-categories/{id}/accept`  | Admit a candidate category: writes the training file, records the correction, marks the email VERIFIED. Takes effect for classification after the next retrain. |
| `POST` | `/logs/candidate-categories/{id}/discard` | Discard a candidate new-category flag (no training-data or Gmail changes). |
| `POST` | `/admin/check-corrections`       | Discovery job for external label changes.            |
| `POST` | `/admin/force-check-corrections` | Exhaustive re-check of all history labels.           |
| `POST` | `/admin/backfill-training-data`  | Rebuild JSONL from database corrections.             |
| `POST` | `/admin/trigger-update`          | Pull code/model and restart service.                 |
| `POST` | `/admin/push-training-data`      | Manually push `TrainingData/` to Git.                |
| `GET`  | `/admin/update-errors`           | Get history of automated update failures.            |
| `GET`  | `/admin/retrain-status`          | Whether training data is ahead of the deployed model (`retrain_worthwhile` + both commit ids). |
| `POST` | `/admin/upload-model-artifact`   | **Tailscale/localhost ONLY** — stream a model artifact (.gguf/.safetensors/.json/.pkl) into staging; sha256 computed + echoed; optional `sha256_expected`. |
| `POST` | `/admin/install-model-artifact`  | **Tailscale/localhost ONLY** — atomically install a staged artifact (GGUF → llama model path, ST pieces → model/). No auto-restart. |
| `POST` | `/logs/{id}/repair-gmail-labels` | User-triggered Gmail label resync from the DB for one email (removes stale trained-category labels + UNSURE, applies expected category + VERIFIED). Works for archived/trashed mail. Returns a JSON action report. |

## Embedder Backend — llama.cpp (production) vs SetFit (rollback)

`classify.py` classifies via **kNN over per-class centroids** computed from
the LoRA-merged EmbeddingGemma-2 text model, served by a local **llama.cpp**
server (`--embedding --pooling mean --ubatch-size 2048`, OpenAI-style
`POST /v1/embeddings`). The classification prompt prefix
(`task: classification | query: `) is applied **in the caller** — llama.cpp
takes raw text. Output is always 768d (no truncate_dim; MRL-512 storage cuts
would be caller-side `vec[:512]` + re-normalize). The UNSURE gate
(`is_unsure_classification`) is unchanged; on the llama backend it runs with
the thresholds calibrated in `label_centroids.json` (cosine scale differs
from SetFit probabilities).

**Config (`.env`):**
- `EMBEDDER_BACKEND` = `llama_cpp` (default) | `st` (full rollback to the
  in-process SetFit path — flip only as a DELIBERATE ops action)
- `LLAMA_ENDPOINT` = `http://127.0.0.1:8090` (Pi: email-embedder.service)
- `LLAMA_TIMEOUT_SECONDS` / `LLAMA_RETRIES` / `LLAMA_RETRY_BACKOFF_SECONDS`
  (Pi reality 2026-10-25: a long email embeds in ~10–90 s on the Pi 4's A72 —
  the 30 s default aborts exactly those; the Pi sets 180)
- `CENTROIDS_PATH` = `model/label_centroids.json` (ships with the model)

**Outage semantics:** embedder failures surface LOUDLY — retried, then a
`warning` lands in `update_history.json` and every classification errors
out until the server recovers or ops flips the flag. There is NO automatic
fallback to SetFit.

**Provenance + rebuild:** `label_centroids.json` carries the merged-GGUF
sha256, the adapter (`email_real/lora_adapter_final.pt`), the corpus commit
(17a86de), and the validation numbers; rebuild it with
`build_label_centroids.py --splits <corpus> --endpoint <server>` (the script
validates against the §16 numbers before writing). The embedding route is
INFERENCE ONLY — training (train.py/evaluate.py) stays on SetFit/ST.

## Google Drive — Model Storage

The trained model (~490 MB) is stored in Google Drive instead of Git. Use
[rclone](https://rclone.org/) to sync:

```bash
# Install rclone
sudo apt install rclone    # Debian / RPi
brew install rclone         # macOS

# Configure Google Drive remote (one-time)
rclone config  # → New remote → name it "gdrive" → Google Drive → authenticate
```

After training, `retrain.sh` handles the upload automatically.

## Security

State-modifying and administrative endpoints are protected by an **API Key**
mechanism.

- **Header:** `X-API-Key`
- **Environment Variable:** `ADMIN_API_KEY`
- If `ADMIN_API_KEY` is not set in `.env`, all protected endpoints will return
  `500 Server Error` as a safety measure.
- **Trusted-network bypass:** requests arriving from localhost
  (`127.0.0.0/8`, `::1/128`) or the Tailscale CGNAT range (`100.64.0.0/10`)
  skip the key check entirely — detected from the socket peer address, no
  headers involved. The bypass grants full read **and** write access.

### Read-only key (`READ_API_KEY`, optional)

Setting `READ_API_KEY` adds a second, weaker tier: that key is accepted for
**GET/HEAD only** — every key-gated read endpoint (`/stats`, `/labels`,
`/notifications*`, `/jobs/status`, `/jobs/history`, `/logs/ambiguous`,
`/logs/candidate-categories`, `/admin/retrain-status`,
`/admin/update-errors`) — while **mutating routes (POST) still require the
admin key or the trusted-network bypass** (a read key gets `403` there):
corrections, candidate accept/discard, re-check/re-classify/backfill/update
triggers, notification ack/pop, job cancel. `GET /health` stays public
(its optional `check_imap=true` probe remains admin/bypass-only). Leaving
`READ_API_KEY` unset or empty changes nothing. Keys are never logged.

Typical split: anything that only displays state uses the read key;
anything that can change labeling, training data, or the service itself
keeps the admin key.

## Background Jobs

The service uses `APScheduler` to handle recurring tasks:

| Job                | Default Interval | Env Override                                         | Description                                                 |
| :----------------- | :--------------- | :--------------------------------------------------- | :---------------------------------------------------------- |
| **Classification** | 5 mins           | `ENABLE_AUTO_CLASSIFICATION` (on/off)                | Fetches unprocessed emails from Gmail and applies labels.   |
| **Re-check**       | 12 hrs           | `RECHECK_INTERVAL_HOURS`, `ENABLE_RECHECK_JOB`       | Discovers label changes made manually on the server.        |
| **Re-classify**    | 12 hrs           | `RECLASSIFY_INTERVAL_HOURS`, `ENABLE_RECLASSIFY_JOB` | Updates local history labels if the model has improved.     |
| **Auto-Update**    | 1 day            | `ZERO_CORRECTION_WARN_DAYS` (warn threshold)         | Pushes training data to Git, emits loop-health signals, and triggers a service refresh. |

### Read-mail backfill sweep (WI-103)

**Enabled by default** (`ENABLE_READ_SWEEP`; set it to `false` to disable).
The classification job has two strictly ordered phases. The UNSEEN phase (new
mail) always runs first; the read sweep runs only afterwards and only when it
is allowed to. Enabling the sweep changes no safety rail — the hard unread-first
gate, the trained-label filter, and the processed-marker skip_ids below all
apply unchanged, and the pace stays gradual.

**Hard gate:** the sweep runs only when the unread queue is *fully drained* —
the service probes the live mailbox after the unread phase and requires that
**no UNSEEN email in the INBOX lacks all trained-category labels**. Any new
unread mail arriving mid-cycle re-blocks the sweep; a probe failure fails
closed (sweep skipped). Unread always wins; the phases never interleave.

**Pacing:** when enabled, at most `READ_SWEEP_BATCH_SIZE` (default 5) read
emails are labelled per classification cycle — roughly 1440/day on the default
5-minute cadence, so multi-thousand-email history sweeps over days instead of
blasting the mailbox. Sweep activity is observable via `GET /stats`
(`read_sweep_attempts_total`, `read_sweep_gated_total`,
`read_sweep_candidates_total`, `read_sweep_labeled_total`).

**Safety rules:**

- Emails already carrying a trained-category label are never candidates — the
  sweep never re-labels or overwrites anything (corrections MOVE via the
  re-check job, they are never appended).
- Zero-label read emails that already have a database log row count as
  processed and are skipped (`database.get_all_log_ids()` passed as
  `skip_ids`) — so a read email whose labels were stripped is not endlessly
  re-fetched.
- The DB log row is written only after Gmail accepted the label; an email
  whose label application failed stays eligible for the next cycle.
- The sweep has no cursor of its own: after a restart it re-derives the
  remaining history from Gmail labels + DB rows.

The re-check job uses a "gliding scale" based on email age — checking newer
emails more frequently than older ones to optimize performance. Among the
rows that are due, ordering prioritises (1) ambiguous rows, (2) rows whose
label changed most recently, (3) rows checked longest ago — so a manual
re-tag is picked up on the next pass instead of waiting behind a
newest-first queue.

**Cadence ≠ freshness guarantee — with one hardcoded exception.** The glide
windows re-suppress rows: a label change on a fresh email is picked up on
the first pass at least 12 h after that email's previous re-check (1–7
days old: ≥ 24 h; 7–30 days: ≥ 7 days; older: ≥ 30 days), no matter how
often the job runs — so a sub-12h `RECHECK_INTERVAL_HOURS` speeds up
*scans*, not *row revisits*. The exception, hardcoded by choice: an email
younger than **2 hours** (by its own Date header) is eligible on **every**
pass — a correction made within an email's first 2 hours resolves on the
next cadence pass. A correction made later in day 0 follows the 12 h
suppression, counted from the last sub-2h stamp (e.g. an email re-labeled
in the evening after a midday re-check resolves the next morning).
Immediate pickup for a specific known edit remains
`POST /admin/force-check-corrections` (the only path that bypasses the
glide windows).

### Daily health signals (update_history.json)

The daily auto-update job appends to `update_history.json` (the same file
`run_service.sh` writes) whenever the correction→training-data loop shows
failure signatures — push failures, training-data commits that never
reached the remote, or a `ZERO_CORRECTION_WARN_DAYS`-day stretch with zero
corrections while mail volume is non-zero (the March 2026 corpus-freeze
signature). It also emits a one-time **"Retrain worthwhile"** entry when
the training data contains commits the deployed model was never trained
on; training itself is never automatic — `./retrain.sh` on the trainer
machine stays the invocation path, and `GET /admin/retrain-status` exposes
the same state.

### Corrections vs. archive & delete (known behaviors)

- **Correct via the UI/API first, then archive or trash → safe.** The
  correction endpoint writes the training-data line immediately from the
  database copy — what Gmail does with the message afterwards never edits
  or removes it.
- **Re-label in Gmail, then remove it from the Inbox (archive/trash) before
  a re-check runs → still picked up.** The label reader falls back to a
  bounded out-of-inbox supplement: ids the INBOX pass missed are retried
  with the same `X-GM-MSGID` search in `[Gmail]/All Mail` and, if still
  missing, `[Gmail]/Trash` (batch ≤ 50, the session mailbox is always
  restored to INBOX). **Spam is deliberately not searched** — repairing a
  classification from inside Spam is not a supported flow, and searching
  Spam could resurrect junk into training rankings. If a message is
  permanently purged or sits in Spam, the recheck keeps the row pending;
  the UI **Correct** button remains the universal fallback because it
  rewrites the example from what the system already stored.
- **Resolving a correction clears the classifier's UNSURE flag.** When a
  correction resolves (re-check, force re-check, or the UI/API endpoint),
  the brain removes the old category label **and** the
  `UNSURE_CLASSIFICATION` flag from the email in Gmail, then applies
  `VERIFIED` as before. Only classifier-owned labels are removed — your own
  (unknown) labels are never touched and are surfaced as candidate new
  categories instead. Label writes now cover archived and
  trashed-but-not-purged mail too (uid resolution walks All Mail/Trash),
  and every removal is **verified server-side** after the STORE. Label
  writes also reach archived and trashed-but-not-purged mail (uid
  resolution walks All Mail/Trash in the same selection as the STORE).
- **If Gmail and the database ever disagree, you can see it and fix it.**
  Failed cleanup after a resolved correction (DB updated, Gmail stale —
  e.g. 'VERIFIED applied but the old label is still on the message', hit
  live 2026-10) lands in `update_history.json` as a warning naming the
  email. Fix one email with
  `POST /logs/{id}/repair-gmail-labels`: it recomputes classifier-owned
  label state from the database (removes stale trained-category labels and
  UNSURE, applies the expected category and VERIFIED) and returns a JSON
  action report of exactly what it did. Nothing repairs Gmail
  automatically — the endpoint is user-triggered by design. **Model-artifact deploy loop (replaces the scp/GDrive model
  dance):** `POST /admin/upload-model-artifact` (raw body + `?filename=`)
  streams into a staging dir with streaming sha256 (optional
  `sha256_expected`, ~2 GiB cap, basename+extension whitelist), then
  `POST /admin/install-model-artifact?staged_name=…` atomically moves it to
  the active location (GGUF → the llama.cpp model path per
  `LLAMA_MODEL_FILE`; ST pieces → `model/`). **No auto-restart** — the
  target service loads on its next restart. Both steps are
  **Tailscale/localhost-peer-only**: even a valid admin key from an
  untrusted network is rejected (403) — one of the accepted file types is
  a pickle that executes on load. Both steps land in
  `update_history.json` (file, size, sha256, peer). GDrive storage backups
  stay as-is; only the manual model-transfer dance is replaced. **Confirmed
  rows only:** on a row that is still model-predicted (`corrected_category`
  NULL) the repair endpoint REFUSES with a 409 whose payload contains the
  plan it *would* have run — a repair must never stamp the model's guess
  as VERIFIED; confirm the label with `POST /logs/{id}/correction` first.
  A restore of the classifier's storage from the (up to a day stale)
  Google Drive backup — the empty/unreadable-DB path in `run_service.sh` —
  is likewise surfaced as a warning in `update_history.json`, because it
  silently regresses classifier state (corrections, recheck stamps) to the
  backup's age.
- **Confirming a correct prediction = add `VERIFIED`, not removing the UNSURE flag.**
  When the re-check sees `VERIFIED` on an email whose single trained label
  matches what the database already believes, it writes the confirmed
  example into that category's `.jsonl`, marks the row as
  confirmed-by-human (`corrected_category`) — which **excludes it from
  re-classification**, so the model never second-guesses that email again —
  and strips a leftover `UNSURE_CLASSIFICATION` flag automatically (a no-op
  when the email doesn't carry it). Removing `UNSURE_CLASSIFICATION` alone
  does nothing: no example is written, and while the row is still
  unconfirmed the re-classify job may re-apply the flag (fresh model
  confidence decides) or even re-label the email if its prediction has
  changed. Works the same for archived and not-yet-purged trashed mail
  (the out-of-inbox label supplement makes those readable).
- **Deleting an email never removes its training example.** Nothing watches
  Trash — a corrected example stays in its `.jsonl` and keeps teaching the
  model even after the email is gone. Deleting mail is not a way to retract
  a label: edit the training file (or correct the email again before deletion).

## Retraining Workflow

On the trainer machine (workstation with GPU/MPS):

```bash
./retrain.sh
```

This single command:

1. Pulls the latest training data from the private Git repo
2. Trains the model (SetFit few-shot learning)
3. Uploads the model artifacts to Google Drive
4. Commits and pushes any new label corrections to the training data repo

## Deploying on Raspberry Pi

### llama.cpp embedder — one-command bootstrap + zero-touch model updates

The production embedder is a local llama.cpp server (`email-embedder.service`)
holding the LoRA-merged GGUF (`merged_text_q8_0.gguf`; sha in the
`artifacts/llama_cpp_migration_20261024/SHA256SUMS`). One-time bootstrap on
the Pi:

```bash
sudo ./bootstrap_llama_cpp.sh    # from email_classifier_brain/ (units/ sibling used)
```

It fetches the newest ubuntu-arm64 llama.cpp nightly FROM THE RELEASES LIST
(never a pinned release-notes URL — those rot; effort-capped with manual
fallback), installs `llama-server` to `/home/email_bot/llama.cpp/`, installs
`email-embedder.service` (only if absent) plus the
`email-embedder-reload.path` + `email-embedder-reload.service` watcher pair,
creates the model dir, and daemon-reloads. No keys; no other services.

After that, model updates are zero-touch: upload via
`/admin/upload-model-artifact`, install via `/admin/install-model-artifact`
— the path unit sees the installed GGUF and restarts the embedder
automatically. The email-classifier service itself is NOT touched by the
reload: switching its `model/` set stays the manual install + service
restart flow as documented. Reminder: a plain service restart does NOT pull
code or sync models — only the `.update_request` marker path (daily
auto-update) does, unchanged.

```bash
# Initial setup
git clone <code-repo-url> && cd email_classifier_brain
pip install -r requirements.txt

# Install and configure rclone for model sync
sudo apt install rclone
rclone config  # Set up "gdrive" remote

# Pull the model
rclone sync gdrive:email-classifier-model/ model/

# Start the service
./run_service.sh
```

## Running as a Service (systemd)

For production deployment, use `systemd` to manage the service. This ensures it
starts on boot and restarts automatically if it crashes.

### 1. Install the service

The `email-classifier.service` file has been pre-configured for your
environment.

```bash
# Copy the service file to the system directory
sudo cp email-classifier.service /etc/systemd/system/

# Reload systemd and enable the service
sudo systemctl daemon-reload
sudo systemctl enable email-classifier
sudo systemctl start email-classifier

# Check service status
sudo systemctl status email-classifier
```

### 2. Service Management

| Action        | Command                                   |
| :------------ | :---------------------------------------- |
| **Start**     | `sudo systemctl start email-classifier`   |
| **Stop**      | `sudo systemctl stop email-classifier`    |
| **Restart**   | `sudo systemctl restart email-classifier` |
| **Status**    | `sudo systemctl status email-classifier`  |
| **View Logs** | `journalctl -u email-classifier -f`       |

### 3. Automatic Updates

The service uses `run_service.sh`, which supports a safe update-and-rollback
mechanism. To trigger an update (syncs model from Drive, pulls latest code,
updates dependencies):

```bash
touch .update_request
sudo systemctl restart email-classifier
```

This triggers: sync from Drive → `git pull` → `pip install` → health check →
serve. If the health check fails, it automatically rolls back the code to the
previous commit.
