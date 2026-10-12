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
after the next retrain. Discarding it just clears the flag.

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

## Background Jobs

The service uses `APScheduler` to handle recurring tasks:

| Job                | Default Interval | Env Override                                         | Description                                                 |
| :----------------- | :--------------- | :--------------------------------------------------- | :---------------------------------------------------------- |
| **Classification** | 5 mins           | `ENABLE_AUTO_CLASSIFICATION` (on/off)                | Fetches unprocessed emails from Gmail and applies labels.   |
| **Re-check**       | 12 hrs           | `RECHECK_INTERVAL_HOURS`, `ENABLE_RECHECK_JOB`       | Discovers label changes made manually on the server.        |
| **Re-classify**    | 12 hrs           | `RECLASSIFY_INTERVAL_HOURS`, `ENABLE_RECLASSIFY_JOB` | Updates local history labels if the model has improved.     |
| **Auto-Update**    | 1 day            | `ZERO_CORRECTION_WARN_DAYS` (warn threshold)         | Pushes training data to Git, emits loop-health signals, and triggers a service refresh. |

The re-check job uses a "gliding scale" based on email age — checking newer
emails more frequently than older ones to optimize performance. Among the
rows that are due, ordering prioritises (1) ambiguous rows, (2) rows whose
label changed most recently, (3) rows checked longest ago — so a manual
re-tag is picked up on the next pass instead of waiting behind a
newest-first queue.

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
- **Re-label in Gmail, then archive/trash before a re-check runs → the
  correction is missed, silently.** Label reads happen over IMAP with
  `INBOX` selected (`imap_client.connect`), so labels on messages that are
  no longer in the Inbox are invisible to the scheduled re-check AND to a
  force re-check. The re-check simply keeps the row pending; as long as the
  message is restored to the Inbox within Gmail's 30-day trash window, the
  next re-check picks the change up. After permanent deletion (Gmail's
  30-day trash purge) the label state is gone for the pipeline; re-labeling
  restored mail or once more correcting via the UI both work.
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
