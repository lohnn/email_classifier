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

## Project Structure

```
.
├── main.py                # FastAPI microservice & background jobs
├── database.py            # SQLite schema & persistence logic
├── imap_client.py         # Gmail/IMAP integration
├── config.py              # Shared configuration & input formatting
├── train.py               # SetFit training script
├── classify.py            # Inference engine
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
    └── MODEL_INFO.json
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

| Variable | Description |
| :--- | :--- |
| `MY_EMAIL` | Comma-separated list of your email addresses. |
| `IMAP_SERVER` | IMAP server (defaults to `imap.gmail.com`). |
| `IMAP_USER` | Email address for authentication. |
| `IMAP_PASSWORD` | App password or standard password. |
| `ADMIN_API_KEY` | Key for protecting admin/write endpoints. |
| `ENABLE_AUTO_CLASSIFICATION` | Set to `false` to disable the 5-min job. |
| `ENABLE_RECHECK_JOB` | Set to `false` to disable correction discovery. |
| `RECHECK_INTERVAL_HOURS` | Frequency of re-check job (default: 12). |
| `ENABLE_RECLASSIFY_JOB` | Set to `false` to disable periodic re-classification. |
| `RECLASSIFY_INTERVAL_HOURS`| Frequency of re-classification (default: re-check). |
| `VERIFICATION_LABEL` | Gmail label used for explicit verification. |
| `MODEL_DIR` | Path to trained model artifacts. |
| `TRAINING_DATA_DIR` | Path to JSONL training files. |
| `GDRIVE_REMOTE` | Rclone remote name (e.g. `gdrive`). |
| `GDRIVE_MODEL_PATH` | Google Drive path for model storage. |
| `STORAGE_DIR` | Directory for SQLite database. |

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

## Model Input Format

The structured string sent to the model looks like:

```
passage: Role: Direct | Mass Mail: No | Attachment Types: [PDF] | From: ops@company.com | To: me@company.com | Subject: Server is down | Body: All services offline...
```

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

## Microservice API

The brain runs a FastAPI microservice for inference, monitoring, and feedback.

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/run` | Manually trigger the classification job immediately. |
| `GET` | `/stats` | Get classification counts per category. |
| `GET` | `/notifications` | Get unread classification logs for the UI. |
| `POST` | `/notifications/ack` | Mark specific (or all) notifications as read. |
| `POST` | `/notifications/pop` | Fetch and mark unread notifications as read. |
| `GET` | `/labels` | List all supported classification categories. |
| `POST` | `/logs/{id}/correction` | Submit a category correction for a logged email. |
| `POST` | `/reclassify` | Re-predict categories for existing logs. |
| `GET` | `/health` | Simple service health check. |

### Administrative Endpoints

Requires `X-API-Key` header validated against `ADMIN_API_KEY`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/admin/check-corrections` | Discovery job for external label changes. |
| `POST` | `/admin/force-check-corrections`| Exhaustive re-check of all history labels. |
| `POST` | `/admin/backfill-training-data` | Rebuild JSONL from database corrections. |
| `POST` | `/admin/trigger-update` | Pull code/model and restart service. |
| `POST` | `/admin/push-training-data` | Manually push `TrainingData/` to Git. |
| `GET` | `/admin/update-errors` | Get history of automated update failures. |

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

| Job | Interval | Description |
| :--- | :--- | :--- |
| **Classification** | 5 mins | Fetches unprocessed emails from Gmail and applies labels. |
| **Re-check** | 12 hrs | Discovers label changes made manually on the server. |
| **Re-classify** | 12 hrs | Updates local history labels if the model has improved. |
| **Auto-Update** | 1 day | Pushes training data to Git and triggers a service refresh. |

The re-check job uses a "gliding scale" based on email age — checking newer
emails more frequently than older ones to optimize performance.

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

For production deployment, use `systemd` to manage the service. This ensures it starts on boot and restarts automatically if it crashes.

### 1. Install the service
The `email-classifier.service` file has been pre-configured for your environment.

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

| Action | Command |
| :--- | :--- |
| **Start** | `sudo systemctl start email-classifier` |
| **Stop** | `sudo systemctl stop email-classifier` |
| **Restart** | `sudo systemctl restart email-classifier` |
| **Status** | `sudo systemctl status email-classifier` |
| **View Logs** | `journalctl -u email-classifier -f` |

### 3. Automatic Updates
The service uses `run_service.sh`, which supports a safe update-and-rollback mechanism. To trigger an update (syncs model from Drive, pulls latest code, updates dependencies):

```bash
touch .update_request
sudo systemctl restart email-classifier
```

This triggers: sync from Drive → `git pull` → `pip install` → health check → serve. If the health check fails, it automatically rolls back the code to the previous commit.
