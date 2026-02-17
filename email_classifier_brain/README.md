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
├── config.py              # Shared config (MY_EMAIL, input formatting)
├── train.py               # Training script
├── classify.py            # Inference script (RPi optimized)
├── retrain.sh             # Trainer workflow (pull → train → upload)
├── run_service.sh         # Service wrapper with auto-upgrade
├── requirements.txt       # Python dependencies
├── TrainingData/ ←─────── # Symlink/path to private email-classifier-data repo
│   ├── NOISE.json         # → label "NOISE"
│   ├── WORK/
│   │   ├── URGENT.json    # → label "WORK/URGENT"
│   │   └── FOCUS.json     # → label "WORK/FOCUS"
│   └── PERSONAL/
│       └── REFERENCE.json # → label "PERSONAL/REFERENCE"
└── model/ ←───────────────# Synced from Google Drive via rclone
    ├── model.safetensors
    ├── model_head.pkl
    ├── label_mapping.json
    └── MODEL_INFO.json    # Provenance: links model to training data commit
```

## Configuration

The easiest way to configure the system is to use the interactive setup script:

```bash
./setup.sh
```

This will:
1. Create a Python virtual environment.
2. Install all necessary dependencies.
3. Guide you through a setup wizard to configure your `.env` file, training data repository, and model syncing.

Alternatively, you can copy `.env.example` to `.env` manually and set your values:

```bash
cp .env.example .env
# Then edit .env
```

This is used to determine your **role** in each email:

- `Direct` — you're in the "To" field
- `CC` — you're in the "CC" field
- `Hidden` — BCC or mailing list

## Training Data Format

Each category has one JSON file in `TrainingData/`. The label is derived from
the file's path relative to `TrainingData/`, with `.json` stripped:

- `TrainingData/NOISE.json` → label **NOISE**
- `TrainingData/WORK/URGENT.json` → label **WORK/URGENT**
- `TrainingData/A/B/C.json` → label **A/B/C** (arbitrary depth)

Subdirectories create hierarchical labels separated by `/`.

```json
[
    {
        "subject": "Server is down",
        "body": "All services are offline! Engineers have been paged.",
        "from": "ops-alert@company.com",
        "to": "me@company.com",
        "cc": "cto@company.com",
        "mass_mail": false,
        "attachment_types": ["PDF"]
    }
]
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

Follow the prompts to configure your environment. Once finished, you can start training or run the service as described below.

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

1. Create `TrainingData/BILLING.json` with examples
2. Run `python train.py` → label `BILLING`

**Nested label:**

1. Create `TrainingData/WORK/BILLING.json` with examples
2. Run `python train.py` → label `WORK/BILLING`

`classify.py` picks up new labels automatically from `model/label_mapping.json`.

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

## Retraining Workflow

On the trainer machine (workstation with GPU/MPS):

```bash
./retrain.sh
```

This single command:

1. Pulls the latest training data from the private Git repo
2. Trains the model
3. Uploads the model to Google Drive
4. Commits and pushes any training data changes

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

The service auto-upgrades when a `.update_request` marker is created:

```bash
touch .update_request
sudo systemctl restart email-classifier
```

This triggers: model sync from Drive → code pull → dependency update → health
check → serve.
