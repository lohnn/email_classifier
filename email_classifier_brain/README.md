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
├── requirements.txt       # Python dependencies
├── TrainingData/           # One .json per category (subdirs = nested labels)
│   ├── NOISE.json         # → label "NOISE"
│   ├── WORK/
│   │   ├── URGENT.json    # → label "WORK/URGENT"
│   │   └── FOCUS.json     # → label "WORK/FOCUS"
│   └── PERSONAL/
│       └── REFERENCE.json # → label "PERSONAL/REFERENCE"
└── model/                  # Output after training
    ├── model.safetensors
    └── label_mapping.json
```

## Configuration

Copy `.env.example` to `.env` and set your email address(es):

```bash
cp .env.example .env
# Then edit .env:
MY_EMAIL=me@company.com,my.alias@company.com
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

### 1. Install Dependencies

```bash
# System dependency (for model versioning)
sudo apt-get install git-lfs  # Debian/RPi
brew install git-lfs           # macOS
git lfs install

# Python
pip install -r requirements.txt
```

### 2. Configure & Train

```bash
# 1. Set MY_EMAIL in config.py
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

## Git LFS — Model Version Control

```bash
git lfs track "model/**"
git add .gitattributes model/
git commit -m "Add trained model"
git push
```

## Deploying on Raspberry Pi

```bash
# Initial setup
git clone <your-repo-url> && cd <repo-name>
git lfs install && git lfs pull
pip install -r requirements.txt

# Update after retraining
git pull
```

## E5 Prefix Note

The `intfloat/multilingual-e5-small` model requires a `"passage: "` prefix. Both
`train.py` and `classify.py` handle this via the shared
`config.format_model_input()` — you never need to add it manually.
