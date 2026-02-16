#!/bin/bash
# retrain.sh — Trainer-side workflow
# ====================================
# Pull latest training data → train → upload model to Google Drive
# → commit & push training data changes.
#
# Prerequisites:
#   - rclone configured with a Google Drive remote (see .env.example)
#   - Private training data repo cloned as sibling directory
#   - TRAINING_DATA_DIR set in .env to point at the private repo
#
# Usage:
#     ./retrain.sh

set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".env" ]; then
    # Export .env variables (skip comments and blank lines)
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Defaults (overridden by .env if set)
TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-../email_classifier_data/TrainingData}"
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"
GDRIVE_MODEL_PATH="${GDRIVE_MODEL_PATH:-email-classifier-model}"
MODEL_DIR="${MODEL_DIR:-../email_classifier_data/model}"

# Python executable (prefer venv if available)
VENV_PYTHON="./venv/bin/python"
if [ -f "$VENV_PYTHON" ]; then
    PYTHON_CMD="$VENV_PYTHON"
else
    PYTHON_CMD="python3"
fi

# Resolve training data repo root (parent of TrainingData/)
DATA_REPO_DIR="$(cd "$TRAINING_DATA_DIR/.." && pwd)"

# Safety check: ensure DATA_REPO_DIR is not the code repo
if [ "$DATA_REPO_DIR" = "$SCRIPT_DIR" ]; then
    echo "ERROR: TRAINING_DATA_DIR resolves to the code repo directory."
    echo "Set TRAINING_DATA_DIR in .env to point at your private data repo."
    echo "Example: TRAINING_DATA_DIR=../email_classifier_data/TrainingData"
    exit 1
fi

echo "============================================"
echo "  Email Classifier — Retrain & Upload"
echo "============================================"

# 1. Pull latest training data
echo ""
echo "→ Pulling latest training data..."
git -C "$DATA_REPO_DIR" pull

# 2. Train the model
echo ""
echo "→ Training model..."
$PYTHON_CMD train.py

# 3. Upload model to Google Drive
echo ""
echo "→ Uploading model to Google Drive ($GDRIVE_REMOTE:$GDRIVE_MODEL_PATH/)..."
rclone sync "$MODEL_DIR/" "$GDRIVE_REMOTE:$GDRIVE_MODEL_PATH/" --progress
echo "  ✓ Model uploaded successfully."

# 4. Commit & push any training data changes
echo ""
echo "→ Checking for training data changes..."
cd "$DATA_REPO_DIR"
git add "$TRAINING_DATA_DIR"
if git diff --cached --quiet; then
    echo "  No training data changes to commit."
else
    git commit -m "Training data update $(date +%Y-%m-%d)"
    git push
    echo "  ✓ Training data pushed."
fi

echo ""
echo "============================================"
echo "  ✓ Done! Model is live on Google Drive."
echo "============================================"
