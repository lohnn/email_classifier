#!/bin/bash
# retrain.sh — Trainer-side workflow
# ====================================
# Pull latest training data → train → upload model to Google Drive
# → commit & push training data changes.
#
# Prerequisites:
#   - rclone configured with a Google Drive remote (see .env.example)
#   - Private training data repo cloned as sibling directory
#   - TRAINING_DATA_DIR set in .env to point at the private repo root
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
# Default assumes a sibling directory structure where the repo root IS the data dir:
#   project_root/
#     email_classifier/ (current repo)
#     email_classifier_data/ (data repo root = TrainingDataDir)
TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-../email_classifier_data}"
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

# Resolve repo root (which is just TRAINING_DATA_DIR now)
# Only proceed if TRAINING_DATA_DIR is set to something custom
if [[ "$TRAINING_DATA_DIR" != "TrainingData" ]]; then
    # Resolve absolute path
    DATA_REPO_DIR="$(cd "$TRAINING_DATA_DIR" 2>/dev/null && pwd || echo "")"
    
    # If DATA_REPO_DIR is empty, it means the path didn't exist. 
    # Let's infer where it *should* be based on the relative path.
    if [ -z "$DATA_REPO_DIR" ]; then
         # e.g. ../email_classifier_data
         DATA_REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/$(basename "$TRAINING_DATA_DIR")"
    fi
else
    # Fallback to current dir if using local mock data (mostly for testing)
    DATA_REPO_DIR="$SCRIPT_DIR"
fi

# Check if data repo exists AND is a git repo
if [ ! -d "$DATA_REPO_DIR/.git" ]; then
    # Special case: if using local TrainingData mock, we don't expect a git repo at root
    if [[ "$TRAINING_DATA_DIR" == "TrainingData" ]]; then
         # Just verify the folder exists
         if [ ! -d "$TRAINING_DATA_DIR" ]; then
             echo "ERROR: Local mock data folder 'TrainingData' not found."
             exit 1
         fi
         # Emulate DATA_REPO_DIR as script dir, but we won't do git pull/push
         DATA_REPO_DIR="$SCRIPT_DIR"
    else
        echo "⚠️  Training data repo not found (or not a git repo) at: $DATA_REPO_DIR"
        echo ""
        read -p "Would you like to clone it now? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # If directory exists but isn't a git repo, back it up first
            if [ -d "$DATA_REPO_DIR" ]; then
                BACKUP_DIR="${DATA_REPO_DIR}_backup_$(date +%s)"
                echo "→ Backing up existing folder to $BACKUP_DIR..."
                mv "$DATA_REPO_DIR" "$BACKUP_DIR"
            fi
    
            echo ""
            read -p "Enter git repository URL (e.g., git@github.com:user/repo.git): " REPO_URL
            echo "→ Cloning $REPO_URL into $DATA_REPO_DIR..."
            git clone "$REPO_URL" "$DATA_REPO_DIR"
        else
            echo "Please clone your private data repo manually or update .env."
            exit 1
        fi
    fi
fi

# Safety check: ensure DATA_REPO_DIR is not the code repo
# UNLESS we are explicitly using the local mock data folder "TrainingData"
if [ "$DATA_REPO_DIR" = "$SCRIPT_DIR" ] && [[ "$TRAINING_DATA_DIR" != "TrainingData" ]]; then
    echo "ERROR: TRAINING_DATA_DIR resolves to the code repo directory."
    echo "Set TRAINING_DATA_DIR in .env to point at your private data repo."
    echo "Example: TRAINING_DATA_DIR=../email_classifier_data"
    exit 1
fi

echo "============================================"
echo "  Email Classifier — Retrain & Upload"
echo "============================================"

# 1. Pull latest training data
echo ""
echo "→ Pulling latest training data..."
if [[ "$TRAINING_DATA_DIR" != "TrainingData" ]]; then
    git -C "$DATA_REPO_DIR" pull
else
    echo "  (Skipping git pull for local mock data)"
fi

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
if [[ "$TRAINING_DATA_DIR" != "TrainingData" ]]; then
    cd "$DATA_REPO_DIR"
    git add .
    if git diff --cached --quiet; then
        echo "  No training data changes to commit."
    else
        git commit -m "Training data update $(date +%Y-%m-%d)"
        git push
        echo "  ✓ Training data pushed."
    fi
else
    echo "  (Skipping git push for local mock data)"
fi

echo ""
echo "============================================"
echo "  ✓ Done! Model is live on Google Drive."
echo "============================================"
