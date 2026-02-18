#!/bin/bash
# setup.sh — Entry point for Email Classifier setup

set -e

# Change to the script's directory
cd "$(dirname "$0")"

echo "============================================"
echo "  Email Classifier — Setup"
echo "============================================"

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# 1. Create directories
mkdir -p TrainingData
mkdir -p model
mkdir -p checkpoints
mkdir -p storage
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate venv and install requirements
echo "Installing/Updating dependencies..."
. venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# 3. Run the setup wizard
python setup_wizard.py
