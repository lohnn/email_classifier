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

# 1. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate venv and install requirements
echo "Installing/Updating dependencies..."
# Use '.' instead of 'source' for better shell compatibility
. venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 3. Run the setup wizard
python3 setup_wizard.py
