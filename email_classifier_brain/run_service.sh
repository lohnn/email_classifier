#!/bin/bash

# Configuration
UPDATE_MARKER=".update_request"
HISTORY_FILE="update_history.json"
VENV_BIN="./venv/bin"
UVICORN_CMD="$VENV_BIN/uvicorn"
PIP_CMD="$VENV_BIN/pip"
PYTHON_CMD="$VENV_BIN/python"

# Fallback to system commands if venv not found (useful for dev/testing)
if [ ! -f "$UVICORN_CMD" ]; then
    UVICORN_CMD="uvicorn"
fi
if [ ! -f "$PIP_CMD" ]; then
    PIP_CMD="pip"
fi
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Function to log to history file (JSON format)
log_event() {
    local status="$1"
    local message="$2"
    local timestamp=$(date -Iseconds)

    # Use python to generate safe JSON
    # We use python3 from system as json is standard library
    $PYTHON_CMD -c "import json, sys; print(json.dumps({'timestamp': '$timestamp', 'status': '$status', 'message': sys.argv[1]}))" "$message" >> "$HISTORY_FILE"

    # Rotate log: Keep last 50 lines
    if [ -f "$HISTORY_FILE" ]; then
        tail -n 50 "$HISTORY_FILE" > "$HISTORY_FILE.tmp" && mv "$HISTORY_FILE.tmp" "$HISTORY_FILE"
    fi
}

if [ -f "$UPDATE_MARKER" ]; then
    echo "Update marker found. Attempting update..."

    # 1. Save current state
    CURRENT_COMMIT=$(git rev-parse HEAD)
    log_event "info" "Starting update from commit $CURRENT_COMMIT"

    # 2. Update code
    if ! git pull; then
        log_event "error" "git pull failed"
        # Remove marker so we don't loop forever on a git error
        rm "$UPDATE_MARKER"
    else
        # 3. Update dependencies
        if ! $PIP_CMD install -r requirements.txt; then
             log_event "error" "pip install failed"
             git reset --hard "$CURRENT_COMMIT"
             rm "$UPDATE_MARKER"
        else
            # 4. Verify startup
            echo "Verifying new version..."
            # Start in background, capturing output
            # We use a different port for verification if needed?
            # But the main service is down (we are in the wrapper), so 8000 is free.
            $UVICORN_CMD main:app --host 0.0.0.0 --port 8000 > startup.log 2>&1 &
            PID=$!

            # Wait 15 seconds to catch immediate crashes
            sleep 15

            if kill -0 $PID 2>/dev/null; then
                # It's still running! Success.
                echo "Update verified."
                kill $PID
                wait $PID 2>/dev/null
                rm "$UPDATE_MARKER"
                NEW_COMMIT=$(git rev-parse HEAD)
                log_event "success" "Update to $NEW_COMMIT successful"
            else
                # It died.
                echo "Update failed. Rolling back."
                # Capture last few lines of log
                ERROR_MSG=$(tail -n 10 startup.log | tr '\n' ' ')
                log_event "error" "Server crashed after update: $ERROR_MSG"

                # Rollback
                git reset --hard "$CURRENT_COMMIT"
                rm "$UPDATE_MARKER"
            fi
        fi
    fi
fi

# Normal startup
echo "Starting email classifier service..."
exec $UVICORN_CMD main:app --host 0.0.0.0 --port 8000
