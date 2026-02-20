#!/bin/bash

# Configuration
UPDATE_MARKER=".update_request"
HISTORY_FILE="update_history.json"
VENV_BIN="./venv/bin"
UVICORN_CMD="$VENV_BIN/uvicorn"
PIP_CMD="$VENV_BIN/pip"
PYTHON_CMD="$VENV_BIN/python"
RCLONE_CMD="rclone"

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

# Load .env if present (for GDRIVE_REMOTE, GDRIVE_MODEL_PATH, MODEL_DIR)
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    . .env
    set +a
fi

# Google Drive sync defaults
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"
GDRIVE_MODEL_PATH="${GDRIVE_MODEL_PATH:-email-classifier-model}"
MODEL_DIR="${MODEL_DIR:-model}"
STORAGE_DIR="${STORAGE_DIR:-storage}"
GDRIVE_STORAGE_PATH="${GDRIVE_STORAGE_PATH:-email-classifier-storage}"

# Function to log to history file (JSON format)
log_event() {
    local status="$1"
    local message="$2"
    local timestamp=$(date -Iseconds)

    # Use python to generate safe JSON
    # We use python3 from system as json is standard library
    "$PYTHON_CMD" -c "import json, sys; print(json.dumps({'timestamp': '$timestamp', 'status': '$status', 'message': sys.argv[1]}))" "$message" >> "$HISTORY_FILE"

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

    # 2. Sync latest model from Google Drive (non-fatal)
    echo "Syncing model from Google Drive..."
    if $RCLONE_CMD sync "$GDRIVE_REMOTE:$GDRIVE_MODEL_PATH/" "$MODEL_DIR/"; then
        log_event "info" "Model synced from Google Drive"
    else
        log_event "warning" "Google Drive model sync failed, using existing model"
    fi

    # 3. Update code
    if ! git pull; then
        log_event "error" "git pull failed"
        # Remove marker so we don't loop forever on a git error
        rm "$UPDATE_MARKER"
    else
        # 4. Update dependencies
        if ! $PIP_CMD install -r requirements.txt; then
             log_event "error" "pip install failed"
             git reset --hard "$CURRENT_COMMIT"
             rm "$UPDATE_MARKER"
        else
            # 5. Verify startup
            echo "Verifying new version..."
            # Start in background, capturing output
            # We use a different port for verification if needed?
            # But the main service is down (we are in the wrapper), so 8000 is free.
            $UVICORN_CMD main:app --host 0.0.0.0 --port 8000 > startup.log 2>&1 &
            PID=$!

            # Wait up to 30 seconds for the health check to pass
            echo "Waiting for health check..."
            HEALTH_CHECK_PASSED=false
            for i in {1..30}; do
                # Check if process is still alive
                if ! kill -0 $PID 2>/dev/null; then
                    echo "Process died unexpectedly."
                    break
                fi

                # Check health endpoint
                # We use curl to check if the status is 200
                if curl -s -f http://localhost:8000/health >/dev/null; then
                    HEALTH_CHECK_PASSED=true
                    break
                fi
                sleep 1
            done

            if [ "$HEALTH_CHECK_PASSED" = true ]; then
                # It's running and healthy! Success.
                echo "Update verified (Health check passed)."
                kill $PID
                wait $PID 2>/dev/null
                rm "$UPDATE_MARKER"
                NEW_COMMIT=$(git rev-parse HEAD)
                log_event "success" "Update to $NEW_COMMIT successful"
            else
                # It died or timed out.
                echo "Update failed (Health check failed)."
                # Capture last few lines of log
                if kill -0 $PID 2>/dev/null; then
                    kill $PID
                    wait $PID 2>/dev/null
                fi
                ERROR_MSG=$(tail -n 10 startup.log | tr '\n' ' ')
                log_event "error" "Server failed verification after update: $ERROR_MSG"

                # Rollback
                git reset --hard "$CURRENT_COMMIT"
                rm "$UPDATE_MARKER"
            fi
        fi
    fi
fi

# Normal startup
# Restore storage from Google Drive if local storage is empty
if [ -z "$(ls -A \"$STORAGE_DIR\" 2>/dev/null)" ]; then
    echo "Local storage is empty. Restoring from Google Drive..."
    $RCLONE_CMD copy "$GDRIVE_REMOTE:$GDRIVE_STORAGE_PATH/" "$STORAGE_DIR/" --progress || echo "Warning: Storage restore failed. A new database will be created if one does not exist."
fi

echo "Backing up storage to Google Drive..."
# We use copy to avoid deleting files on remote if they are missing locally (safety first)
$RCLONE_CMD copy "$STORAGE_DIR/" "$GDRIVE_REMOTE:$GDRIVE_STORAGE_PATH/" --progress || echo "Warning: Storage backup failed"

echo "Starting email classifier service..."
exec $UVICORN_CMD main:app --host 0.0.0.0 --port 8000 "$@"
