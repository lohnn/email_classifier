"""
jobs/update.py — Auto-update Job
================================

Handles the daily update cycle: pushing training data to Git,
requesting a service restart, and graceful server shutdown.

Also emits the daily HEALTH SIGNALS of the correction→training-data loop
into update_history.json (the same file run_service.sh writes, which the
user reads):

  - git push failure in the daily training-data push
  - local training-data commits that never reached the remote
  - a many-day zero-correction streak while mail volume is non-zero
    (the signature of the 2026-03 → 2026-10 corpus freeze)
  - 'retrain worthwhile' when training data has commits the deployed
    model was never trained on (manual ./retrain.sh remains the
    invocation path; training is never run automatically)
"""

import datetime
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

import config
import database
from jobs.training_data import count_unpushed_commits, push_training_data_to_git

logger = logging.getLogger(__name__)

# Same file run_service.sh logs to (its log_event shape is one JSON line per
# event), so all update-cycle signals live in one place the user already reads.
HISTORY_FILE = "update_history.json"
HISTORY_MAX_LINES = 50

# Marker file: remembers the last data-repo commit a 'retrain worthwhile'
# signal was emitted for, so the daily job doesn't repeat the entry.
RETRAIN_MARKER_FILE = ".retrain_notified_commit"

# How many days without a recorded correction (while mail keeps flowing)
# before the daily job raises the freeze warning.
ZERO_CORRECTION_WARN_DAYS = getattr(config, "ZERO_CORRECTION_WARN_DAYS", 14)

RETRAIN_MESSAGE = (
    "Retrain worthwhile: training data is ahead of the deployed model "
    "(data commit {data}, model trained on {model}). "
    "Training is NOT run automatically — run ./retrain.sh on the trainer "
    "machine; the Pi picks the new model up via Google Drive on next restart."
)


def shutdown_server():
    """
    Shuts down the server gracefully to allow for updates/restarts.
    """
    logger.info("Shutting down server for update/restart in 2 seconds...")
    time.sleep(2)
    os.kill(os.getpid(), signal.SIGTERM)


def append_update_history(status: str, message: str) -> None:
    """
    Append one JSON line to update_history.json, matching the shape
    run_service.sh's log_event() writes: {timestamp, status, message}.
    Rotates the file to the last HISTORY_MAX_LINES lines.
    """
    try:
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": status,
            "message": message,
        }
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > HISTORY_MAX_LINES:
            with open(HISTORY_FILE + ".tmp", "w", encoding="utf-8") as f:
                f.writelines(lines[-HISTORY_MAX_LINES:])
            os.replace(HISTORY_FILE + ".tmp", HISTORY_FILE)
    except Exception as e:
        logger.error(f"Could not write update history entry: {e}")


def _git_commit(repo_dir: str) -> str:
    """HEAD commit of a git work tree, or '' when unknown."""
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def get_retrain_status() -> dict:
    """
    Compare the training-data repo HEAD against the commit the deployed
    model was trained on (recorded in MODEL_INFO.json at training time,
    train.py _write_model_info, and synced with the model itself).

    retrain_worthwhile is True whenever training data contains commits the
    current model never saw.
    """
    data_commit = _git_commit(config.TRAINING_DATA_DIR)
    model_commit = None
    try:
        info_path = os.path.join(config.MODEL_OUTPUT_DIR, "MODEL_INFO.json")
        with open(info_path, "r", encoding="utf-8") as f:
            model_commit = json.load(f).get("training_data_commit")
    except Exception:
        model_commit = None

    retrain_worthwhile = bool(data_commit and data_commit != model_commit)
    return {
        "data_commit": data_commit,
        "model_trained_commit": model_commit,
        "retrain_worthwhile": retrain_worthwhile,
    }


def _check_retrain_signal() -> None:
    """Emit a 'retrain worthwhile' update_history entry once per data state."""
    status = get_retrain_status()
    if not status["retrain_worthwhile"]:
        return

    marker = Path(RETRAIN_MARKER_FILE)
    try:
        if marker.exists() and marker.read_text().strip() == status["data_commit"]:
            return  # already notified for this exact data commit
        marker.write_text(status["data_commit"] + "\n")
    except Exception as e:
        logger.error(f"Could not update retrain marker file: {e}")

    append_update_history(
        "info",
        RETRAIN_MESSAGE.format(
            data=status["data_commit"][:12],
            model=status["model_trained_commit"] or "unknown",
        ),
    )


def _check_zero_correction_streak() -> None:
    """
    Warn when no label corrections have been recorded for
    ZERO_CORRECTION_WARN_DAYS while emails were still being classified —
    the observable signature of a stalled correction→training-data loop.
    """
    volume_cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
    stats = database.get_stats(start_time=volume_cutoff)
    volume = sum(stats.values())
    if volume == 0:
        return  # nothing flowing in — no expectation of corrections either

    since = (datetime.datetime.now() - datetime.timedelta(days=ZERO_CORRECTION_WARN_DAYS)).isoformat()
    if database.has_correction_since(since):
        return

    append_update_history(
        "warning",
        f"No label corrections recorded for {ZERO_CORRECTION_WARN_DAYS}+ days "
        f"while {volume} emails were classified in the last 7 — the "
        f"correction→training-data loop may be stalled (recheck job failing, "
        f"git push dead, or genuinely nothing to correct).",
    )


def scheduled_update_job(trigger: str = "scheduled"):
    """
    Scheduled job to trigger the daily update.
    """
    logger.info("Scheduled update job triggering...")
    run_id = database.start_job_run("auto_update", trigger)
    error_messages = []
    try:
        push_ok = push_training_data_to_git()
        if not push_ok:
            error_messages.append("Training data git push failed")
            append_update_history(
                "error",
                "Daily training-data push failed — GitHub corpus is not "
                "receiving new training data. Check git credentials on this "
                "machine and local unpushed commits in the training-data repo.",
            )

        unpushed = count_unpushed_commits()
        if unpushed:
            append_update_history(
                "warning",
                f"{unpushed} local training-data commit(s) not pushed to the "
                f"remote — the GitHub corpus is behind this machine's data.",
            )

        try:
            _check_zero_correction_streak()
        except Exception as e:
            logger.error(f"Zero-correction streak check failed: {e}")

        try:
            _check_retrain_signal()
        except Exception as e:
            logger.error(f"Retrain signal check failed: {e}")

        Path(".update_request").touch()
        if error_messages:
            database.finish_job_run(run_id, "error", error_message="; ".join(error_messages))
        else:
            database.finish_job_run(run_id, "success")
        shutdown_server()
    except Exception as e:
        logger.error(f"Error in scheduled update job: {e}")
        database.finish_job_run(run_id, "error", error_message=str(e))
