"""
jobs/update.py — Auto-update Job
=================================

Handles the daily update cycle: pushing training data to Git,
requesting a service restart, and graceful server shutdown.
"""

import logging
import os
import signal
import time
from pathlib import Path

import database
from jobs.training_data import push_training_data_to_git

logger = logging.getLogger(__name__)


def shutdown_server():
    """
    Shuts down the server gracefully to allow for updates/restarts.
    """
    logger.info("Shutting down server for update/restart in 2 seconds...")
    time.sleep(2)
    os.kill(os.getpid(), signal.SIGTERM)


def scheduled_update_job(trigger: str = "scheduled"):
    """
    Scheduled job to trigger the daily update.
    """
    logger.info("Scheduled update job triggering...")
    run_id = database.start_job_run("auto_update", trigger)
    try:
        push_training_data_to_git()
        Path(".update_request").touch()
        database.finish_job_run(run_id, "success")
        shutdown_server()
    except Exception as e:
        logger.error(f"Error in scheduled update job: {e}")
        database.finish_job_run(run_id, "error", error_message=str(e))
