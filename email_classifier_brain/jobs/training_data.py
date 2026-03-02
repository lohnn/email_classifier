"""
jobs/training_data.py — Training Data Management
=================================================

Functions to write, deduplicate, and push email training data,
and to backfill training data from corrected database entries.
"""

import datetime
import json
import logging
import os
import subprocess

import config
import database
from config import TRAINING_DATA_DIR
from job_queue import job_queue
from retry import with_retry

logger = logging.getLogger(__name__)


def add_to_training_data(log_entry: dict, corrected_category: str):
    """
    Append a corrected email to the training data JSONL files.
    """
    try:
        if isinstance(log_entry.get("attachment_types"), str):
            att_types = json.loads(log_entry.get("attachment_types"))
        else:
            att_types = log_entry.get("attachment_types") or []
    except Exception:
        att_types = []

    example = {
        "subject": config.clean_subject(log_entry.get("subject", "")),
        "body": config.clean_body(log_entry.get("body", "")),
        "from": log_entry.get("sender", ""),
        "to": log_entry.get("recipient", ""),
        "cc": log_entry.get("cc", ""),
        "mass_mail": bool(log_entry.get("mass_mail", False)),
        "attachment_types": att_types
    }

    # Ensure TRAINING_DATA_DIR exists
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

    file_path = os.path.join(TRAINING_DATA_DIR, f"{corrected_category}.jsonl")

    # Dedup check: skip if an entry with the same subject+body already exists
    if os.path.exists(file_path):
        try:
            existing_keys = set()
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing = json.loads(line)
                    existing_keys.add((existing.get("subject"), existing.get("body")))
            if (example["subject"], example["body"]) in existing_keys:
                logger.info(f"Skipping duplicate in {corrected_category}.jsonl")
                return
        except Exception as e:
            logger.warning(f"Error reading {file_path} for dedup check: {e}")

    # Append-only for efficiency and scalability
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(example) + "\n")
    logger.info(f"Added email to {corrected_category}.jsonl training data.")


def push_training_data_to_git():
    """
    Commit and push changes in the training data directory to Git.
    """
    logger.info("Attempting to push training data to git...")

    if not os.path.exists(TRAINING_DATA_DIR):
        logger.warning(f"Training data directory {TRAINING_DATA_DIR} does not exist. Skipping push.")
        return

    try:
        subprocess.run(["git", "add", "."], cwd=TRAINING_DATA_DIR, check=True, capture_output=True)

        status = subprocess.run(["git", "status", "--porcelain"], cwd=TRAINING_DATA_DIR, check=True, capture_output=True, text=True)

        if status.stdout.strip():
            logger.info("Changes detected in training data. Committing...")
            subprocess.run([
                "git",
                "-c", "user.name=Classifier Bot",
                "-c", "user.email=bot@example.com",
                "commit",
                "-m", f"Auto-update training data: {datetime.datetime.now().isoformat()}"
            ], cwd=TRAINING_DATA_DIR, check=True, capture_output=True)
            logger.info("Pushing to remote...")
            with_retry(
                subprocess.run,
                ["git", "push"],
                cwd=TRAINING_DATA_DIR,
                check=True,
                capture_output=True,
                retries=3,
                backoff=5.0,
                exceptions=(subprocess.CalledProcessError, OSError),
            )
            logger.info("Training data pushed successfully.")
        else:
            logger.info("No changes to push in training data.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error while pushing training data: {e}")


def backfill_training_data_job(trigger: str = "scheduled"):
    """
    Rebuild training data files from all corrected entries in the database.
    Use this to recover training data if the training data directory was
    accidentally emptied or lost.

    Note: This appends to existing .jsonl files, so duplicates may be created
    if some entries already exist. The training pipeline should handle dedup.
    """
    logger.info("Starting backfill_training_data_job...")
    run_id = database.start_job_run("backfill", trigger)

    try:
        corrected_logs = database.get_all_corrected_logs()
        if not corrected_logs:
            logger.info("No corrected logs found in database. Nothing to backfill.")
            database.finish_job_run(run_id, "success", emails_processed=0, emails_updated=0)
            return

        logger.info(f"Backfilling training data from {len(corrected_logs)} corrected entries...")

        success_count = 0
        error_count = 0

        was_cancelled = False
        for log in corrected_logs:
            if job_queue.is_cancelled():
                logger.info("Backfill training data job cancelled.")
                was_cancelled = True
                break
            try:
                add_to_training_data(log, log['corrected_category'])
                success_count += 1
            except Exception as e:
                logger.error(f"Error backfilling training data for {log['id']}: {e}")
                error_count += 1

        logger.info(f"Backfill finished. Success: {success_count}, Errors: {error_count}")
        final_status = "cancelled" if was_cancelled else "success"
        database.finish_job_run(run_id, final_status, emails_processed=len(corrected_logs), emails_updated=success_count, error_count=error_count)

    except Exception as e:
        logger.error(f"Error in backfill_training_data_job: {e}")
        database.finish_job_run(run_id, "error", error_message=str(e))
