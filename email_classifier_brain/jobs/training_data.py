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


def _remove_examples(file_path: str, subject: str, body: str) -> int:
    """
    Remove every entry whose (subject, body) matches from a JSONL file.

    Rewrites the file atomically via a temp file + os.replace, so a crash
    mid-write can never truncate the file to a partial line.

    Returns the number of lines removed (-1 if the file does not exist).
    """
    if not os.path.exists(file_path):
        return -1

    kept_lines: list[str] = []
    removed = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                # Keep malformed lines untouched — cleanup is not this job's task
                kept_lines.append(line)
                continue
            if (existing.get("subject"), existing.get("body")) == (subject, body):
                removed += 1
                continue
            kept_lines.append(line)

    if removed == 0:
        return 0

    temp_path = file_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        for kept in kept_lines:
            f.write(kept + "\n")
    os.replace(temp_path, file_path)
    return removed


def add_to_training_data(
    log_entry: dict,
    corrected_category: str,
    previous_category: str | None = None,
):
    """
    Append a corrected email to the training data JSONL files.

    Correction is a MOVE, not an append: when ``previous_category`` is given
    (the label the email currently carries in the database / Gmail), any copy
    of the same (subject, body) example already in the previous category's
    file is removed. This keeps the corpus single-labelled — the old
    append-only behaviour let the same email train under two labels at once
    when the user re-labelled an already-corrected email.
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

    # Place the example in the new category's file (dedup within that file).
    def _append_example():
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example) + "\n")
        logger.info(f"Added email to {corrected_category}.jsonl training data.")

    new_key = (example["subject"], example["body"])
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
            if new_key in existing_keys:
                logger.info(f"Skipping duplicate in {corrected_category}.jsonl")
            else:
                _append_example()
        except Exception as e:
            # Unreadable target file: keep writing the example anyway (legacy
            # behaviour) — losing a corrected example is worse than a
            # duplicate, and load_training_data resolves duplicates.
            logger.warning(f"Error reading {file_path} for dedup check: {e}")
            _append_example()
    else:
        # Append-only for efficiency and scalability
        _append_example()

    # Purge the copy in the previous label's file AFTER the new copy is in
    # place: if the purge fails or the process dies here, the worst case is a
    # leftover duplicate (which load_training_data detects and resolves),
    # never a lost example.
    if previous_category and previous_category != corrected_category:
        old_path = os.path.join(TRAINING_DATA_DIR, f"{previous_category}.jsonl")
        try:
            removed = _remove_examples(old_path, example["subject"], example["body"])
            if removed > 0:
                logger.info(
                    f"Moved example out of {previous_category}.jsonl "
                    f"({removed} line(s) removed) into {corrected_category}.jsonl"
                )
            elif removed == 0:
                logger.info(
                    f"No matching example in {previous_category}.jsonl to remove "
                    f"(email may not have been in training data under that label)"
                )
        except Exception as e:
            logger.error(f"Error purging example from {old_path}: {e}")


def push_training_data_to_git() -> bool:
    """
    Commit and push changes in the training data directory to Git.

    Returns True when nothing failed (including the legitimate
    nothing-to-push case), False when the commit or the push failed.
    Failure must be OBSERVABLE, not a swallowed log line — the daily
    update job turns this into an update_history.json entry and an
    errored job_runs row so dead push credentials don't silently freeze
    the corpus (as happened 2026-03 → 2026-10).
    """
    logger.info("Attempting to push training data to git...")

    if not os.path.exists(TRAINING_DATA_DIR):
        logger.warning(f"Training data directory {TRAINING_DATA_DIR} does not exist. Skipping push.")
        return True

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
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while pushing training data: {e}")
        return False


def count_unpushed_commits() -> int | None:
    """
    Count local commits on the training-data repo that the remote does not
    have yet. Returns None when the answer is unknown (no upstream configured,
    git missing, or the directory is not a repo).

    A non-zero count means the corpus the user sees on GitHub is behind what
    the Pi has produced — the other half of the push-failure freeze signal.
    """
    data_dir = TRAINING_DATA_DIR
    if not os.path.isdir(os.path.join(data_dir, ".git")) and \
       subprocess.run(["git", "-C", data_dir, "rev-parse", "--is-inside-work-tree"],
                      capture_output=True).stdout.strip() != b"true":
        return None
    try:
        result = subprocess.run(
            ["git", "-C", data_dir, "rev-list", "--count", "@{u}..HEAD"],
            capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, OSError, ValueError):
        # No upstream branch configured or git failure — unknowable here.
        return None


def backfill_training_data_job(trigger: str = "scheduled"):
    """
    Rebuild training data files from all corrected entries in the database.
    Use this to recover training data if the training data directory was
    accidentally emptied or lost.

    Note: This appends to existing .jsonl files, so duplicates may be created
    if some entries already exist. The training pipeline resolves
    (subject, body) duplicates to their latest version with a loud warning.
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
