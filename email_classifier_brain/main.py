import logging
import os
import signal
import json
import time
import datetime
import subprocess
from typing import List, Optional, Any, Dict
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Local imports handling (standalone vs package)
try:
    import classify
    import database
    import imap_client
    import config
    from config import TRAINING_DATA_DIR
except ImportError:
    from classifier_brain import classify, database, imap_client, config
    from classifier_brain.config import TRAINING_DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global lock for the classification job to prevent concurrent execution
# Since we are running in a sync endpoint, a threading.Lock is appropriate.
import threading
job_lock = threading.Lock()

scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    database.init_db()

    # Start scheduler
    logger.info("Starting scheduler...")
    # Run every 5 minutes if auto-classification is enabled
    if config.ENABLE_AUTO_CLASSIFICATION:
        scheduler.add_job(
            classification_job,
            trigger=IntervalTrigger(minutes=5),
            id="classification_job",
            replace_existing=True
        )
    else:
        logger.info("Automatic classification is disabled via ENABLE_AUTO_CLASSIFICATION.")

    # Run re-check job
    if config.ENABLE_RECHECK_JOB:
        scheduler.add_job(
            check_corrections_job,
            trigger=IntervalTrigger(hours=config.RECHECK_INTERVAL_HOURS),
            id="check_corrections_job",
            replace_existing=True
        )
    else:
        logger.info("Re-check job disabled.")

    # Run reclassify job
    if config.ENABLE_RECLASSIFY_JOB:
        # Offset by half the interval to avoid overlapping with check_corrections_job
        reclassify_offset = datetime.timedelta(hours=config.RECLASSIFY_INTERVAL_HOURS / 2)
        scheduler.add_job(
            reclassify_job,
            trigger=IntervalTrigger(hours=config.RECLASSIFY_INTERVAL_HOURS),
            id="reclassify_job",
            replace_existing=True,
            next_run_time=datetime.datetime.now() + reclassify_offset,
        )
    else:
        logger.info("Reclassify job disabled.")

    # Run auto-update every day
    scheduler.add_job(
        scheduled_update_job,
        trigger=IntervalTrigger(days=1),
        id="auto_update_job",
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduler started.")

    yield

    # Shutdown
    scheduler.shutdown()
    logger.info("Scheduler shutdown.")

app = FastAPI(title="Email Classifier Microservice", lifespan=lifespan)

# Security
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_scheme)):
    """
    Validates the API key against ADMIN_API_KEY in the environment.
    If ADMIN_API_KEY is not set, access is denied (500).
    """
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key:
        # Fail safe: if no key is configured, no one can access
        logger.error("ADMIN_API_KEY not set in environment. Blocking admin access.")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: ADMIN_API_KEY not set"
        )

    if api_key != expected_key:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials"
        )
    return api_key

# Job
def classification_job(limit: int = 20):
    if not job_lock.acquire(blocking=False):
        logger.warning("Classification job already running. Skipping this execution.")
        return {
            "status": "skipped",
            "reason": "Job already running",
            "processed_count": 0,
            "details": []
        }

    logger.info("Starting classification job...")
    results = []
    client = None
    try:
        # Connect to IMAP
        client = imap_client.GmailClient()

        # Get known categories to skip
        known_labels = classify.get_available_categories()

        # Fetch emails, stopping early once we have enough
        emails = client.fetch_unprocessed_emails(known_labels, limit=limit)
        logger.info(f"Fetched {len(emails)} unprocessed emails (limit={limit}).")

        for e_id, msg in emails:
            try:
                # Extract full info
                info = classify.extract_email_info(msg)

                # Predict
                label, score = classify.predict_email(
                    subject=info["subject"],
                    body=info["body"],
                    sender=info["sender"],
                    to=info["to"],
                    cc=info["cc"],
                    mass_mail=info["mass_mail"],
                    attachment_types=info["attachment_types"],
                    return_score=True
                )
                logger.info(f"Classified email {e_id}: {label} ({score:.2f})")

                # Apply label
                client.apply_label(e_id, label)

                # Extract date
                date_str = msg.get("Date")
                email_timestamp = None
                if date_str:
                    try:
                        from email.utils import parsedate_to_datetime
                        email_timestamp = parsedate_to_datetime(date_str)
                    except Exception:
                        logger.warning(f"Could not parse date: {date_str}")

                # Log to DB with full info
                database.add_log(
                    id=e_id,  # This is now the gmail_id string
                    sender=info["sender"],
                    recipient=info["to"],
                    subject=info["subject"],
                    predicted_category=label,
                    confidence_score=score,
                    timestamp=email_timestamp,
                    body=info["body"],
                    cc=info["cc"],
                    mass_mail=info["mass_mail"],
                    attachment_types=info["attachment_types"]
                )

                results.append({
                    "id": e_id,
                    "sender": info["sender"],
                    "recipient": info["to"],
                    "subject": info["subject"],
                    "label": label,
                    "score": score
                })
            except Exception as e_inner:
                logger.error(f"Error processing email {e_id}: {e_inner}")

        logger.info("Classification job finished.")
        return results

    except Exception as e:
        logger.error(f"Error in classification job: {e}")
        return []
    finally:
        if client:
            client.disconnect()
        job_lock.release()

def shutdown_server():
    """
    Shuts down the server gracefully to allow for updates/restarts.
    """
    logger.info("Shutting down server for update/restart in 2 seconds...")
    time.sleep(2)
    os.kill(os.getpid(), signal.SIGTERM)

def scheduled_update_job():
    """
    Scheduled job to trigger the daily update.
    """
    logger.info("Scheduled update job triggering...")
    try:
        push_training_data_to_git()
        Path(".update_request").touch()
        shutdown_server()
    except Exception as e:
        logger.error(f"Error in scheduled update job: {e}")

def add_to_training_data(log_entry: dict, corrected_category: str):
    """
    Append a corrected email to the training data JSONL files.
    """
    # Prepare the example in the format expected by training
    # attachment_types in DB is a JSON string
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
        # git add .
        subprocess.run(["git", "add", "."], cwd=TRAINING_DATA_DIR, check=True, capture_output=True)

        # Check if there are changes to commit
        status = subprocess.run(["git", "status", "--porcelain"], cwd=TRAINING_DATA_DIR, check=True, capture_output=True, text=True)

        if status.stdout.strip():
            logger.info("Changes detected in training data. Committing...")
            # Use -c to provide git config for environments where it might not be set
            subprocess.run([
                "git",
                "-c", "user.name=Classifier Bot",
                "-c", "user.email=bot@example.com",
                "commit",
                "-m", f"Auto-update training data: {datetime.datetime.now().isoformat()}"
            ], cwd=TRAINING_DATA_DIR, check=True, capture_output=True)
            logger.info("Pushing to remote...")
            subprocess.run(["git", "push"], cwd=TRAINING_DATA_DIR, check=True, capture_output=True)
            logger.info("Training data pushed successfully.")
        else:
            logger.info("No changes to push in training data.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error while pushing training data: {e}")

def reclassify_job(limit: int = 100):
    """
    Background job to re-classify existing logs.
    """
    if not job_lock.acquire(blocking=False):
        logger.warning("Job already running. Skipping re-classification.")
        return {"status": "skipped", "reason": "Job already running"}

    client = None
    updated_count = 0
    errors = 0
    
    try:
        logger.info("Starting re-classification job...")
        logs = database.get_logs_for_reclassification()
        
        # Connect to IMAP
        client = imap_client.GmailClient()
        
        # Limit processing
        if len(logs) > limit:
            logger.info(f"Limiting re-classification to {limit} emails (out of {len(logs)}).")
            logs = logs[:limit]

        for log in logs:
            gmail_id = log['id']
            current_label = log['predicted_category']
            
            try:
                # 1. Fetch email content using Gmail ID
                msg = client.fetch_email_by_gmail_id(gmail_id)
                
                info = None
                if msg:
                     info = classify.extract_email_info(msg)
                else:
                    # Fallback: use stored body if available, though less reliable for full re-eval if we added new features dependent on headers not stored
                    # But for now, if we can't fetch from Gmail (maybe deleted?), we skip or use stored?
                    # Let's skip if we can't find it in Gmail, as we can't update labels there anyway.
                    logger.warning(f"Could not fetch email {gmail_id} from Gmail. Skipping.")
                    continue

                # 2. Re-predict
                label, score = classify.predict_email(
                    subject=info["subject"],
                    body=info["body"],
                    sender=info["sender"],
                    to=info["to"],
                    cc=info["cc"],
                    mass_mail=info["mass_mail"],
                    attachment_types=info["attachment_types"],
                    return_score=True
                )
                
                # 3. Check if changed
                if label != current_label:
                    logger.info(f"Re-classification change for {gmail_id}: {current_label} -> {label} ({score:.2f})")
                    
                    # 4. Update Gmail Labels
                    # Remove old label
                    if current_label:
                        client.remove_label(gmail_id, current_label)
                    # Apply new label
                    client.apply_label(gmail_id, label)
                    
                    # 5. Update Database
                    # We accept specific 'update_log_prediction' but currently 'add_log' handles updates via upsert/duplicate check logic we added? 
                    # Actually I added logic to update if exists in add_log.
                    # So calling add_log with same ID should update it.
                    
                    # Re-extract timestamp to be safe or keep original? 
                    # We should probably keep original timestamp.
                    # add_log uses 'timestamp' arg.
                    
                    orig_ts = None
                    if log['timestamp']:
                        try:
                            orig_ts = datetime.datetime.fromisoformat(log['timestamp'])
                        except:
                            pass
                            
                    database.add_log(
                        id=gmail_id,
                        sender=info["sender"],
                        recipient=info["to"],
                        subject=info["subject"],
                        predicted_category=label,
                        confidence_score=score,
                        timestamp=orig_ts,
                        body=info["body"],
                        cc=info["cc"],
                        mass_mail=info["mass_mail"],
                        attachment_types=info["attachment_types"]
                    )
                    updated_count += 1
                else:
                    # Update score/metadata even if label same? 
                    # Maybe useful if model confidence changed.
                    pass

            except Exception as e:
                logger.error(f"Error re-classifying {gmail_id}: {e}")
                errors += 1

        logger.info(f"Re-classification finished. Updated {updated_count} emails.")
        return {
            "status": "success", 
            "processed": len(logs), 
            "updated": updated_count,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Error in re-classification job: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if client:
            client.disconnect()
        job_lock.release()

def _resolve_correction(trained_found, is_verified, current_local):
    """
    Determine correction action based on found IMAP labels vs local state.

    Returns a dict with keys:
      - is_ambiguous: bool
      - correction_candidate: str or None
      - cleanup_needed: bool
      - verified_candidate: str or None
    """
    is_ambiguous = False
    correction_candidate = None
    cleanup_needed = False
    verified_candidate = None

    if len(trained_found) == 0:
        pass
    elif len(trained_found) == 1:
        candidate = trained_found[0]
        if is_verified:
            verified_candidate = candidate
            if candidate != current_local:
                correction_candidate = candidate
        else:
            if candidate != current_local:
                correction_candidate = candidate
    else:
        # Multiple trained labels
        if is_verified:
            if current_local in trained_found:
                others = [l for l in trained_found if l != current_local]
                if len(others) == 1:
                    correction_candidate = others[0]
                    verified_candidate = others[0]
                    cleanup_needed = True
                else:
                    is_ambiguous = True
            else:
                is_ambiguous = True
        else:
            if current_local in trained_found:
                others = [l for l in trained_found if l != current_local]
                if len(others) == 1:
                    correction_candidate = others[0]
                    cleanup_needed = True
                else:
                    is_ambiguous = True
            else:
                is_ambiguous = True

    return {
        "is_ambiguous": is_ambiguous,
        "correction_candidate": correction_candidate,
        "cleanup_needed": cleanup_needed,
        "verified_candidate": verified_candidate,
    }

def check_corrections_job(limit: int = 200):
    """
    Background job to check for label corrections from the server (IMAP).
    Checks emails based on a gliding scale of age.
    """
    if not job_lock.acquire(blocking=False):
        logger.warning("Job already running. Skipping check_corrections_job.")
        return

    client = None
    try:
        logger.info("Starting check_corrections_job...")

        # 1. Get candidates
        candidates = database.get_candidate_logs_for_recheck(limit)
        if not candidates:
            logger.info("No candidates for re-check.")
            return

        logger.info(f"Checking {len(candidates)} emails for external corrections...")

        # 2. Get labels from IMAP
        client = imap_client.GmailClient()
        candidate_ids = [c['id'] for c in candidates]

        current_labels_map = client.get_labels_for_emails(candidate_ids)

        known_categories = set(classify.get_available_categories())

        updates_count = 0
        ambiguous_count = 0

        for log in candidates:
            gid = log['id']
            # If fetch failed or email deleted, we might not have it in map
            if gid not in current_labels_map:
                # Update recheck anyway so we don't loop on it
                database.update_recheck_status(gid, log.get('ambiguous_labels'))
                continue

            found_labels = current_labels_map[gid]

            # Identify trained labels (excluding VERIFIED label)
            trained_found = [lbl for lbl in found_labels if lbl in known_categories and lbl != config.VERIFICATION_LABEL]

            # Check for explicit verification
            is_verified = config.VERIFICATION_LABEL in found_labels

            current_local = log['corrected_category'] or log['predicted_category']

            result = _resolve_correction(trained_found, is_verified, current_local)
            is_ambiguous = result["is_ambiguous"]
            correction_candidate = result["correction_candidate"]
            cleanup_needed = result["cleanup_needed"]
            verified_candidate = result["verified_candidate"]

            # Execute Actions
            if is_ambiguous:
                logger.info(f"Ambiguous labels for {gid}: {trained_found}")
                database.update_recheck_status(gid, ambiguous_labels=trained_found)
                ambiguous_count += 1
            else:
                processed = False

                # Apply correction if detected
                if correction_candidate:
                    logger.info(f"Detected external correction for {gid}: {current_local} -> {correction_candidate}")

                    # Write training data FIRST, then update DB.
                    # If training data write fails, the DB won't be updated.
                    add_to_training_data(log, correction_candidate)
                    database.update_log_correction(gid, correction_candidate)

                    # Cleanup old label if needed
                    if cleanup_needed:
                        logger.info(f"Removing old label {current_local} from {gid}")
                        client.remove_label(gid, current_local)

                    # Mark as verified in IMAP (permanent marker)
                    if not is_verified:
                        logger.info(f"Adding {config.VERIFICATION_LABEL} to {gid}")
                        client.apply_label(gid, config.VERIFICATION_LABEL)

                    processed = True
                    updates_count += 1

                # Apply verification if detected (even if no correction, or after correction)
                if verified_candidate:
                    logger.info(f"Verified correctness for {gid}: {verified_candidate}")

                    # If we didn't just add it via correction, add to training data now
                    # (Prevent duplicates if correction_candidate == verified_candidate)
                    if not correction_candidate:
                        # Write training data FIRST, then update DB.
                        add_to_training_data(log, verified_candidate)
                        database.update_log_correction(gid, verified_candidate)
                        processed = True
                        updates_count += 1

                # Mark recheck done (clears ambiguous if any)
                database.update_recheck_status(gid, ambiguous_labels=None)

        logger.info(f"Re-check finished. Updates: {updates_count}, Ambiguous: {ambiguous_count}")

    except Exception as e:
        logger.error(f"Error in check_corrections_job: {e}")
    finally:
        if client:
            client.disconnect()
        job_lock.release()

def force_check_corrections_job():
    """
    Force re-check ALL emails for label corrections, bypassing the gliding
    scale schedule. Also imports any labeled emails from IMAP that are
    missing from the local database (e.g. after a DB reset).

    WARNING: This is an expensive operation. It should ONLY be used when you
    have manually re-labelled emails in Gmail and want to pick up those
    corrections immediately to update training data. Do NOT call this as part
    of regular scheduled operation — use check_corrections_job instead.
    """
    BATCH_SIZE = 200

    if not job_lock.acquire(blocking=False):
        logger.warning("Job already running. Skipping force_check_corrections_job.")
        return

    client = None
    try:
        logger.info("Starting force_check_corrections_job (bypassing schedule)...")

        client = imap_client.GmailClient()
        known_categories_list = classify.get_available_categories()
        known_categories = set(known_categories_list)

        # ---------------------------------------------------------------
        # Phase 0: Import labeled emails from IMAP that are missing in DB
        # ---------------------------------------------------------------
        logger.info("Phase 0: Scanning IMAP for labeled emails missing from DB...")
        labeled_emails = client.scan_labeled_emails(known_categories_list)
        import_count = 0

        for gid, (labels, msg) in labeled_emails.items():
            # Check if this email already exists in the DB
            existing = database.get_log_by_id(gid)
            if existing:
                continue

            # Find the trained label on this email
            trained_labels = [lbl for lbl in labels if lbl in known_categories and lbl != config.VERIFICATION_LABEL]
            if len(trained_labels) != 1:
                # Ambiguous or no trained label — skip import
                if trained_labels:
                    logger.info(f"Skipping import of {gid}: ambiguous labels {trained_labels}")
                continue

            label = trained_labels[0]

            try:
                info = classify.extract_email_info(msg)

                # Extract date
                date_str = msg.get("Date")
                email_timestamp = None
                if date_str:
                    try:
                        from email.utils import parsedate_to_datetime
                        email_timestamp = parsedate_to_datetime(date_str)
                    except Exception:
                        logger.warning(f"Could not parse date for imported email {gid}: {date_str}")

                database.add_log(
                    id=gid,
                    sender=info["sender"],
                    recipient=info["to"],
                    subject=info["subject"],
                    predicted_category=label,
                    confidence_score=0.0,  # Imported, not predicted
                    timestamp=email_timestamp,
                    body=info["body"],
                    cc=info["cc"],
                    mass_mail=info["mass_mail"],
                    attachment_types=info["attachment_types"]
                )
                import_count += 1
                logger.info(f"Imported email {gid} with label {label}")
            except Exception as e:
                logger.error(f"Error importing email {gid}: {e}")

        logger.info(f"Phase 0 complete. Imported {import_count} emails from IMAP.")

        # ---------------------------------------------------------------
        # Phase 1: Check corrections on all DB entries
        # ---------------------------------------------------------------
        all_candidates = database.get_all_logs_for_recheck()
        if not all_candidates:
            logger.info("No candidates for forced re-check.")
            return

        total = len(all_candidates)
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Phase 1: Force-checking {total} emails in {total_batches} batches of {BATCH_SIZE}...")

        updates_count = 0
        ambiguous_count = 0

        for batch_num in range(total_batches):
            batch_start = batch_num * BATCH_SIZE
            batch = all_candidates[batch_start:batch_start + BATCH_SIZE]
            logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch)} emails)...")

            batch_ids = [c['id'] for c in batch]
            current_labels_map = client.get_labels_for_emails(batch_ids)

            for log in batch:
                gid = log['id']
                if gid not in current_labels_map:
                    database.update_recheck_status(gid, log.get('ambiguous_labels'))
                    continue

                found_labels = current_labels_map[gid]

                trained_found = [lbl for lbl in found_labels if lbl in known_categories and lbl != config.VERIFICATION_LABEL]

                is_verified = config.VERIFICATION_LABEL in found_labels

                current_local = log['corrected_category'] or log['predicted_category']

                result = _resolve_correction(trained_found, is_verified, current_local)
                is_ambiguous = result["is_ambiguous"]
                correction_candidate = result["correction_candidate"]
                cleanup_needed = result["cleanup_needed"]
                verified_candidate = result["verified_candidate"]

                # Execute Actions
                if is_ambiguous:
                    logger.info(f"Ambiguous labels for {gid}: {trained_found}")
                    database.update_recheck_status(gid, ambiguous_labels=trained_found)
                    ambiguous_count += 1
                else:
                    if correction_candidate:
                        logger.info(f"Detected external correction for {gid}: {current_local} -> {correction_candidate}")
                        # Write training data FIRST, then update DB.
                        add_to_training_data(log, correction_candidate)
                        database.update_log_correction(gid, correction_candidate)
                        if cleanup_needed:
                            logger.info(f"Removing old label {current_local} from {gid}")
                            client.remove_label(gid, current_local)
                        # Mark as verified in IMAP (permanent marker)
                        if not is_verified:
                            logger.info(f"Adding {config.VERIFICATION_LABEL} to {gid}")
                            client.apply_label(gid, config.VERIFICATION_LABEL)
                        updates_count += 1

                    if verified_candidate:
                        logger.info(f"Verified correctness for {gid}: {verified_candidate}")
                        if not correction_candidate:
                            # Write training data FIRST, then update DB.
                            add_to_training_data(log, verified_candidate)
                            database.update_log_correction(gid, verified_candidate)
                            updates_count += 1

                    database.update_recheck_status(gid, ambiguous_labels=None)

            logger.info(f"Batch {batch_num + 1}/{total_batches} done. Running totals — Updates: {updates_count}, Ambiguous: {ambiguous_count}")

        logger.info(f"Force re-check finished. Total updates: {updates_count}, Total ambiguous: {ambiguous_count}")

    except Exception as e:
        logger.error(f"Error in force_check_corrections_job: {e}")
    finally:
        if client:
            client.disconnect()
        job_lock.release()

def backfill_training_data_job():
    """
    Rebuild training data files from all corrected entries in the database.
    Use this to recover training data if the training data directory was
    accidentally emptied or lost.

    Note: This appends to existing .jsonl files, so duplicates may be created
    if some entries already exist. The training pipeline should handle dedup.
    """
    logger.info("Starting backfill_training_data_job...")

    corrected_logs = database.get_all_corrected_logs()
    if not corrected_logs:
        logger.info("No corrected logs found in database. Nothing to backfill.")
        return

    logger.info(f"Backfilling training data from {len(corrected_logs)} corrected entries...")

    success_count = 0
    error_count = 0

    for log in corrected_logs:
        try:
            add_to_training_data(log, log['corrected_category'])
            success_count += 1
        except Exception as e:
            logger.error(f"Error backfilling training data for {log['id']}: {e}")
            error_count += 1

    logger.info(f"Backfill finished. Success: {success_count}, Errors: {error_count}")


# Models
class CorrectionRequest(BaseModel):
    corrected_category: str

class StatsResponse(BaseModel):
    stats: dict

class Notification(BaseModel):
    id: str  # Changed to str for Gmail ID
    timestamp: str
    sender: Optional[str]
    recipient: Optional[str] = None
    subject: Optional[str]
    predicted_category: Optional[str]
    confidence_score: Optional[float]
    is_read: Any

class AckRequest(BaseModel):
    ids: Optional[List[str]] = None

class RunResponse(BaseModel):
    status: str
    processed_count: int
    details: List[dict]

# Endpoints
@app.post("/run", response_model=RunResponse)
def run_classification(background_tasks: BackgroundTasks, limit: int = Query(20, description="Limit the number of emails to process")):
    """
    Manually trigger the classification job immediately.
    Optionally limit the number of emails processed (default: 20).
    Returns 'skipped' status if a job is already in progress.
    """
    # classification_job returns a list if successful, or a dict if skipped/error structure logic changes
    # We need to adapt the return type handling since classification_job now might return a dict for 'skipped'

    output = classification_job(limit=limit)

    if isinstance(output, dict) and output.get("status") == "skipped":
        return {
            "status": "skipped",
            "processed_count": 0,
            "details": []
        }

    # If output is list (results), it was successful
    return {
        "status": "success",
        "processed_count": len(output),
        "details": output
    }

@app.get("/stats", response_model=StatsResponse)
def get_stats(
    start_time: Optional[datetime.datetime] = None,
    end_time: Optional[datetime.datetime] = None
):
    """
    Get classification statistics (counts per category).
    Optionally filter by start_time and end_time (ISO format).
    """
    stats = database.get_stats(start_time, end_time)
    return {"stats": stats}

@app.get("/notifications", response_model=List[Notification])
def get_notifications():
    """
    Get all unread notifications.
    """
    notifs = database.get_unread_notifications()
    return notifs

@app.post("/notifications/ack")
def ack_notifications(req: AckRequest):
    """
    Acknowledge notifications (mark as read).
    If `ids` is provided, marks those specific IDs.
    If `ids` is empty or null, marks ALL unread notifications.
    """
    database.ack_notifications(req.ids)
    return {"status": "success"}

@app.post("/notifications/pop", response_model=List[Notification])
def pop_notifications():
    """
    Get all unread notifications AND mark them as read immediately.
    Useful for one-time fetch-and-ack clients.
    """
    notifs = database.pop_unread_notifications()
    return notifs

@app.get("/notifications/read", response_model=List[Notification])
def get_read_notifications(
    start_time: datetime.datetime,
    end_time: datetime.datetime
):
    """
    Get already read notifications within a time range.
    Start and end times are required.
    """
    notifs = database.get_read_notifications(start_time, end_time)
    return notifs

@app.get("/labels", response_model=List[str])
def get_labels():
    """
    Get all supported labels (categories) for email classification.
    """
    return classify.get_available_categories()

@app.get("/logs/ambiguous", dependencies=[Depends(get_api_key)])
def get_ambiguous_logs():
    """
    Get logs that have been flagged as ambiguous (multiple trained labels found on server).
    """
    return database.get_ambiguous_logs()

@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}

@app.post("/logs/{log_id}/correction", dependencies=[Depends(get_api_key)])
def correct_label(log_id: str, req: CorrectionRequest):
    """
    Correct the label for a specific email log.
    Updates the database and adds the email to training data.
    """
    # Validate category
    available_categories = classify.get_available_categories()
    if req.corrected_category not in available_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {req.corrected_category}. Must be one of: {', '.join(available_categories)}"
        )

    log_entry = database.get_log_by_id(log_id)
    if not log_entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    # Write training data FIRST, then update DB.
    # If training data write fails, the DB won't be updated.
    add_to_training_data(log_entry, req.corrected_category)
    database.update_log_correction(log_id, req.corrected_category)

    # Apply VERIFIED label in IMAP as permanent marker
    client = None
    try:
        client = imap_client.GmailClient()
        client.apply_label(log_id, req.corrected_category)
        # Remove old label if different
        old_label = log_entry.get('corrected_category') or log_entry.get('predicted_category')
        if old_label and old_label != req.corrected_category:
            client.remove_label(log_id, old_label)
        client.apply_label(log_id, config.VERIFICATION_LABEL)
    except Exception as e:
        logger.error(f"Failed to update IMAP labels for {log_id}: {e}")
    finally:
        if client:
            client.disconnect()

    return {"status": "success", "message": f"Label corrected to {req.corrected_category} and added to training data."}

@app.post("/admin/push-training-data", dependencies=[Depends(get_api_key)])
def trigger_push_training_data():
    """
    Manually trigger pushing training data to Git.
    Requires X-API-Key header if ADMIN_API_KEY is set in .env.
    """
    try:
        push_training_data_to_git()
        return {"status": "success", "message": "Training data push initiated."}
    except Exception as e:
        logger.error(f"Failed to push training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reclassify", dependencies=[Depends(get_api_key)])
def trigger_reclassify(background_tasks: BackgroundTasks, limit: int = Query(100, description="Limit emails to re-check")):
    """
    Trigger the re-classification process for existing logs.
    """
    # Run in background to avoid timeout
    background_tasks.add_task(reclassify_job, limit=limit)
    return {"status": "accepted", "message": "Re-classification started in background."}

@app.post("/admin/check-corrections", dependencies=[Depends(get_api_key)])
def trigger_check_corrections(background_tasks: BackgroundTasks):
    """
    Trigger the check corrections process for existing logs.
    """
    # Run in background to avoid timeout
    background_tasks.add_task(check_corrections_job)
    return {"status": "accepted", "message": "Check corrections started in background."}

# WARNING: This endpoint is expensive and should ONLY be used when you have
# manually re-labelled emails in Gmail and need to pick up those corrections
# immediately (e.g. to update training data before a model retrain).
# Do NOT use this for regular periodic checks — use /admin/check-corrections instead.
@app.post("/admin/force-check-corrections", dependencies=[Depends(get_api_key)])
def trigger_force_check_corrections(background_tasks: BackgroundTasks):
    """
    Force re-check ALL emails for label corrections, bypassing the gliding
    scale schedule. Use this after manually re-labelling emails in Gmail to
    update training data.
    """
    background_tasks.add_task(force_check_corrections_job)
    return {"status": "accepted", "message": "Force check corrections started in background."}

@app.post("/admin/backfill-training-data", dependencies=[Depends(get_api_key)])
def trigger_backfill_training_data(background_tasks: BackgroundTasks):
    """
    Rebuild training data files from all corrected entries in the database.
    Use this to recover training data if the training data directory was
    accidentally emptied or lost.
    """
    background_tasks.add_task(backfill_training_data_job)
    return {"status": "accepted", "message": "Backfill training data started in background."}

@app.post("/admin/trigger-update", dependencies=[Depends(get_api_key)])
def trigger_update(background_tasks: BackgroundTasks):
    """
    Manually trigger the update process.
    Requires X-API-Key header if ADMIN_API_KEY is set in .env.
    """
    try:
        push_training_data_to_git()
        Path(".update_request").touch()
        logger.info("Update requested via API. Server will restart.")
        background_tasks.add_task(shutdown_server)
        return {"status": "update_initiated", "message": "Server will shut down and update in a few seconds."}
    except Exception as e:
        logger.error(f"Failed to initiate update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/update-errors", dependencies=[Depends(get_api_key)])
def get_update_errors():
    """
    Get the history of update attempts and errors.
    Requires X-API-Key header if ADMIN_API_KEY is set in .env.
    """
    history_file = Path("update_history.json")
    if not history_file.exists():
        return []

    logs = []
    try:
        with open(history_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error(f"Error reading update history: {e}")
        raise HTTPException(status_code=500, detail="Could not read update history")

    return logs

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 for external access if needed
    uvicorn.run(app, host="0.0.0.0", port=8008)
