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

        # Fetch emails
        # Note: A limit could be implemented in fetch_unprocessed_emails to avoid locking too long
        emails = client.fetch_unprocessed_emails(known_labels)
        logger.info(f"Found {len(emails)} unprocessed emails.")

        # Simple limiting (though fetch still gets all headers)
        if len(emails) > limit:
            logger.info(f"Limiting processing to first {limit} emails.")
            emails = emails[:limit]

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
        "subject": log_entry.get("subject", ""),
        "body": log_entry.get("body", ""),
        "from": log_entry.get("sender", ""),
        "to": log_entry.get("recipient", ""),
        "cc": log_entry.get("cc", ""),
        "mass_mail": bool(log_entry.get("mass_mail", False)),
        "attachment_types": att_types
    }

    # Ensure TRAINING_DATA_DIR exists
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

    file_path = os.path.join(TRAINING_DATA_DIR, f"{corrected_category}.jsonl")

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

    # Update database
    database.update_log_correction(log_id, req.corrected_category)

    # Add to training data
    add_to_training_data(log_entry, req.corrected_category)

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
