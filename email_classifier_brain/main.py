import logging
import os
from datetime import datetime
from typing import List, Optional, Any

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Local imports handling (standalone vs package)
try:
    import classify
    import database
    import imap_client
except ImportError:
    from classifier_brain import classify, database, imap_client

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
    # Run every 5 minutes
    scheduler.add_job(
        classification_job,
        trigger=IntervalTrigger(minutes=5),
        id="classification_job",
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduler started.")

    yield

    # Shutdown
    scheduler.shutdown()
    logger.info("Scheduler shutdown.")

app = FastAPI(title="Email Classifier Microservice", lifespan=lifespan)

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
                # Predict
                label, score = classify.predict_raw_email(msg, return_score=True)
                logger.info(f"Classified email {e_id!r}: {label} ({score:.2f})")

                # Apply label
                client.apply_label(e_id, label)

                # Log to DB
                sender = msg.get("From", "") or ""
                recipient = msg.get("To", "") or ""
                subject = msg.get("Subject", "") or ""

                # Extract date
                date_str = msg.get("Date")
                email_timestamp = None
                if date_str:
                    try:
                        from email.utils import parsedate_to_datetime
                        email_timestamp = parsedate_to_datetime(date_str)
                    except Exception:
                        logger.warning(f"Could not parse date: {date_str}")

                database.add_log(sender, recipient, subject, label, score, timestamp=email_timestamp)

                results.append({
                    "id": e_id.decode() if isinstance(e_id, bytes) else str(e_id),
                    "sender": sender,
                    "recipient": recipient,
                    "subject": subject,
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


# Models
class StatsResponse(BaseModel):
    stats: dict

class Notification(BaseModel):
    id: int
    timestamp: str
    sender: Optional[str]
    recipient: Optional[str] = None
    subject: Optional[str]
    predicted_category: Optional[str]
    confidence_score: Optional[float]
    is_read: Any

class AckRequest(BaseModel):
    ids: Optional[List[int]] = None

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
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
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
    start_time: datetime,
    end_time: datetime
):
    """
    Get already read notifications within a time range.
    Start and end times are required.
    """
    notifs = database.get_read_notifications(start_time, end_time)
    return notifs

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 for external access if needed
    uvicorn.run(app, host="0.0.0.0", port=8000)
