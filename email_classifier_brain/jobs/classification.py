"""
jobs/classification.py — Email Classification Job
==================================================

Fetches unprocessed emails from IMAP, classifies them, applies
labels, and logs results to the database.
"""

import logging
from email.utils import parsedate_to_datetime

import classify
import config
import database
import imap_client
from job_queue import job_queue

logger = logging.getLogger(__name__)


def classification_job(limit: int = 20, trigger: str = "scheduled"):
    logger.info("Starting classification job...")
    run_id = database.start_job_run("classification", trigger)
    results = []
    client = None
    emails_processed = 0
    error_count = 0
    try:
        # Connect to IMAP
        client = imap_client.GmailClient()

        # Get known categories to skip
        known_labels = classify.get_available_categories()

        # Fetch emails, stopping early once we have enough
        emails = client.fetch_unprocessed_emails(known_labels, limit=limit)
        logger.info(f"Fetched {len(emails)} unprocessed emails (limit={limit}).")
        emails_processed = len(emails)

        was_cancelled = False
        for e_id, msg in emails:
            if job_queue.is_cancelled():
                logger.info("Classification job cancelled.")
                was_cancelled = True
                break
            try:
                # Extract full info
                info = classify.extract_email_info(msg)

                # Predict
                label, score, is_unsure = classify.predict_email(
                    subject=info["subject"],
                    body=info["body"],
                    sender=info["sender"],
                    to=info["to"],
                    cc=info["cc"],
                    mass_mail=info["mass_mail"],
                    attachment_types=info["attachment_types"],
                    return_score=True
                )
                logger.info(f"Classified email {e_id}: {label} ({score:.2f}){' [UNSURE]' if is_unsure else ''}")

                # Apply primary label
                client.apply_label(e_id, label)

                # Apply unsure label if classifier is not confident
                if is_unsure:
                    client.apply_label(e_id, config.UNSURE_LABEL)

                # Extract date
                date_str = msg.get("Date")
                email_timestamp = None
                if date_str:
                    try:
                        email_timestamp = parsedate_to_datetime(date_str)
                    except Exception:
                        logger.warning(f"Could not parse date: {date_str}")

                # Log to DB with full info
                database.add_log(
                    id=e_id,
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
                error_count += 1

        logger.info("Classification job finished.")
        final_status = "cancelled" if was_cancelled else "success"
        database.finish_job_run(run_id, final_status, emails_processed=emails_processed, emails_updated=len(results), error_count=error_count)
        return results

    except Exception as e:
        logger.error(f"Error in classification job: {e}")
        database.finish_job_run(run_id, "error", emails_processed=emails_processed, error_count=error_count, error_message=str(e))
        return []
    finally:
        if client:
            client.disconnect()
