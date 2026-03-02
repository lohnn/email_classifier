"""
jobs/reclassify.py — Re-classification Job
===========================================

Background job to re-run predictions on existing log entries
using the current model, updating labels in Gmail and the database
when the prediction changes.
"""

import datetime
import logging

import classify
import config
import database
import imap_client
from job_queue import job_queue

logger = logging.getLogger(__name__)


def reclassify_job(limit: int = 100, trigger: str = "scheduled"):
    """
    Background job to re-classify existing logs.
    """
    logs = []
    updated_count = 0
    errors = 0
    run_id = database.start_job_run("reclassify", trigger)

    try:
        logger.info("Starting re-classification job...")
        logs = database.get_logs_for_reclassification(limit=limit)

        client = imap_client.gmail_client

        was_cancelled = False
        for log in logs:
            if job_queue.is_cancelled():
                logger.info("Reclassify job cancelled.")
                was_cancelled = True
                break
            gmail_id = log['id']
            current_label = log['predicted_category']

            try:
                msg = client.fetch_email_by_gmail_id(gmail_id)

                info = None
                if msg:
                    info = classify.extract_email_info(msg)
                else:
                    logger.warning(f"Could not fetch email {gmail_id} from Gmail. Skipping.")
                    continue

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

                if label != current_label:
                    logger.info(f"Re-classification change for {gmail_id}: {current_label} -> {label} ({score:.2f}){' [UNSURE]' if is_unsure else ''}")

                    if current_label:
                        client.remove_label(gmail_id, current_label)
                    client.apply_label(gmail_id, label)

                    orig_ts = None
                    if log['timestamp']:
                        try:
                            orig_ts = datetime.datetime.fromisoformat(log['timestamp'])
                        except Exception:
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

                # Sync UNSURE label regardless of whether primary label changed
                if is_unsure:
                    client.apply_label(gmail_id, config.UNSURE_LABEL)
                else:
                    client.remove_label(gmail_id, config.UNSURE_LABEL)

                database.update_reclassified_at(gmail_id)

            except Exception as e:
                logger.error(f"Error re-classifying {gmail_id}: {e}")
                errors += 1

        logger.info(f"Re-classification finished. Updated {updated_count} emails.")
        final_status = "cancelled" if was_cancelled else "success"
        database.finish_job_run(run_id, final_status, emails_processed=len(logs), emails_updated=updated_count, error_count=errors)
        return {
            "status": "success",
            "processed": len(logs),
            "updated": updated_count,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Error in re-classification job: {e}")
        database.finish_job_run(run_id, "error", emails_processed=len(logs), error_count=errors, error_message=str(e))
        return {"status": "error", "message": str(e)}
