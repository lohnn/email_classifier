"""
jobs/classification.py — Email Classification Job
==================================================

Fetches unprocessed emails from IMAP, classifies them, applies
labels, and logs results to the database.

Two strictly ordered phases (WI-103):
  1. UNSEEN phase — classify new mail (as always).
  2. Read sweep — label older already-read mail that was never
     classified, HARD-GATED behind a fully drained unread queue, at a
     bounded per-cycle pace (config.ENABLE_READ_SWEEP /
     config.READ_SWEEP_BATCH_SIZE, conservatively OFF / 5 by default).
     Unread always wins; the phases never interleave.
"""

import logging
from email.utils import parsedate_to_datetime

import classify
import config
import database
import imap_client
from job_queue import job_queue

logger = logging.getLogger(__name__)


def _classify_and_log(client, e_id: str, msg, results: list) -> bool:
    """
    Classify one email end-to-end: predict, apply label(s) in Gmail, log to
    the database. Returns True iff the email was classified AND its label
    application succeeded.

    The DB log row is the system's processed-marker for the read sweep
    (WI-103), so it is written ONLY after Gmail accepted the label: a read
    email whose label never lands must stay eligible for the next sweep
    cycle instead of being marked processed while unlabeled (the DB-says-
    labeled / Gmail-says-nothing drift class).
    """
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
        applied = client.apply_label(e_id, label)

        # Apply unsure label if classifier is not confident
        if is_unsure:
            applied = bool(client.apply_label(e_id, config.UNSURE_LABEL)) and applied

        if not applied:
            logger.error(
                f"Label application failed for {e_id} ({label!r}); DB log not "
                f"written — email stays eligible for the next cycle."
            )
            return False

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
        return True
    except Exception as e:
        logger.error(f"Error processing email {e_id}: {e}")
        return False


def _read_sweep_phase(client, known_labels, results) -> int:
    """
    WI-103 read-mail backfill phase. Returns the number of read emails
    fetched (candidates) this phase.

    HARD GATE — "unread queue fully drained", enforced in code:
      the phase runs only when client.has_unprocessed_unseen_emails() is
      False, i.e. NO UNSEEN email in the INBOX selection lacks all of the
      trained-category labels. The probe is a fresh metadata query issued
      AFTER the unread phase completes, so mail that arrived mid-cycle
      re-blocks the sweep. The probe fails closed (see its docstring).

    Resumability: candidate emails carrying a known label are excluded by
    the IMAP metadata scan (never re-labelled — corrections MOVE, they are
    not overwritten), and zero-label read emails the system already
    processed once are excluded via database.get_all_log_ids() (the
    processed-marker) passed as skip_ids. The sweep keeps no cursor of its
    own: after a restart it simply re-derives the remaining history from
    Gmail labels + DB rows.

    Never raises when the client is well-behaved: unexpected failures from
    the sweep itself are swallowed by the caller so the sweep can never
    fail a job run whose unread phase succeeded.
    """
    config.sweep_stats.record(attempt=True)  # one attempt per enabled cycle

    if not known_labels:
        logger.info("Read sweep skipped: no trained categories available yet.")
        return 0

    # --- hard gate: unread always wins -----------------------------------
    if client.has_unprocessed_unseen_emails(known_labels):
        config.sweep_stats.record(gated=True)
        logger.info("Read sweep skipped: unread queue is not fully drained.")
        return 0

    batch_size = config.READ_SWEEP_BATCH_SIZE
    skip_ids = database.get_all_log_ids()
    emails = client.fetch_read_unprocessed_emails(
        known_labels, limit=batch_size, skip_ids=skip_ids,
    )
    config.sweep_stats.record(candidates=len(emails))
    logger.info(
        f"Read sweep: {len(emails)} read emails eligible for labelling "
        f"(batch={batch_size}, {len(skip_ids)} already-processed ids)."
    )

    labeled = 0
    for e_id, msg in emails:
        if job_queue.is_cancelled():
            logger.info("Read sweep cancelled.")
            break
        if _classify_and_log(client, e_id, msg, results):
            labeled += 1
    config.sweep_stats.record(labeled=labeled)
    return len(emails)


def classification_job(limit: int = 20, trigger: str = "scheduled"):
    logger.info("Starting classification job...")
    run_id = database.start_job_run("classification", trigger)
    results = []
    emails_processed = 0
    error_count = 0
    try:
        client = imap_client.gmail_client

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
            if not _classify_and_log(client, e_id, msg, results):
                error_count += 1

        # ------------------------------------------------------------------
        # WI-103 read-mail sweep — hard-gated, paced, never interleaved with
        # the unread phase. Runs only if the unread phase wasn't cancelled.
        # A sweep failure must not fail a run whose unread phase succeeded.
        # ------------------------------------------------------------------
        if config.ENABLE_READ_SWEEP and not was_cancelled:
            try:
                emails_processed += _read_sweep_phase(client, known_labels, results)
            except Exception as e:
                logger.error(f"Read sweep phase failed: {e}")
                error_count += 1

        logger.info("Classification job finished.")
        final_status = "cancelled" if was_cancelled else "success"
        database.finish_job_run(run_id, final_status, emails_processed=emails_processed, emails_updated=len(results), error_count=error_count)
        return results

    except Exception as e:
        logger.error(f"Error in classification job: {e}")
        database.finish_job_run(run_id, "error", emails_processed=emails_processed, error_count=error_count, error_message=str(e))
        return []
