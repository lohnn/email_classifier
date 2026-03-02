"""
jobs/correction.py — Label Correction Jobs
===========================================

Background jobs that detect manual label corrections in Gmail and
sync them to the local database and training data.
"""

import logging
from email.utils import parsedate_to_datetime

import classify
import config
import database
import imap_client
from job_queue import job_queue
from jobs.training_data import add_to_training_data

logger = logging.getLogger(__name__)


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


def check_corrections_job(limit: int = 200, trigger: str = "scheduled"):
    """
    Background job to check for label corrections from the server (IMAP).
    Checks emails based on a gliding scale of age.
    """
    candidates = []
    updates_count = 0
    run_id = database.start_job_run("recheck", trigger)
    try:
        logger.info("Starting check_corrections_job...")

        candidates = database.get_candidate_logs_for_recheck(limit)
        if not candidates:
            logger.info("No candidates for re-check.")
            database.finish_job_run(run_id, "success", emails_processed=0, emails_updated=0)
            return

        logger.info(f"Checking {len(candidates)} emails for external corrections...")

        client = imap_client.gmail_client
        candidate_ids = [c['id'] for c in candidates]

        current_labels_map = client.get_labels_for_emails(candidate_ids)

        known_categories = set(classify.get_available_categories())

        updates_count = 0
        ambiguous_count = 0

        was_cancelled = False
        for log in candidates:
            if job_queue.is_cancelled():
                logger.info("Check corrections job cancelled.")
                was_cancelled = True
                break
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

            if is_ambiguous:
                logger.info(f"Ambiguous labels for {gid}: {trained_found}")
                database.update_recheck_status(gid, ambiguous_labels=trained_found)
                ambiguous_count += 1
            else:
                if correction_candidate:
                    logger.info(f"Detected external correction for {gid}: {current_local} -> {correction_candidate}")

                    add_to_training_data(log, correction_candidate)
                    database.update_log_correction(gid, correction_candidate)

                    if cleanup_needed:
                        logger.info(f"Removing old label {current_local} from {gid}")
                        client.remove_label(gid, current_local)

                    if not is_verified:
                        logger.info(f"Adding {config.VERIFICATION_LABEL} to {gid}")
                        client.apply_label(gid, config.VERIFICATION_LABEL)

                    updates_count += 1

                if verified_candidate:
                    logger.info(f"Verified correctness for {gid}: {verified_candidate}")

                    if not correction_candidate:
                        add_to_training_data(log, verified_candidate)
                        database.update_log_correction(gid, verified_candidate)
                        updates_count += 1

                database.update_recheck_status(gid, ambiguous_labels=None)

        logger.info(f"Re-check finished. Updates: {updates_count}, Ambiguous: {ambiguous_count}")
        final_status = "cancelled" if was_cancelled else "success"
        database.finish_job_run(run_id, final_status, emails_processed=len(candidates), emails_updated=updates_count)

    except Exception as e:
        logger.error(f"Error in check_corrections_job: {e}")
        database.finish_job_run(run_id, "error", emails_processed=len(candidates), emails_updated=updates_count, error_message=str(e))


def force_check_corrections_job(trigger: str = "scheduled"):
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

    run_id = database.start_job_run("force_recheck", trigger)
    total_processed = 0
    import_count = 0
    try:
        logger.info("Starting force_check_corrections_job (bypassing schedule)...")

        client = imap_client.gmail_client
        known_categories_list = classify.get_available_categories()
        known_categories = set(known_categories_list)

        # ---------------------------------------------------------------
        # Phase 0: Import labeled emails from IMAP that are missing in DB
        # ---------------------------------------------------------------
        logger.info("Phase 0: Scanning IMAP for labeled emails missing from DB...")
        labeled_emails = client.scan_labeled_emails(known_categories_list)
        import_count = 0

        for gid, (labels, msg) in labeled_emails.items():
            existing = database.get_log_by_id(gid)
            if existing:
                continue

            trained_labels = [lbl for lbl in labels if lbl in known_categories and lbl != config.VERIFICATION_LABEL]
            if len(trained_labels) != 1:
                if trained_labels:
                    logger.info(f"Skipping import of {gid}: ambiguous labels {trained_labels}")
                continue

            label = trained_labels[0]

            try:
                info = classify.extract_email_info(msg)

                date_str = msg.get("Date")
                email_timestamp = None
                if date_str:
                    try:
                        email_timestamp = parsedate_to_datetime(date_str)
                    except Exception:
                        logger.warning(f"Could not parse date for imported email {gid}: {date_str}")

                database.add_log(
                    id=gid,
                    sender=info["sender"],
                    recipient=info["to"],
                    subject=info["subject"],
                    predicted_category=label,
                    confidence_score=0.0,
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
        # Phase 1: Check corrections on all DB entries (in batches)
        # ---------------------------------------------------------------
        updates_count = 0
        ambiguous_count = 0
        batch_num = 0

        was_cancelled = False
        while True:
            if job_queue.is_cancelled():
                logger.info("Force check corrections job cancelled.")
                was_cancelled = True
                break
            batch = database.get_all_logs_for_recheck(limit=BATCH_SIZE, offset=batch_num * BATCH_SIZE)
            if not batch:
                if batch_num == 0:
                    logger.info("No candidates for forced re-check.")
                break

            batch_num += 1
            total_processed += len(batch)
            logger.info(f"Phase 1: Processing batch {batch_num} ({len(batch)} emails)...")

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

                if is_ambiguous:
                    logger.info(f"Ambiguous labels for {gid}: {trained_found}")
                    database.update_recheck_status(gid, ambiguous_labels=trained_found)
                    ambiguous_count += 1
                else:
                    if correction_candidate:
                        logger.info(f"Detected external correction for {gid}: {current_local} -> {correction_candidate}")
                        add_to_training_data(log, correction_candidate)
                        database.update_log_correction(gid, correction_candidate)
                        if cleanup_needed:
                            logger.info(f"Removing old label {current_local} from {gid}")
                            client.remove_label(gid, current_local)
                        if not is_verified:
                            logger.info(f"Adding {config.VERIFICATION_LABEL} to {gid}")
                            client.apply_label(gid, config.VERIFICATION_LABEL)
                        updates_count += 1

                    if verified_candidate:
                        logger.info(f"Verified correctness for {gid}: {verified_candidate}")
                        if not correction_candidate:
                            add_to_training_data(log, verified_candidate)
                            database.update_log_correction(gid, verified_candidate)
                            updates_count += 1

                    database.update_recheck_status(gid, ambiguous_labels=None)

            logger.info(f"Batch {batch_num} done. Running totals — Updates: {updates_count}, Ambiguous: {ambiguous_count}")

        logger.info(f"Force re-check finished. Total updates: {updates_count}, Total ambiguous: {ambiguous_count}")
        final_status = "cancelled" if was_cancelled else "success"
        database.finish_job_run(run_id, final_status, emails_processed=import_count + total_processed, emails_updated=import_count + updates_count)

    except Exception as e:
        logger.error(f"Error in force_check_corrections_job: {e}")
        database.finish_job_run(run_id, "error", emails_processed=import_count + total_processed, error_message=str(e))
