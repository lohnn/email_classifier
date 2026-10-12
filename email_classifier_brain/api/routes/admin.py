"""
api/routes/admin.py — Admin & Log Management Endpoints
=======================================================

Protected endpoints for manual job triggers, label correction,
training data management, and update control.
"""

import json
import logging
from pathlib import Path

import classify
import config
import database
import imap_client
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from api.models import CandidateCategoryRequest, CorrectionRequest
from api.security import get_api_key
from job_queue import job_queue
from jobs.correction import check_corrections_job, force_check_corrections_job
from jobs.training_data import add_to_training_data, backfill_training_data_job, push_training_data_to_git
from jobs.update import get_retrain_status, shutdown_server

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/logs/{log_id}/correction", dependencies=[Depends(get_api_key)])
def correct_label(log_id: str, req: CorrectionRequest):
    """
    Correct the label for a specific email log.
    Updates the database and moves the email in the training data: the
    example is written to the new category's file and any copy under the
    previous category is removed (correction is a MOVE, not an append).
    """
    available_categories = classify.get_available_categories()
    if req.corrected_category not in available_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {req.corrected_category}. Must be one of: {', '.join(available_categories)}"
        )

    log_entry = database.get_log_by_id(log_id)
    if not log_entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    old_label = log_entry.get('corrected_category') or log_entry.get('predicted_category')

    # Write training data FIRST, then update DB.
    # If training data write fails, the DB won't be updated.
    add_to_training_data(log_entry, req.corrected_category, previous_category=old_label)
    database.update_log_correction(log_id, req.corrected_category)

    # Apply VERIFIED label in IMAP as permanent marker
    client = None
    try:
        client = imap_client.GmailClient()
        client.apply_label(log_id, req.corrected_category)
        if old_label and old_label != req.corrected_category:
            client.remove_label(log_id, old_label)
        # A human has resolved the label — clear the classifier's UNSURE flag
        # (classifier-owned only: the user's own labels are never touched)
        client.remove_label(log_id, config.UNSURE_LABEL)
        client.apply_label(log_id, config.VERIFICATION_LABEL)
    except Exception as e:
        logger.error(f"Failed to update IMAP labels for {log_id}: {e}")
    finally:
        if client:
            client.disconnect()

    return {"status": "success", "message": f"Label corrected to {req.corrected_category} and added to training data."}


@router.get("/logs/ambiguous", dependencies=[Depends(get_api_key)])
def get_ambiguous_logs():
    """
    Get logs that have been flagged as ambiguous (multiple trained labels found on server).
    """
    return database.get_ambiguous_logs()


def _validate_new_category(category: str) -> str:
    """
    Validate a user-proposed new category before admitting it as a training
    file. Must map to a safe single jsonl path (train.py derives the label
    from the file path; '/' creates hierarchy, path traversal is rejected).
    """
    category = (category or "").strip()
    if not category:
        raise HTTPException(status_code=400, detail="Category must not be empty")
    parts = category.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise HTTPException(status_code=400, detail=f"Invalid category path: {category!r}")
    if category in classify.get_available_categories():
        raise HTTPException(status_code=400, detail=f"Category {category} is already known to the model")
    return category


@router.get("/logs/candidate-categories", dependencies=[Depends(get_api_key)])
def get_candidate_categories():
    """
    Get logs flagged by the recheck job as carrying Gmail labels the model
    has never trained on (candidate new categories the user tagged manually).
    """
    return database.get_candidate_category_logs()


@router.post("/logs/candidate-categories/{log_id}/accept", dependencies=[Depends(get_api_key)])
def accept_candidate_category(log_id: str, req: CandidateCategoryRequest):
    """
    Consciously admit a candidate category: writes the email to
    TrainingData/<category>.jsonl (the corpus starts teaching it), records the
    correction, clears the candidate flag, and marks the email VERIFIED in
    Gmail. The new category only takes effect for classification after the
    next retrain (the served model's label set is rebuilt from the corpus).
    """
    log_entry = database.get_log_by_id(log_id)
    if not log_entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    category = _validate_new_category(req.category)

    add_to_training_data(log_entry, category)
    database.update_log_correction(log_id, category)
    database.clear_candidate_labels(log_id)

    client = None
    try:
        client = imap_client.GmailClient()
        client.apply_label(log_id, category)
        client.apply_label(log_id, config.VERIFICATION_LABEL)
    except Exception as e:
        logger.error(f"Failed to update IMAP labels for {log_id}: {e}")
    finally:
        if client:
            client.disconnect()

    return {
        "status": "success",
        "category": category,
        "message": f"Category {category} admitted to training data; classification adopts it after the next retrain.",
    }


@router.post("/logs/candidate-categories/{log_id}/discard", dependencies=[Depends(get_api_key)])
def discard_candidate_category(log_id: str):
    """
    Discard a candidate new-category flag without touching training data or
    Gmail labels. The recheck job may re-flag the email on a later pass.
    """
    cleared = database.clear_candidate_labels(log_id)
    if not cleared:
        raise HTTPException(status_code=404, detail="No candidate category flag on this log")
    return {"status": "success", "message": "Candidate category flag discarded."}


@router.post("/admin/push-training-data", dependencies=[Depends(get_api_key)])
def trigger_push_training_data():
    """
    Manually trigger pushing training data to Git.
    """
    try:
        push_ok = push_training_data_to_git()
        if not push_ok:
            raise RuntimeError("Training data git push failed (see service logs / update_history.json)")
        return {"status": "success", "message": "Training data push initiated."}
    except Exception as e:
        logger.error(f"Failed to push training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/retrain-status", dependencies=[Depends(get_api_key)])
def retrain_status():
    """
    Whether the training data has commits the deployed model was never
    trained on ('retrain worthwhile'), with both commit ids. Training is
    never triggered from here — ./retrain.sh on the trainer machine is the
    invocation path; this is the readable signal.
    """
    return get_retrain_status()


@router.post("/admin/check-corrections", dependencies=[Depends(get_api_key)])
def trigger_check_corrections():
    """
    Trigger the check corrections process for existing logs.
    """
    accepted = job_queue.enqueue("recheck", check_corrections_job, trigger="manual")
    if accepted:
        return {"status": "accepted", "message": "Check corrections queued."}
    return {"status": "already_queued", "message": "Job already queued or running."}


# WARNING: This endpoint is expensive and should ONLY be used when you have
# manually re-labelled emails in Gmail and need to pick up those corrections
# immediately (e.g. to update training data before a model retrain).
# Do NOT use this for regular periodic checks — use /admin/check-corrections instead.
@router.post("/admin/force-check-corrections", dependencies=[Depends(get_api_key)])
def trigger_force_check_corrections():
    """
    Force re-check ALL emails for label corrections, bypassing the gliding
    scale schedule. Use this after manually re-labelling emails in Gmail to
    update training data.
    """
    accepted = job_queue.enqueue("force_recheck", force_check_corrections_job, trigger="manual")
    if accepted:
        return {"status": "accepted", "message": "Force check corrections queued."}
    return {"status": "already_queued", "message": "Job already queued or running."}


@router.post("/admin/backfill-training-data", dependencies=[Depends(get_api_key)])
def trigger_backfill_training_data():
    """
    Rebuild training data files from all corrected entries in the database.
    Use this to recover training data if the training data directory was
    accidentally emptied or lost.
    """
    accepted = job_queue.enqueue("backfill", backfill_training_data_job, trigger="manual")
    if accepted:
        return {"status": "accepted", "message": "Backfill training data queued."}
    return {"status": "already_queued", "message": "Job already queued or running."}


@router.post("/admin/trigger-update", dependencies=[Depends(get_api_key)])
def trigger_update(background_tasks: BackgroundTasks):
    """
    Manually trigger the update process.
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


@router.get("/admin/update-errors", dependencies=[Depends(get_api_key)])
def get_update_errors():
    """
    Get the history of update attempts and errors.
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
