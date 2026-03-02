"""
api/routes/classification.py — Classification Endpoints
========================================================

Endpoints for triggering email classification and re-classification,
and for listing available labels.
"""

import logging
from typing import List

import classify
from fastapi import APIRouter, Depends, Query

from api.models import RunResponse
from api.security import get_api_key
from job_queue import job_queue
from jobs.classification import classification_job
from jobs.reclassify import reclassify_job

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/run", response_model=RunResponse, dependencies=[Depends(get_api_key)])
def run_classification(limit: int = Query(20, description="Limit the number of emails to process")):
    """
    Manually trigger the classification job immediately.
    Optionally limit the number of emails processed (default: 20).
    """
    accepted = job_queue.enqueue("classification", classification_job, limit=limit, trigger="manual")
    if accepted:
        return {"status": "accepted", "message": "Classification job queued."}
    else:
        return {"status": "already_queued", "message": "Classification job is already running or queued."}


@router.post("/reclassify", dependencies=[Depends(get_api_key)])
def trigger_reclassify(limit: int = Query(100, description="Limit emails to re-check")):
    """
    Trigger the re-classification process for existing logs.
    """
    accepted = job_queue.enqueue("reclassify", reclassify_job, limit=limit, trigger="manual")
    if accepted:
        return {"status": "accepted", "message": "Re-classification queued."}
    return {"status": "already_queued", "message": "Job already queued or running."}


@router.get("/labels", response_model=List[str], dependencies=[Depends(get_api_key)])
def get_labels():
    """
    Get all supported labels (categories) for email classification.
    """
    return classify.get_available_categories()
