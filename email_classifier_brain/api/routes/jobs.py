"""
api/routes/jobs.py — Job Queue Endpoints
=========================================

Endpoints for monitoring and controlling the background job queue.
"""

import logging
from typing import List, Optional

import database
from fastapi import APIRouter, Depends, Query

from api.models import CancelResponse, JobRunEntry, JobStatusResponse
from api.security import get_api_key
from job_queue import job_queue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs")


@router.get("/status", response_model=JobStatusResponse, dependencies=[Depends(get_api_key)])
def get_jobs_status():
    """
    Return the current job queue state: what is running and what is waiting.

    Each entry includes ``name``, ``enqueued_at``, and ``started_at`` (ISO 8601 UTC).
    ``started_at`` is only set for the currently running job.
    """
    snapshot = job_queue.status()
    return snapshot


@router.post("/cancel", response_model=CancelResponse, dependencies=[Depends(get_api_key)])
def cancel_jobs():
    """
    Cancel the currently running job (if any) and clear the pending queue.

    Cancellation is cooperative: the running job is asked to stop at its next
    iteration checkpoint. Jobs that do not reach a checkpoint (e.g. they are
    blocked on a network call) will finish that step before exiting.

    Response ``status`` values:
    - ``"cancelling"`` — a job was running and has been signalled to stop
    - ``"cleared"``    — no job was running, but pending jobs were removed
    - ``"idle"``       — nothing was running or queued
    """
    result = job_queue.cancel()
    running = result["cancelled_job"]
    cleared = result["cleared_queue"]

    if running:
        status = "cancelling"
    elif cleared:
        status = "cleared"
    else:
        status = "idle"

    return {"status": status, "cancelled_job": running, "cleared_queue": cleared}


@router.get("/history", response_model=List[JobRunEntry], dependencies=[Depends(get_api_key)])
def get_jobs_history(
    limit: int = Query(50, description="Maximum number of records to return"),
    job_name: Optional[str] = Query(None, description="Filter by job name (e.g. 'classification', 'recheck')")
):
    """
    Return per-run metadata for completed and in-progress jobs.

    Fields: ``job_name``, ``trigger`` (scheduled/manual), ``started_at``,
    ``finished_at``, ``duration_seconds``, ``status``, ``emails_processed``,
    ``emails_updated``, ``error_count``, ``error_message``.
    """
    return database.get_job_runs(limit=limit, job_name=job_name)
