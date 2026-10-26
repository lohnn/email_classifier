"""
api/models.py — Pydantic Request/Response Models
=================================================

Shared data models for all API endpoints.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel


class CorrectionRequest(BaseModel):
    corrected_category: str


class CandidateCategoryRequest(BaseModel):
    """A user-proposed new category for a log flagged with unknown Gmail labels."""
    category: str


class StatsResponse(BaseModel):
    stats: dict
    model_inputs_total: int = 0
    model_inputs_truncated: int = 0
    # Read-mail sweep prevalence (WI-103): attempts / gate-blocks / fetched
    # candidates / labelled emails, per process lifetime (W-022 observability).
    read_sweep_attempts_total: int = 0
    read_sweep_gated_total: int = 0
    read_sweep_candidates_total: int = 0
    read_sweep_labeled_total: int = 0
    # WI-103 debug round — self-diagnosis surface:
    read_sweep_mailbox_last: str = ""          # mailbox actually searched by the last sweep fetch
    read_sweep_last_skip_reason: str = "none"  # none|no_trained_categories|gated_unread|disabled
    read_sweep_skipped_no_categories_total: int = 0
    read_sweep_trained_categories_count: int = 0


class Notification(BaseModel):
    id: str
    timestamp: str
    sender: Optional[str]
    recipient: Optional[str] = None
    subject: Optional[str]
    predicted_category: Optional[str]
    confidence_score: Optional[float]
    is_read: bool


class AckRequest(BaseModel):
    ids: Optional[List[str]] = None


class RunResponse(BaseModel):
    status: str
    message: str


class JobStatusEntry(BaseModel):
    name: str
    enqueued_at: Optional[str]
    started_at: Optional[str]


class JobStatusResponse(BaseModel):
    running: Optional[JobStatusEntry]
    queued: List[JobStatusEntry]


class CancelResponse(BaseModel):
    status: Literal["cancelling", "cleared", "idle"]
    cancelled_job: Optional[str]
    cleared_queue: List[str]


class JobRunEntry(BaseModel):
    id: int
    job_name: str
    trigger: str
    started_at: str
    finished_at: Optional[str]
    duration_seconds: Optional[float]
    status: str
    emails_processed: Optional[int]
    emails_updated: Optional[int]
    error_count: int
    error_message: Optional[str]
