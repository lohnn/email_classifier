"""
api/models.py — Pydantic Request/Response Models
=================================================

Shared data models for all API endpoints.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel


class CorrectionRequest(BaseModel):
    corrected_category: str


class StatsResponse(BaseModel):
    stats: dict


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
