"""
api/models.py — Pydantic Request/Response Models
=================================================

Shared data models for all API endpoints.
"""

import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CorrectionRequest(BaseModel):
    corrected_category: str


class CandidateCategoryRequest(BaseModel):
    """A user-proposed new category for a log flagged with unknown Gmail labels."""
    category: str


class StatsResponse(BaseModel):
    stats: dict
    model_inputs_total: int = 0
    model_inputs_truncated: int = 0
    # Read-mail sweep prevalence: attempts / gate-blocks / fetched
    # candidates / labelled emails, per process lifetime (observability).
    read_sweep_attempts_total: int = 0
    read_sweep_gated_total: int = 0
    read_sweep_candidates_total: int = 0
    read_sweep_labeled_total: int = 0
    # Debug-round addition — self-diagnosis surface:
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


# ---------------------------------------------------------------------------
# Retrain-event status mirror (POST/GET /admin/retrain-events) — design §6.
#
# Field set per docs/RETRAIN-ORCHESTRATOR-DESIGN.md §6, validated strictly:
# unknown fields are rejected (extra="forbid") so a drift between the
# orchestrator app and this mirror surfaces as a 422, never as silently
# dropped data. `event`/`status` are closed enums — the mirror is a ledger,
# not free text. For skipped events the skip reason ("no new training
# data", "classifier unreachable") rides in `error` with status="skipped".
# ---------------------------------------------------------------------------

class RetrainEventRequest(BaseModel):
    """One event of one retraining run, as pushed by the orchestrator Mac.

    Idempotency on the POST endpoint: (run_id, event) is the uniqueness
    key — see api/routes/admin.py post_retrain_event.
    """
    model_config = ConfigDict(extra="forbid")

    event: Literal[
        "run_started", "run_finished", "run_failed",
        "pr_opened", "deploy_verified", "skipped",
    ]
    at: datetime
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    status: Literal["ok", "failed", "skipped"]
    eval: Optional[Dict[str, Any]] = None
    data_sha: Optional[str] = Field(default=None, pattern=r"^[a-f0-9]{7,64}$")
    artifact_shas: Optional[Dict[str, str]] = None
    durations: Optional[Dict[str, float]] = None
    error: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("artifact_shas")
    @classmethod
    def _check_artifact_names(cls, value: Optional[Dict[str, str]]):
        """Keys must be basename-only filenames (they name shipped model
        artifacts, never paths); values must be full lowercase sha256."""
        if value is None:
            return None
        for name, sha in value.items():
            if (not name or os.path.basename(name) != name
                    or name in (".", "..") or "/" in name or "\\" in name):
                raise ValueError(f"Artifact name must be a basename: {name!r}")
            if not re.fullmatch(r"[a-f0-9]{64}", sha or ""):
                raise ValueError(
                    f"artifact_shas['{name}'] is not a 64-char lowercase sha256"
                )
        return value

    @field_validator("durations")
    @classmethod
    def _check_durations(cls, value: Optional[Dict[str, float]]):
        """Step durations are finite, non-negative seconds."""
        if value is None:
            return None
        for step, seconds in value.items():
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(
                    f"durations['{step}'] must be a finite, non-negative number of seconds"
                )
        return value
