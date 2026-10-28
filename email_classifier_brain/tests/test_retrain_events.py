"""
tests/test_retrain_events.py — Retrain-event status mirror endpoints
====================================================================

POST/GET /admin/retrain-events + retrain_store.py (design §6, storage
decision 2026-10-28: retrain_events.json JSONL in the service workdir,
~50 newest kept; one human-readable update_history.json line per recorded
event).

Mirror files are patched into tmp dirs per test — the repo checkout's
service workdir must never receive test events. DB is patched per the
test_app.py convention.
"""

import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

import database  # noqa: E402
import jobs.update  # noqa: E402
import retrain_store  # noqa: E402
from main import app  # noqa: E402
from retrain_store import MAX_RETRAIN_EVENTS  # noqa: E402

ADMIN = {"X-API-Key": "testkey"}


def _event(run_id="run-20261028-a", event="run_finished", **overrides):
    """A valid event body; overrides replace fields, `del` via None."""
    body = {
        "event": event,
        "at": "2026-10-28T12:00:00Z",
        "run_id": run_id,
        "status": "ok",
        "eval": {"accuracy": 0.91},
        "data_sha": "a" * 40,
        "artifact_shas": {"merged_text_q8_0.gguf": "b" * 64},
        "durations": {"train": 640.0, "embed": 55.5},
        "error": None,
    }
    for key, value in overrides.items():
        if value is None:
            body.pop(key, None)
        else:
            body[key] = value
    return body


@pytest.fixture
def client(tmp_path):
    """TestClient with the mirror files (and DB) redirected into tmp_path."""
    events_file = tmp_path / "retrain_events.json"
    history_file = tmp_path / "update_history.json"
    with patch.object(retrain_store, "RETRAIN_EVENTS_FILE", str(events_file)), \
            patch.object(jobs.update, "HISTORY_FILE", str(history_file)), \
            patch("database.DB_FILE", str(tmp_path / "test.db")):
        database.init_db()
        with TestClient(app) as c:
            yield c, events_file, history_file


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# GET shape
# ---------------------------------------------------------------------------

def test_get_empty_mirror(client):
    c, _, _ = client
    response = c.get("/admin/retrain-events", headers=ADMIN)
    assert response.status_code == 200
    assert response.json() == {"latest": None, "history": []}


def test_get_shape_after_post(client):
    c, _, _ = client
    post = c.post("/admin/retrain-events", json=_event(), headers=ADMIN)
    assert post.status_code == 200

    data = c.get("/admin/retrain-events", headers=ADMIN).json()
    assert data["latest"]["event"] == "run_finished"
    assert data["latest"]["run_id"] == "run-20261028-a"
    assert data["history"] == [data["latest"]]  # latest == history[0]


# ---------------------------------------------------------------------------
# POST recording + idempotency
# ---------------------------------------------------------------------------

def test_post_records_event_and_human_line(client):
    c, events_file, history_file = client
    body = _event(
        event="skipped", status="skipped", error="no new training data",
        eval=None, artifact_shas=None, durations=None,
    )
    post = c.post("/admin/retrain-events", json=body, headers=ADMIN)
    assert post.status_code == 200
    payload = post.json()
    assert payload["status"] == "recorded"
    assert payload["duplicate"] is False
    assert payload["event"]["status"] == "skipped"

    # Mirror file: exactly one event line, byte-shaped as pushed
    events = _read_jsonl(events_file)
    assert len(events) == 1
    assert events[0]["event"] == "skipped"
    assert events[0]["error"] == "no new training data"

    # Human file: exactly one line naming the skip reason
    lines = history_file.read_text().splitlines()
    assert len(lines) == 1
    assert "skipped" in lines[0].lower()
    assert "no new training data" in lines[0]


def test_post_duplicate_run_id_event_is_noop(client):
    """Retried POST of the same (run_id, event): first record stands,
    nothing appended, human file untouched."""
    c, events_file, history_file = client
    first = c.post("/admin/retrain-events", json=_event(), headers=ADMIN).json()
    later = c.post(
        "/admin/retrain-events",
        json=_event(at="2026-10-29T09:00:00Z", status="failed", error="boom"),
        headers=ADMIN,
    ).json()

    assert first["status"] == "recorded"
    assert later["status"] == "already_recorded"
    assert later["duplicate"] is True
    assert later["event"]["at"] == first["event"]["at"]  # ORIGINAL record
    assert later["event"]["status"] == "ok"

    events = _read_jsonl(events_file)
    assert len(events) == 1  # no second line
    assert len(history_file.read_text().splitlines()) == 1  # no extra human line


def test_same_run_id_different_events_both_record(client):
    """A run's lifecycle events share run_id but each records once."""
    c, events_file, history_file = client
    c.post("/admin/retrain-events", json=_event(event="run_started"), headers=ADMIN)
    c.post("/admin/retrain-events", json=_event(event="run_finished"), headers=ADMIN)
    events = _read_jsonl(events_file)
    assert [ev["event"] for ev in events] == ["run_started", "run_finished"]
    assert len(history_file.read_text().splitlines()) == 2

    data = c.get("/admin/retrain-events", headers=ADMIN).json()
    assert data["latest"]["event"] == "run_finished"
    assert [ev["event"] for ev in data["history"]] == ["run_finished", "run_started"]  # newest first


# ---------------------------------------------------------------------------
# Strict validation (extra=forbid, closed enums, malformed values)
# ---------------------------------------------------------------------------

def _post_and_expect_422(c, body):
    response = c.post("/admin/retrain-events", json=body, headers=ADMIN)
    assert response.status_code == 422, response.text
    return response


def test_reject_unknown_field(client):
    c, _, _ = client
    body = _event()
    body["extra_field"] = "drift"  # schema drift between app and mirror must be loud
    _post_and_expect_422(c, body)


def test_reject_unknown_event_and_status(client):
    c, _, _ = client
    _post_and_expect_422(c, _event(event="something_happened"))
    _post_and_expect_422(c, _event(status="meh"))


def test_reject_malformed_run_id_and_data_sha(client):
    c, _, _ = client
    _post_and_expect_422(c, _event(run_id="bad id with spaces"))
    _post_and_expect_422(c, _event(data_sha="XYZ123"))        # not lowercase hex
    _post_and_expect_422(c, _event(data_sha="abc"))           # too short for a git sha


def test_reject_unsafe_artifact_name_and_bad_sha_value(client):
    c, _, _ = client
    _post_and_expect_422(c, _event(artifact_shas={"../evil.gguf": "b" * 64}))
    _post_and_expect_422(c, _event(artifact_shas={"m.gguf": "short"}))


def test_reject_bad_durations_and_error_length(client):
    c, _, _ = client
    _post_and_expect_422(c, _event(durations={"train": -1.0}))
    _post_and_expect_422(c, _event(error="x" * 2001))


def test_reject_bad_timestamp(client):
    c, _, _ = client
    _post_and_expect_422(c, _event(at="not-a-timestamp"))


# ---------------------------------------------------------------------------
# Retention window
# ---------------------------------------------------------------------------

def test_keeps_only_newest_50(client):
    c, events_file, _ = client
    total = MAX_RETRAIN_EVENTS + 5
    for i in range(total):
        c.post(
            "/admin/retrain-events",
            json=_event(run_id=f"run-{i:04d}"),
            headers=ADMIN,
        )
    events = _read_jsonl(events_file)
    assert len(events) == MAX_RETRAIN_EVENTS
    # oldest entries rotated away, newest retained, newest-first GET
    data = c.get("/admin/retrain-events", headers=ADMIN).json()
    assert data["latest"]["run_id"] == f"run-{total - 1:04d}"
    assert data["history"][0]["run_id"] == data["latest"]["run_id"]
    assert data["history"][-1]["run_id"] == f"run-{total - MAX_RETRAIN_EVENTS:04d}"


# ---------------------------------------------------------------------------
# Auth (untrusted peer needs the admin key; GET respects the read tier)
# ---------------------------------------------------------------------------

def test_post_requires_key_from_untrusted_peer(client):
    c, _, _ = client
    response = c.post("/admin/retrain-events", json=_event())
    assert response.status_code == 403


def test_get_without_key_rejected(client):
    c, _, _ = client
    response = c.get("/admin/retrain-events")
    assert response.status_code == 403
