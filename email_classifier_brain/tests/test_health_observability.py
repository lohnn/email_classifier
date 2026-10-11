
"""Fix E + F — daily-update health signals.

A failed training-data push, unpushed local commits, and a long
zero-correction streak while mail flows must land in update_history.json
(the file the user reads) and in job_runs — not just a swallowed log line.
A 'retrain worthwhile' signal fires once per data-repo state.
"""

import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

os.environ["TRAINING_DATA_DIR"] = tempfile.mkdtemp()
os.environ["ADMIN_API_KEY"] = "testkey"
os.environ["TESTING"] = "true"

import main  # noqa: F401
from main import app
import database
import jobs.update as upd


@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("database.DB_FILE", tmp.name):
            database.init_db()
            with patch.object(upd, "HISTORY_FILE", os.path.join(tmp.name + ".history")), \
                 patch.object(upd, "RETRAIN_MARKER_FILE", os.path.join(tmp.name + ".retrain")):
                from fastapi.testclient import TestClient
                with TestClient(app) as c:
                    yield c, tmp.name


def _history(client_tuple):
    _, _ = client_tuple
    path = upd.HISTORY_FILE
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _run_daily(push_ok=True, unpushed=0, retrain_worthwhile=False, data_commit="d1", model_commit="d1"):
    with patch.object(upd, "shutdown_server"), \
         patch("jobs.update.push_training_data_to_git", return_value=push_ok), \
         patch("jobs.update.count_unpushed_commits", return_value=unpushed), \
         patch.object(upd, "get_retrain_status",
                      return_value={"data_commit": data_commit,
                                    "model_trained_commit": model_commit,
                                    "retrain_worthwhile": retrain_worthwhile}):
        upd.scheduled_update_job(trigger="manual")


def test_push_failure_visible_in_history_and_job_runs(client):
    c, _ = client
    _run_daily(push_ok=False)

    entries = _history(client)
    assert any(e["status"] == "error" and "push failed" in e["message"] for e in entries)

    runs = database.get_job_runs(job_name="auto_update")
    assert runs[0]["status"] == "error"
    assert "push failed" in runs[0]["error_message"]


def test_unpushed_commits_warn(client):
    c, _ = client
    _run_daily(push_ok=True, unpushed=3)

    entries = _history(client)
    assert any(e["status"] == "warning" and "3 local training-data commit" in e["message"]
               for e in entries)
    runs = database.get_job_runs(job_name="auto_update")
    assert runs[0]["status"] == "success"


def test_zero_correction_streak_warns_only_when_volume_flows(client):
    c, _ = client
    database.add_log(
        id="g1", sender="s@x.com", recipient="r@x.com", subject="S",
        predicted_category="NOISE", confidence_score=0.9,  # timestamp -> now
        body="B", cc="", mass_mail=False, attachment_types=[],
    )

    _run_daily(push_ok=True)  # no recheck runs -> zero-correction streak
    entries = _history(client)
    assert any(e["status"] == "warning" and "No label corrections" in e["message"]
               for e in entries)

    # A recent correction silences the warning on the next daily run
    database.finish_job_run(
        database.start_job_run("recheck", "scheduled"), "success", emails_updated=2
    )
    with open(upd.HISTORY_FILE, "w") as f:
        f.write("")  # clear history for the assertion below
    _run_daily(push_ok=True)
    assert not any("No label corrections" in e["message"] for e in _history(client))


def test_retrain_signal_fires_once_per_data_state(client):
    c, _ = client
    _run_daily(push_ok=True, retrain_worthwhile=True, data_commit="abc123", model_commit="old999")

    entries = _history(client)
    signals = [e for e in entries if "Retrain worthwhile" in e["message"]]
    assert len(signals) == 1
    assert "abc123" in signals[0]["message"]
    assert "NOT run automatically" in signals[0]["message"]

    # Second run with the same data commit: no duplicate signal
    _run_daily(push_ok=True, retrain_worthwhile=True, data_commit="abc123", model_commit="old999")
    signals = [e for e in _history(client) if "Retrain worthwhile" in e["message"]]
    assert len(signals) == 1


def test_retrain_status_endpoint(client):
    c, _ = client
    with patch("api.routes.admin.get_retrain_status",
               return_value={"data_commit": "aaa", "model_trained_commit": "bbb",
                             "retrain_worthwhile": True}):
        response = c.get("/admin/retrain-status", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    assert response.json() == {
        "data_commit": "aaa",
        "model_trained_commit": "bbb",
        "retrain_worthwhile": True,
    }
