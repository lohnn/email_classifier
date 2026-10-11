
"""Fix C — unknown Gmail labels become observable candidate-new-category events.

The recheck job must record non-VERIFIED, non-UNSURE labels that the model
never trained on (candidate_labels on the log), the API must expose them,
and accept/discard endpoints must resolve them. Known labels and system
labels must never produce candidate events.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

temp_data_dir = tempfile.mkdtemp()
os.environ["TRAINING_DATA_DIR"] = temp_data_dir
os.environ["ADMIN_API_KEY"] = "testkey"
os.environ["TESTING"] = "true"

import main  # noqa: F401
from main import app
import database


@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("database.DB_FILE", tmp.name), \
             patch("jobs.training_data.TRAINING_DATA_DIR", temp_data_dir):
            database.init_db()
            with TestClient(app) as c:
                yield c


@pytest.fixture
def mock_classify():
    mock = MagicMock()
    with patch("api.routes.admin.classify", mock), \
         patch("jobs.correction.classify", mock):
        yield mock


@pytest.fixture
def mock_imap():
    mock_client = MagicMock()
    with patch("imap_client.gmail_client", mock_client), \
         patch("imap_client.GmailClient", return_value=mock_client):
        yield mock_client


def _seed_log(gid, predicted="NOISE"):
    database.add_log(
        id=gid,
        sender="s@x.com",
        recipient="me@x.com",
        subject=f"Seed {gid}",
        predicted_category=predicted,
        confidence_score=0.9,
        body=f"Body {gid}",
        cc="",
        mass_mail=False,
        attachment_types=[],
    )


def _labels_map(mapping):
    def fake(gmail_ids):
        return {g: mapping[g] for g in gmail_ids if g in mapping}
    return fake


def test_unknown_label_recorded_as_candidate(client, mock_classify, mock_imap):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    mock_imap.get_labels_for_emails.side_effect = _labels_map({"g1": ["PURCHASES"]})

    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")

    row = database.get_log_by_id("g1")
    assert json.loads(row["candidate_labels"]) == ["PURCHASES"]
    assert row["ambiguous_labels"] is None


def test_verifed_and_unsure_labels_never_candidates(client, mock_classify, mock_imap):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    mock_imap.get_labels_for_emails.side_effect = _labels_map(
        {"g1": ["VERIFIED", "UNSURE_CLASSIFICATION"]}
    )

    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")

    row = database.get_log_by_id("g1")
    assert row["candidate_labels"] is None


def test_known_label_flow_unchanged_no_candidate(client, mock_classify, mock_imap):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    mock_imap.get_labels_for_emails.side_effect = _labels_map({"g1": ["URGENT"]})

    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")

    row = database.get_log_by_id("g1")
    assert row["candidate_labels"] is None
    assert row["corrected_category"] == "URGENT"  # correction recorded as before
    assert os.path.exists(os.path.join(temp_data_dir, "URGENT.jsonl"))


def test_correction_move_removes_previous_training_copy(client, mock_classify, mock_imap):
    """End-to-end: a per-file NOISE copy must vanish when the recheck moves the email to URGENT."""
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    entry = {"subject": "Seed g1", "body": "Body g1", "from": "s@x.com"}
    with open(os.path.join(temp_data_dir, "NOISE.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    mock_imap.get_labels_for_emails.side_effect = _labels_map({"g1": ["URGENT"]})

    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")

    with open(os.path.join(temp_data_dir, "NOISE.jsonl"), encoding="utf-8") as f:
        assert not [l for l in f if l.strip()]  # old copy purged


def test_candidate_listing_and_accept_endpoint(client, mock_classify, mock_imap):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    database.update_recheck_status("g1", ambiguous_labels=None, candidate_labels=["PURCHASES"])

    response = client.get("/logs/candidate-categories", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1 and items[0]["id"] == "g1"

    response = client.post(
        "/logs/candidate-categories/g1/accept",
        json={"category": "PURCHASES"},
        headers={"X-API-Key": "testkey"},
    )
    assert response.status_code == 200
    assert response.json()["category"] == "PURCHASES"

    # Teaching file created, DB corrected, candidate cleared
    assert os.path.exists(os.path.join(temp_data_dir, "PURCHASES.jsonl"))
    assert database.get_log_by_id("g1")["corrected_category"] == "PURCHASES"
    assert database.get_log_by_id("g1")["candidate_labels"] is None
    assert mock_imap.apply_label.call_count >= 2  # category + VERIFIED

    # Listing no longer contains it
    response = client.get("/logs/candidate-categories", headers={"X-API-Key": "testkey"})
    assert response.json() == []


def test_accept_rejects_path_traversal_and_known_categories(client, mock_classify):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    database.update_recheck_status("g1", candidate_labels=["../evil"])

    response = client.post(
        "/logs/candidate-categories/g1/accept",
        json={"category": "../evil"},
        headers={"X-API-Key": "testkey"},
    )
    assert response.status_code == 400

    response = client.post(
        "/logs/candidate-categories/g1/accept",
        json={"category": "NOISE"},
        headers={"X-API-Key": "testkey"},
    )
    assert response.status_code == 400


def test_discard_endpoint_clears_flag(client):
    _seed_log("g1")
    database.update_recheck_status("g1", candidate_labels=["PURCHASES"])

    response = client.post(
        "/logs/candidate-categories/g1/discard", headers={"X-API-Key": "testkey"}
    )
    assert response.status_code == 200
    assert database.get_log_by_id("g1")["candidate_labels"] is None

    response = client.post(
        "/logs/candidate-categories/g1/discard", headers={"X-API-Key": "testkey"}
    )
    assert response.status_code == 404
