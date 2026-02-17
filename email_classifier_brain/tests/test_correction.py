
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
import sys
import json
import shutil
import tempfile
from datetime import datetime

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

# Mock TRAINING_DATA_DIR before importing main
temp_data_dir = tempfile.mkdtemp()
os.environ["TRAINING_DATA_DIR"] = temp_data_dir
os.environ["ADMIN_API_KEY"] = "testkey"

# Mock classify to avoid loading model
mock_classify = MagicMock()
mock_classify.get_available_categories.return_value = ["FOCUS", "NOISE", "REFERENCE", "URGENT"]
sys.modules["classify"] = mock_classify
sys.modules["classifier_brain.classify"] = mock_classify

import main
from main import app
import database

@pytest.fixture
def client():
    # Ensure temp_data_dir exists for each test
    if not os.path.exists(temp_data_dir):
        os.makedirs(temp_data_dir)

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("database.DB_FILE", tmp.name):
            database.init_db()
            with TestClient(app) as c:
                yield c
    # Cleanup after all tests in this fixture
    if os.path.exists(temp_data_dir):
        shutil.rmtree(temp_data_dir)

def test_correction_endpoint(client):
    # 1. Add a log entry first
    database.add_log(
        sender="test@example.com",
        recipient="me@example.com",
        subject="Test subject",
        predicted_category="NOISE",
        confidence_score=0.5,
        body="Test body",
        cc="cc@example.com",
        mass_mail=False,
        attachment_types=["PDF"]
    )

    # Get the ID of the inserted log
    logs = database.get_unread_notifications()
    log_id = logs[0]["id"]

    # 2. Call correction endpoint (requires API key)
    response = client.post(
        f"/logs/{log_id}/correction",
        json={"corrected_category": "FOCUS"},
        headers={"X-API-Key": "testkey"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # 3. Verify DB update
    updated_log = database.get_log_by_id(log_id)
    assert updated_log["corrected_category"] == "FOCUS"

    # 4. Verify training data file (.jsonl)
    focus_jsonl_path = os.path.join(temp_data_dir, "FOCUS.jsonl")
    assert os.path.exists(focus_jsonl_path)

    with open(focus_jsonl_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = [json.loads(line) for line in lines]
        assert data[0]["subject"] == "Test subject"
        assert data[0]["body"] == "Test body"
        assert data[0]["from"] == "test@example.com"
        assert data[0]["attachment_types"] == ["PDF"]

def test_correction_nonexistent_log(client):
    response = client.post(
        "/logs/999/correction",
        json={"corrected_category": "FOCUS"},
        headers={"X-API-Key": "testkey"}
    )
    assert response.status_code == 404

def test_correction_unauthorized(client):
    response = client.post("/logs/1/correction", json={"corrected_category": "FOCUS"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Could not validate credentials"

@patch("subprocess.run")
def test_push_training_data_admin(mock_run, client):
    # Mock git status to show changes
    mock_run.return_value = MagicMock(stdout=" M FOCUS.json\n")

    # Need to make sure TRAINING_DATA_DIR exists
    if not os.path.exists(temp_data_dir):
        os.makedirs(temp_data_dir)

    response = client.post("/admin/push-training-data", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Verify git commands were called (add, status, commit, push)
    assert mock_run.call_count >= 2
