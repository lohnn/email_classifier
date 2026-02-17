
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
import sys
from datetime import datetime, timedelta

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

# Mock things that might be imported during main import
import tempfile
temp_data_dir = tempfile.mkdtemp()
os.environ["TRAINING_DATA_DIR"] = temp_data_dir
os.environ["ADMIN_API_KEY"] = "testkey"

# Now import main
import main
from main import app, job_lock
import database

@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("database.DB_FILE", tmp.name):
            database.init_db()
            with TestClient(app) as c:
                yield c

@pytest.fixture
def mock_imap_client():
    with patch("imap_client.GmailClient") as mock:
        yield mock

@pytest.fixture
def mock_classify():
    with patch("main.classify") as mock:
        yield mock

def test_get_stats_empty(client):
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json() == {"stats": {}}

def test_run_classification(client, mock_imap_client, mock_classify):
    if job_lock.locked():
        job_lock.release()

    mock_instance = mock_imap_client.return_value
    from email.message import Message
    mock_msg = Message()
    mock_msg["From"] = "sender@test.com"
    mock_msg["To"] = "recipient@test.com"
    mock_msg["Subject"] = "Test Subject"
    mock_msg["Date"] = "Wed, 02 Oct 2024 10:00:00 -0000"

    mock_instance.fetch_unprocessed_emails.return_value = [(b"123", mock_msg)]

    # Setup mocks
    mock_classify.extract_email_info.return_value = {
        "sender": "sender@test.com",
        "to": "recipient@test.com",
        "cc": "",
        "subject": "Test Subject",
        "body": "Test Body",
        "mass_mail": False,
        "attachment_types": []
    }
    mock_classify.predict_email.return_value = ("URGENT", 0.95)
    mock_classify.get_available_categories.return_value = ["URGENT", "FOCUS"]

    response = client.post("/run")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["processed_count"] == 1
    assert data["details"][0]["label"] == "URGENT"

    mock_instance.apply_label.assert_called_with(b"123", "URGENT")

    stats_response = client.get("/stats")
    assert stats_response.json()["stats"]["URGENT"] == 1

def test_run_classification_limit(client, mock_imap_client, mock_classify):
    if job_lock.locked():
        job_lock.release()

    mock_instance = mock_imap_client.return_value
    from email.message import Message
    mock_msg = Message()
    mock_instance.fetch_unprocessed_emails.return_value = [(str(i).encode(), mock_msg) for i in range(25)]

    mock_classify.extract_email_info.return_value = {
        "sender": "s@t.com", "to": "r@t.com", "cc": "", "subject": "S", "body": "B", "mass_mail": False, "attachment_types": []
    }
    mock_classify.predict_email.return_value = ("NOISE", 0.1)

    response = client.post("/run")
    assert response.status_code == 200
    data = response.json()
    assert data["processed_count"] == 20

def test_run_concurrently(client):
    if job_lock.locked():
        job_lock.release()
    job_lock.acquire()
    try:
        response = client.post("/run")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "skipped"
    finally:
        job_lock.release()

def test_pop_notifications(client, mock_imap_client, mock_classify):
    if job_lock.locked():
        job_lock.release()

    mock_instance = mock_imap_client.return_value
    from email.message import Message
    mock_msg = Message()
    mock_msg["From"] = "sender@test.com"
    mock_msg["Subject"] = "Test Pop"

    mock_instance.fetch_unprocessed_emails.return_value = [(b"123", mock_msg)]
    mock_classify.extract_email_info.return_value = {
        "sender": "sender@test.com", "to": "r@t.com", "cc": "", "subject": "Test Pop", "body": "B", "mass_mail": False, "attachment_types": []
    }
    mock_classify.predict_email.return_value = ("URGENT", 0.95)
    mock_classify.get_available_categories.return_value = ["URGENT"]

    client.post("/run")
    response = client.get("/notifications")
    assert len(response.json()) == 1

    pop_response = client.post("/notifications/pop")
    assert pop_response.status_code == 200
    assert len(pop_response.json()) == 1
    assert pop_response.json()[0]["subject"] == "Test Pop"

    response = client.get("/notifications")
    assert len(response.json()) == 0

def test_get_read_notifications(client, mock_imap_client, mock_classify):
    if job_lock.locked():
        job_lock.release()

    mock_instance = mock_imap_client.return_value
    from email.message import Message
    mock_msg = Message()
    mock_msg["Subject"] = "Test Read"

    now = datetime.utcnow()
    mock_instance.fetch_unprocessed_emails.return_value = [(b"123", mock_msg)]
    mock_classify.extract_email_info.return_value = {
        "sender": "s@t.com", "to": "r@t.com", "cc": "", "subject": "Test Read", "body": "B", "mass_mail": False, "attachment_types": []
    }
    mock_classify.predict_email.return_value = ("URGENT", 0.95)

    client.post("/run")
    client.post("/notifications/ack", json={})

    start_time = (now - timedelta(hours=1)).isoformat()
    end_time = (now + timedelta(hours=1)).isoformat()

    response = client.get(f"/notifications/read?start_time={start_time}&end_time={end_time}")
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_get_labels(client, mock_classify):
    expected_labels = ["FOCUS", "NOISE", "REFERENCE", "URGENT"]
    mock_classify.get_available_categories.return_value = expected_labels

    response = client.get("/labels")
    assert response.status_code == 200
    assert response.json() == expected_labels
