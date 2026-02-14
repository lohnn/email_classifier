
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
import sys
from datetime import datetime, timedelta

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock 'classify' module entirely to avoid loading ML models and files
# We must do this BEFORE importing main
mock_classify_module = MagicMock()
sys.modules["classify"] = mock_classify_module
# Also mock classifier_brain.classify just in case
sys.modules["classifier_brain.classify"] = mock_classify_module

from main import app, job_lock
from database import init_db

@pytest.fixture
def client():
    # Use a temporary file for the database
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        # Patch the DB_FILE in the database module
        with patch("database.DB_FILE", tmp.name):
            # Re-initialize DB on the temp file
            init_db()
            with TestClient(app) as c:
                yield c

@pytest.fixture
def mock_imap_client():
    with patch("imap_client.GmailClient") as mock:
        yield mock

@pytest.fixture
def mock_classify_functions():
    # Configure the mocked module functions
    # Reset them for each test
    mock_classify_module.predict_raw_email.reset_mock()
    mock_classify_module.get_available_categories.reset_mock()
    return mock_classify_module

def test_get_stats_empty(client):
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json() == {"stats": {}}

def test_run_classification(client, mock_imap_client, mock_classify_functions):
    # Ensure lock is free
    if job_lock.locked():
        job_lock.release()

    # Setup mock behavior
    mock_instance = mock_imap_client.return_value

    # Mock a message object
    from email.message import Message
    mock_msg = Message()
    mock_msg["From"] = "sender@test.com"
    mock_msg["To"] = "recipient@test.com"
    mock_msg["Subject"] = "Test Subject"
    # Valid date string
    mock_msg["Date"] = "Wed, 02 Oct 2024 10:00:00 -0000"

    # Mock fetch_unprocessed_emails to return one email
    mock_instance.fetch_unprocessed_emails.return_value = [(b"123", mock_msg)]

    # Mock predict_raw_email on our module mock
    mock_classify_functions.predict_raw_email.return_value = ("URGENT", 0.95)
    mock_classify_functions.get_available_categories.return_value = ["URGENT", "FOCUS"]

    # Call the run endpoint
    response = client.post("/run")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["processed_count"] == 1
    assert data["details"][0]["label"] == "URGENT"
    assert data["details"][0]["score"] == 0.95
    assert data["details"][0]["recipient"] == "recipient@test.com"

    # Verify label was applied
    mock_instance.apply_label.assert_called_with(b"123", "URGENT")

    # verify stats updated
    stats_response = client.get("/stats")
    assert stats_response.json()["stats"]["URGENT"] == 1

def test_run_classification_limit(client, mock_imap_client, mock_classify_functions):
    if job_lock.locked():
        job_lock.release()

    mock_instance = mock_imap_client.return_value
    from email.message import Message
    mock_msg = Message()
    # Create 25 mock messages
    mock_instance.fetch_unprocessed_emails.return_value = [(str(i).encode(), mock_msg) for i in range(25)]
    mock_classify_functions.predict_raw_email.return_value = ("NOISE", 0.1)

    # Call with default limit (should be 20)
    response = client.post("/run")
    assert response.status_code == 200
    data = response.json()
    assert data["processed_count"] == 20

    # Call with explicit limit=5
    response = client.post("/run?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["processed_count"] == 5

def test_run_concurrently(client):
    """Test that if the job is running (locked), subsequent calls are skipped."""
    if job_lock.locked():
        job_lock.release()

    # Manually acquire the lock to simulate a running job
    job_lock.acquire()

    try:
        response = client.post("/run")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "skipped"
        assert data["processed_count"] == 0
    finally:
        job_lock.release()

def test_pop_notifications(client, mock_imap_client, mock_classify_functions):
    if job_lock.locked():
        job_lock.release()

    # Setup mock behavior
    mock_instance = mock_imap_client.return_value

    from email.message import Message
    mock_msg = Message()
    mock_msg["From"] = "sender@test.com"
    mock_msg["Subject"] = "Test Pop"

    mock_instance.fetch_unprocessed_emails.return_value = [(b"123", mock_msg)]
    mock_classify_functions.predict_raw_email.return_value = ("URGENT", 0.95)
    mock_classify_functions.get_available_categories.return_value = ["URGENT"]

    # Run classification
    client.post("/run")

    # Check unread notifications first
    response = client.get("/notifications")
    assert len(response.json()) == 1

    # Pop notifications
    pop_response = client.post("/notifications/pop")
    assert pop_response.status_code == 200
    popped = pop_response.json()
    assert len(popped) == 1
    assert popped[0]["subject"] == "Test Pop"

    # Check unread notifications again (should be empty)
    response = client.get("/notifications")
    assert len(response.json()) == 0

def test_get_read_notifications(client, mock_imap_client, mock_classify_functions):
    if job_lock.locked():
        job_lock.release()

    # Setup mock behavior
    mock_instance = mock_imap_client.return_value
    from email.message import Message
    mock_msg = Message()
    mock_msg["Subject"] = "Test Read"

    # Use a date we can query
    now = datetime.utcnow()
    # Note: The mock date string in email vs system time might differ but database uses parsed or system time.
    # In main.py we try to parse the Date header. If missing/invalid, we use system time.
    # Let's rely on system time for simplicity in this test or ensure parsing works.

    mock_instance.fetch_unprocessed_emails.return_value = [(b"123", mock_msg)]
    mock_classify_functions.predict_raw_email.return_value = ("URGENT", 0.95)

    # Run classification
    client.post("/run")

    # Ack notifications to mark as read
    client.post("/notifications/ack", json={})

    # Query read notifications
    # Ensure our query range covers the insertion time
    start_time = (now - timedelta(hours=1)).isoformat()
    end_time = (now + timedelta(hours=1)).isoformat()

    response = client.get(f"/notifications/read?start_time={start_time}&end_time={end_time}")
    assert response.status_code == 200
    read_notifs = response.json()
    assert len(read_notifs) == 1
    assert read_notifs[0]["subject"] == "Test Read"
    assert read_notifs[0]["is_read"] == 1
