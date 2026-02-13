
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
import sys

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock 'classify' module entirely to avoid loading ML models and files
# We must do this BEFORE importing main
mock_classify_module = MagicMock()
sys.modules["classify"] = mock_classify_module
# Also mock classifier_brain.classify just in case
sys.modules["classifier_brain.classify"] = mock_classify_module

from main import app
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
    # Setup mock behavior
    mock_instance = mock_imap_client.return_value

    # Mock a message object
    from email.message import Message
    mock_msg = Message()
    mock_msg["From"] = "sender@test.com"
    mock_msg["Subject"] = "Test Subject"

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

    # Verify label was applied
    mock_instance.apply_label.assert_called_with(b"123", "URGENT")

    # verify stats updated
    stats_response = client.get("/stats")
    assert stats_response.json()["stats"]["URGENT"] == 1

def test_notifications(client, mock_imap_client, mock_classify_functions):
    # Setup mock behavior
    mock_instance = mock_imap_client.return_value

    from email.message import Message
    mock_msg = Message()
    mock_msg["From"] = "sender@test.com"
    mock_msg["Subject"] = "Test Notification"

    mock_instance.fetch_unprocessed_emails.return_value = [(b"123", mock_msg)]
    mock_classify_functions.predict_raw_email.return_value = ("URGENT", 0.95)
    mock_classify_functions.get_available_categories.return_value = ["URGENT"]

    # Run classification to generate a notification
    client.post("/run")

    # Check notifications
    response = client.get("/notifications")
    assert response.status_code == 200
    notifs = response.json()

    # Find our notification
    my_notif = next((n for n in notifs if n["subject"] == "Test Notification"), None)
    assert my_notif is not None
    assert my_notif["predicted_category"] == "URGENT"
    assert my_notif["is_read"] == 0

    # Ack notification
    ack_response = client.post("/notifications/ack", json={"ids": [my_notif["id"]]})
    assert ack_response.status_code == 200

    # Check notifications again (should be marked read)
    response = client.get("/notifications")
    # Verify it's no longer in the list (since endpoint returns unread)
    notifs = response.json()
    assert not any(n["id"] == my_notif["id"] for n in notifs)

def test_ack_all_notifications(client, mock_imap_client, mock_classify_functions):
    # Setup mock behavior
    mock_instance = mock_imap_client.return_value

    from email.message import Message
    mock_msg = Message()
    mock_msg["From"] = "sender@test.com"
    mock_msg["Subject"] = "Test All"

    mock_instance.fetch_unprocessed_emails.return_value = [(b"123", mock_msg), (b"124", mock_msg)]
    mock_classify_functions.predict_raw_email.return_value = ("URGENT", 0.95)
    mock_classify_functions.get_available_categories.return_value = ["URGENT"]

    # Run classification to generate notifications
    client.post("/run")

    # Check notifications
    response = client.get("/notifications")
    initial_count = len(response.json())
    assert initial_count >= 2

    # Ack ALL notifications (no IDs provided)
    ack_response = client.post("/notifications/ack", json={})
    assert ack_response.status_code == 200

    # Check notifications again (should be empty)
    response = client.get("/notifications")
    assert len(response.json()) == 0
