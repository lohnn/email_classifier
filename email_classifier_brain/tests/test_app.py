import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
import sys
from datetime import datetime, timedelta

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

from main import app, job_queue

@pytest.fixture(autouse=True)
def stop_queue_worker():
    """Stop the global JobQueue worker thread to prevent race conditions in tests.
    This ensures that when tests call `job_queue._drain()`, jobs run synchronously
    in the main test thread and are fully completed before subsequent assertions."""
    # Stop the worker
    job_queue._stop.set()
    job_queue._has_work.set()
    if job_queue._worker.is_alive():
        job_queue._worker.join(timeout=2)
    # Clear state
    with job_queue._lock:
        job_queue._queue.clear()
        job_queue._running = None

# Mock things that might be imported during main import
import tempfile
temp_data_dir = tempfile.mkdtemp()
os.environ["TRAINING_DATA_DIR"] = temp_data_dir
os.environ["ADMIN_API_KEY"] = "testkey"

os.environ["TESTING"] = "true"

# Now import main
import main
from main import app, job_queue
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
    response = client.get("/stats", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    assert response.json() == {"stats": {}}

def test_run_classification(client, mock_imap_client, mock_classify):

    mock_instance = mock_imap_client.return_value
    from email.message import Message
    mock_msg = Message()
    mock_msg["From"] = "sender@test.com"
    mock_msg["To"] = "recipient@test.com"
    mock_msg["Subject"] = "Test Subject"
    mock_msg["Date"] = "Wed, 02 Oct 2024 10:00:00 -0000"

    mock_instance.fetch_unprocessed_emails.return_value = [("123", mock_msg)]

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

    response = client.post("/run", headers={"X-API-Key": "testkey"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"

    # Drain the queue to execute the job synchronously for testing
    job_queue._drain()

    mock_instance.apply_label.assert_called_with("123", "URGENT")

    stats_response = client.get("/stats", headers={"X-API-Key": "testkey"})
    assert stats_response.json()["stats"]["URGENT"] == 1

def test_run_classification_limit(client, mock_imap_client, mock_classify):

    mock_instance = mock_imap_client.return_value
    from email.message import Message
    mock_msg = Message()
    # Return 25, but we expect it to be called with limit=20
    mock_instance.fetch_unprocessed_emails.return_value = [(str(i), mock_msg) for i in range(25)]

    mock_classify.extract_email_info.return_value = {
        "sender": "s@t.com", "to": "r@t.com", "cc": "", "subject": "S", "body": "B", "mass_mail": False, "attachment_types": []
    }
    mock_classify.predict_email.return_value = ("NOISE", 0.1)
    mock_classify.get_available_categories.return_value = ["NOISE"]

    response = client.post("/run", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    
    # Drain the queue to execute the job synchronously for testing
    job_queue._drain()

    # Verify it was called with default limit 20
    args, kwargs = mock_instance.fetch_unprocessed_emails.call_args
    assert kwargs.get('limit') == 20

def test_run_concurrently(client):
    from main import job_queue
    # Manually simulate a running classification job
    with job_queue._lock:
        job_queue._running = "classification"

    try:
        response = client.post("/run", headers={"X-API-Key": "testkey"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "already_queued"
    finally:
        with job_queue._lock:
            job_queue._running = None

def test_pop_notifications(client, mock_imap_client, mock_classify):

    mock_instance = mock_imap_client.return_value
    from email.message import Message
    mock_msg = Message()
    mock_msg["From"] = "sender@test.com"
    mock_msg["Subject"] = "Test Pop"

    mock_instance.fetch_unprocessed_emails.return_value = [("123", mock_msg)]
    mock_classify.extract_email_info.return_value = {
        "sender": "sender@test.com", "to": "r@t.com", "cc": "", "subject": "Test Pop", "body": "B", "mass_mail": False, "attachment_types": []
    }
    mock_classify.predict_email.return_value = ("URGENT", 0.95)
    mock_classify.get_available_categories.return_value = ["URGENT"]

    client.post("/run", headers={"X-API-Key": "testkey"})
    job_queue._drain()
    
    response = client.get("/notifications", headers={"X-API-Key": "testkey"})
    assert len(response.json()) == 1

    pop_response = client.post("/notifications/pop", headers={"X-API-Key": "testkey"})
    assert pop_response.status_code == 200
    assert len(pop_response.json()) == 1
    assert pop_response.json()[0]["subject"] == "Test Pop"

    response = client.get("/notifications", headers={"X-API-Key": "testkey"})
    assert len(response.json()) == 0

def test_get_read_notifications(client, mock_imap_client, mock_classify):

    mock_instance = mock_imap_client.return_value
    from email.message import Message
    mock_msg = Message()
    mock_msg["Subject"] = "Test Read"

    now = datetime.now()
    mock_instance.fetch_unprocessed_emails.return_value = [("123", mock_msg)]
    mock_classify.extract_email_info.return_value = {
        "sender": "s@t.com", "to": "r@t.com", "cc": "", "subject": "Test Read", "body": "B", "mass_mail": False, "attachment_types": []
    }
    mock_classify.predict_email.return_value = ("URGENT", 0.95)

    client.post("/run", headers={"X-API-Key": "testkey"})
    job_queue._drain()
    client.post("/notifications/ack", json={}, headers={"X-API-Key": "testkey"})

    start_time = (now - timedelta(hours=1)).isoformat()
    end_time = (now + timedelta(hours=1)).isoformat()

    response = client.get(f"/notifications/read?start_time={start_time}&end_time={end_time}", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_get_labels(client, mock_classify):
    expected_labels = ["FOCUS", "NOISE", "REFERENCE", "URGENT"]
    mock_classify.get_available_categories.return_value = expected_labels

    response = client.get("/labels", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    assert response.json() == expected_labels


# --- Health check tests ---

def test_health_model_not_loaded(client):
    """TESTING=true sets classify._model = None, so the health check reports model not_loaded → 503."""
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "error"
    assert data["checks"]["model"]["status"] == "not_loaded"
    assert data["checks"]["database"]["status"] == "ok"


def test_health_ok_when_model_loaded(client):
    """When model is loaded and DB is accessible the endpoint returns 200 ok."""
    with patch("classify._model", new=MagicMock()):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["checks"]["database"]["status"] == "ok"
    assert data["checks"]["model"]["status"] == "ok"
    assert "imap" not in data["checks"]


def test_health_db_error(client):
    """A DB failure marks the check as error and returns 503 with a generic message."""
    with patch("classify._model", new=MagicMock()):
        with patch("database.get_db_connection", side_effect=Exception("db unavailable")):
            response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "error"
    assert data["checks"]["database"]["status"] == "error"
    # Generic message — raw exception detail must not be exposed
    assert data["checks"]["database"]["detail"] == "Database connectivity error"


def test_health_imap_requires_auth(client):
    """check_imap=true without a valid API key returns 401."""
    with patch("classify._model", new=MagicMock()):
        response = client.get("/health?check_imap=true")
    assert response.status_code == 401


def test_health_imap_not_configured(client, mock_imap_client):
    """When check_imap=true but no IMAP credentials are set, reports not_configured."""
    mock_imap_client.return_value.connect.side_effect = ValueError("IMAP_USER (or MY_EMAIL) and IMAP_PASSWORD must be set in .env")
    with patch("classify._model", new=MagicMock()):
        response = client.get("/health?check_imap=true", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["checks"]["imap"]["status"] == "not_configured"


def test_health_imap_ok(client, mock_imap_client):
    """When check_imap=true and IMAP connects successfully, reports ok."""
    with patch("classify._model", new=MagicMock()):
        response = client.get("/health?check_imap=true", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["checks"]["imap"]["status"] == "ok"


def test_health_imap_error_is_degraded(client, mock_imap_client):
    """An IMAP failure results in degraded (HTTP 200) with a generic error message."""
    mock_imap_client.return_value.connect.side_effect = Exception("IMAP unreachable")
    with patch("classify._model", new=MagicMock()):
        response = client.get("/health?check_imap=true", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["checks"]["imap"]["status"] == "error"
    # Generic message — raw exception detail must not be exposed
    assert data["checks"]["imap"]["detail"] == "IMAP connectivity error"


def test_jobs_status_empty(client):
    """When no jobs are running or queued, status returns nulls/empty list."""
    response = client.get("/jobs/status", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    data = response.json()
    assert data["running"] is None
    assert data["queued"] == []


def test_jobs_status_requires_auth(client):
    """GET /jobs/status is protected by API key."""
    response = client.get("/jobs/status")
    assert response.status_code == 403


def test_jobs_status_with_queued_job(client):
    """After enqueuing a job (worker stopped), it appears in the queued list."""
    import datetime

    def noop():
        pass

    job_queue.enqueue("test_job", noop)
    try:
        response = client.get("/jobs/status", headers={"X-API-Key": "testkey"})
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is None
        assert len(data["queued"]) == 1
        entry = data["queued"][0]
        assert entry["name"] == "test_job"
        assert entry["enqueued_at"] is not None
        assert entry["started_at"] is None
        # enqueued_at should be a valid ISO 8601 timestamp
        datetime.datetime.fromisoformat(entry["enqueued_at"])
    finally:
        with job_queue._lock:
            job_queue._queue.clear()
