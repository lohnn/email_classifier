
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
import sys
import json

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock 'classify' module entirely
mock_classify_module = MagicMock()
sys.modules["classify"] = mock_classify_module
sys.modules["classifier_brain.classify"] = mock_classify_module

from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("main.os.getenv")
@patch("main.Path")
@patch("main.shutdown_server")
def test_trigger_update_missing_config(mock_shutdown, mock_path, mock_getenv):
    # Setup mock Path
    mock_file = MagicMock()
    mock_path.return_value = mock_file

    # Mock getenv to return None for API key, simulating misconfiguration
    def getenv_side_effect(key, default=None):
        if key == "ADMIN_API_KEY":
            return None
        return default
    mock_getenv.side_effect = getenv_side_effect

    # Should fail with 500 because ADMIN_API_KEY is not set
    response = client.post("/admin/trigger-update", headers={"X-API-Key": "any"})

    assert response.status_code == 500
    assert "ADMIN_API_KEY not set" in response.json()["detail"]

@patch("main.os.getenv")
@patch("main.Path")
@patch("main.shutdown_server")
def test_trigger_update_with_auth(mock_shutdown, mock_path, mock_getenv):
    # Setup mock for env var
    def getenv_side_effect(key, default=None):
        if key == "ADMIN_API_KEY":
            return "secret-key"
        return default
    mock_getenv.side_effect = getenv_side_effect

    # Setup mock Path
    mock_file = MagicMock()
    mock_path.return_value = mock_file

    # Call without header - should fail
    response = client.post("/admin/trigger-update")
    assert response.status_code == 403

    # Call with wrong header - should fail
    response = client.post("/admin/trigger-update", headers={"X-API-Key": "wrong"})
    assert response.status_code == 403

    # Call with correct header - should succeed
    response = client.post("/admin/trigger-update", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 200
    assert response.json()["status"] == "update_initiated"

    # Verify file touch
    mock_path.assert_called_with(".update_request")
    mock_file.touch.assert_called_once()

    # Background tasks run after the response is sent in TestClient?
    # TestClient runs background tasks automatically.
    mock_shutdown.assert_called_once()

@patch("main.os.getenv")
@patch("main.Path")
def test_get_update_errors_no_file(mock_path, mock_getenv):
    # Setup mock for env var
    def getenv_side_effect(key, default=None):
        if key == "ADMIN_API_KEY":
            return "secret-key"
        return default
    mock_getenv.side_effect = getenv_side_effect

    mock_file = MagicMock()
    mock_path.return_value = mock_file
    mock_file.exists.return_value = False

    response = client.get("/admin/update-errors", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 200
    assert response.json() == []

@patch("main.os.getenv")
@patch("builtins.open", new_callable=MagicMock)
@patch("main.Path")
def test_get_update_errors_with_content(mock_path, mock_open, mock_getenv):
    # Setup mock for env var
    def getenv_side_effect(key, default=None):
        if key == "ADMIN_API_KEY":
            return "secret-key"
        return default
    mock_getenv.side_effect = getenv_side_effect

    mock_file = MagicMock()
    mock_path.return_value = mock_file
    mock_file.exists.return_value = True

    # Mock file content
    log_entry = {"timestamp": "2023-01-01", "status": "error", "message": "Test error"}
    # mock_open return value is the file object. Iterating over it yields lines.
    file_handle = mock_open.return_value.__enter__.return_value
    file_handle.__iter__.return_value = [json.dumps(log_entry) + "\n"]

    response = client.get("/admin/update-errors", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["message"] == "Test error"
