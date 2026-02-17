
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
import sys

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock 'classify' module entirely to avoid loading ML models and files
mock_classify_module = MagicMock()
sys.modules["classify"] = mock_classify_module
sys.modules["classifier_brain.classify"] = mock_classify_module

# We do NOT mock database in sys.modules anymore to avoid breaking other tests
from database import init_db

@pytest.fixture
def clean_scheduler():
    from main import scheduler
    scheduler.remove_all_jobs()
    yield scheduler
    scheduler.remove_all_jobs()

@pytest.fixture
def mock_db():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("database.DB_FILE", tmp.name):
            init_db()
            yield tmp.name

def test_auto_classification_enabled(clean_scheduler, mock_db):
    with patch("config.ENABLE_AUTO_CLASSIFICATION", True):
        from main import app
        with TestClient(app) as client:
            job_ids = [job.id for job in clean_scheduler.get_jobs()]
            assert "classification_job" in job_ids
            assert "auto_update_job" in job_ids

def test_auto_classification_disabled(clean_scheduler, mock_db):
    with patch("config.ENABLE_AUTO_CLASSIFICATION", False):
        from main import app
        with TestClient(app) as client:
            job_ids = [job.id for job in clean_scheduler.get_jobs()]
            assert "classification_job" not in job_ids
            assert "auto_update_job" in job_ids
