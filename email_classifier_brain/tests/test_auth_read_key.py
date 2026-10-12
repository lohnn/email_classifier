
"""Read-only API key (READ_API_KEY) — PART B of the auth wave.

Semantics: READ_API_KEY grants GET/HEAD on every key-gated endpoint;
mutating routes (POST) accept only ADMIN_API_KEY or the trusted-network
bypass. Unset/empty READ_API_KEY = today's exact behavior. /health stays
public. Keys are never logged.
"""

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
import api.security as security


READ_KEY = "readonly-123"
ADMIN_KEY = "testkey"


@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("database.DB_FILE", tmp.name), \
             patch("jobs.training_data.TRAINING_DATA_DIR", temp_data_dir):
            database.init_db()
            with TestClient(app) as c:
                yield c


@pytest.fixture(autouse=True)
def read_key_env(monkeypatch):
    """READ_API_KEY set for most tests; each test can delenv to test compat."""
    monkeypatch.setenv("READ_API_KEY", READ_KEY)
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    yield


ENDPOINTS_PUBLIC = ["/health"]
ENDPOINTS_READ = [
    "/stats", "/labels", "/notifications",
    "/jobs/status", "/jobs/history",
    "/logs/ambiguous", "/logs/candidate-categories",
    "/admin/retrain-status", "/admin/update-errors",
]
ENDPOINTS_READ_WITH_PARAMS = [
    ("/notifications/read", {"start_time": "2026-01-01T00:00:00", "end_time": "2026-12-31T00:00:00"}),
]
ENDPOINTS_WRITE = [
    "/run", "/reclassify", "/jobs/cancel",
    "/notifications/ack", "/notifications/pop",
    "/logs/g1/correction",
    "/logs/candidate-categories/g1/accept", "/logs/candidate-categories/g1/discard",
    "/admin/check-corrections", "/admin/force-check-corrections",
    "/admin/backfill-training-data", "/admin/push-training-data",
]


def test_read_key_gets_all_read_endpoints(client):
    for endpoint in ENDPOINTS_READ:
        response = client.get(endpoint, headers={"X-API-Key": READ_KEY})
        assert response.status_code == 200, f"{endpoint}: {response.status_code}"
    for endpoint, params in ENDPOINTS_READ_WITH_PARAMS:
        response = client.get(endpoint, params=params, headers={"X-API-Key": READ_KEY})
        assert response.status_code == 200, f"{endpoint}: {response.status_code}"


def test_read_key_rejected_on_every_mutating_route(client):
    for endpoint in ENDPOINTS_WRITE:
        response = client.post(
            endpoint, json={}, headers={"X-API-Key": READ_KEY}
        )
        assert response.status_code == 403, f"{endpoint}: {response.status_code}"


def test_read_key_rejected_specifically_on_candidate_accept_discard(client):
    accept = client.post("/logs/candidate-categories/g1/accept",
                         json={"category": "PURCHASES"}, headers={"X-API-Key": READ_KEY})
    discard = client.post("/logs/candidate-categories/g1/discard",
                          headers={"X-API-Key": READ_KEY})
    assert accept.status_code == 403
    assert discard.status_code == 403


def test_health_stays_public_with_and_without_read_key(client):
    # 503 here = model intentionally not loaded under TESTING; never an auth code
    for headers in ({}, {"X-API-Key": READ_KEY}):
        assert client.get("/health", headers=headers).status_code == 503
    assert client.get("/health", headers={}).status_code != 403


def test_admin_key_write_still_works(client, mock_classify_admin):
    response = client.post("/run", headers={"X-API-Key": ADMIN_KEY})
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


def test_admin_key_read_still_works(client):
    for endpoint in ENDPOINTS_READ:
        response = client.get(endpoint, headers={"X-API-Key": ADMIN_KEY})
        assert response.status_code == 200, f"{endpoint}: {response.status_code}"


def test_unset_read_key_restores_exact_old_behavior(client, monkeypatch):
    monkeypatch.delenv("READ_API_KEY", raising=False)
    # A random key: 403 on a read — as before (only admin/bypass worked)
    assert client.get("/stats", headers={"X-API-Key": "anything"}).status_code == 403
    # Admin key still works everywhere
    assert client.get("/stats", headers={"X-API-Key": ADMIN_KEY}).status_code == 200


def test_empty_read_key_is_ignored(client, monkeypatch):
    monkeypatch.setenv("READ_API_KEY", "")
    assert client.get("/stats", headers={"X-API-Key": READ_KEY}).status_code == 403
    assert client.get("/stats", headers={"X-API-Key": ADMIN_KEY}).status_code == 200


def test_admin_unset_with_read_key_reads_ok_but_writes_still_blocked(client, monkeypatch):
    """Misconfigured server (no ADMIN_API_KEY) with a READ_API_KEY: reads work,
    writes keep today's 500-for-untrusted behavior."""
    monkeypatch.setenv("READ_API_KEY", READ_KEY)
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    assert client.get("/stats", headers={"X-API-Key": READ_KEY}).status_code == 200
    response = client.post("/run", headers={"X-API-Key": READ_KEY})
    assert response.status_code == 500     # old unset-ADMIN behavior for non-read access
    assert client.post("/run", json={}).status_code == 500


def test_trusted_network_bypass_unchanged_for_read_and_write(client):
    """#75 semantics: a socket peer from localhost/Tailscale gets full access
    with NO key at all — exercised through the real ASGI stack with a
    forged peer address."""
    import asyncio

    import httpx
    from httpx import ASGITransport

    async def _run():
        transport = ASGITransport(app=app, client=("100.64.0.7", 55555))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as asgi_client:
            read_resp = await asgi_client.get("/stats")               # no key at all
            write_resp = await asgi_client.post("/run", params={"limit": 1})  # bypassed write
            return read_resp, write_resp

    read_resp, write_resp = asyncio.run(_run())
    assert read_resp.status_code == 200
    assert write_resp.status_code == 200
    # 'accepted' or 'already_queued' both prove full (write) access — the
    # auth layer granted it; queue state depends on test order
    assert write_resp.json()["status"] in {"accepted", "already_queued"}


@pytest.fixture
def mock_classify_admin():
    mock = MagicMock()
    with patch("api.routes.classification.classify", mock):
        yield mock
