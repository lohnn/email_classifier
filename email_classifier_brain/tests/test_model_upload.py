
"""Model-artifact upload/install — TAILSCALE-PEER-ONLY model deploy path.

  - POST /admin/upload-model-artifact: trusted-peer only (valid admin key
    from an untrusted network is STILL 403 — these routes write
    code-adjacent payloads incl. a pickle that executes on load),
    streamed to a staging dir with sha256 while streaming, size cap,
    filename basename+extension whitelist, optional expected-sha check.
  - POST /admin/install-model-artifact: atomic move of a staged artifact to
    its active location; nothing activates without this explicit call.
  - Both land in update_history.json (file, size, sha, peer).
"""

import hashlib
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

os.environ.setdefault("TRAINING_DATA_DIR", tempfile.mkdtemp())
os.environ.setdefault("ADMIN_API_KEY", "testkey")
os.environ.setdefault("TESTING", "true")

import main  # noqa: F401
from main import app
import config
import database
import jobs.update as upd


ADMIN = {"X-API-Key": "testkey"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    staging = tmp_path / "model_staging"
    st_model = tmp_path / "model"
    llama_model = tmp_path / "llama_model"
    hist = tmp_path / "history.json"
    hist.write_text("")
    databases = tmp_path / "db.sqlite3"
    databases.write_bytes(b"x")
    monkeypatch.setattr(database, "DB_FILE", str(databases))
    monkeypatch.setattr(config, "MODEL_STAGING_DIR", str(staging))
    monkeypatch.setattr(config, "MODEL_OUTPUT_DIR", str(st_model))
    monkeypatch.setattr(config, "LLAMA_MODEL_FILE", str(llama_model / "merged.gguf"))
    monkeypatch.setattr(config, "MAX_MODEL_UPLOAD_BYTES", 1024 * 1024)
    monkeypatch.setattr(upd, "HISTORY_FILE", str(hist))
    database.init_db()
    return {"staging": str(staging), "st_model": str(st_model),
            "llama": str(llama_model / "merged.gguf"), "hist": str(hist),
            "db": str(databases), "tmp": tmp_path}


class _SyncASGI:
    """Sync wrapper over ASGITransport with a forged trusted peer — the
    async-only transport driven through asyncio.run per request/file."""

    def __init__(self, peer=("100.64.0.7", 55555)):
        self._peer = peer
        import httpx
        from httpx import ASGITransport
        self._transport = ASGITransport(app=app, client=self._peer)

    def _run(self, method, url, **kw):
        import asyncio
        import httpx

        async def once():
            async with httpx.AsyncClient(transport=self._transport,
                                         base_url="http://testserver") as client:
                return await getattr(client, method.lower())(url, **kw)
        return asyncio.run(once())

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, **kw):
        return self._run("POST", url, **kw)

    def get(self, url, **kw):
        return self._run("GET", url, **kw)


def _trusted_client():
    """Real ASGI stack with a forged Tailscale peer (trusted)."""
    return _SyncASGI()


def test_untrusted_peer_with_valid_admin_key_is_403(tmp_path):
    """The strictest rule: key alone is not enough — the peer must be trusted."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("database.DB_FILE", tmp.name):
            database.init_db()
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                response = c.post(
                    "/admin/upload-model-artifact?filename=merged.gguf",
                    content=b"model-bytes",
                    headers=ADMIN,                      # VALID admin key
                )
    assert response.status_code == 403
    assert "Tailscale" in response.json()["detail"] or "localhost" in response.json()["detail"]


def test_upload_stages_with_sha_and_trusted_peer(env):
    payload = b"GGUF-fake-bytes" * 10
    digest = _sha(payload)
    with _trusted_client() as c:
        response = c.post(
            "/admin/upload-model-artifact?filename=merged_text_q8_0.gguf",
            content=payload, headers=ADMIN,
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "staged"
    assert data["sha256"] == digest
    assert data["kind"] == "llama_model"
    staged = os.path.join(env["staging"], "merged_text_q8_0.gguf")
    assert os.path.exists(staged) and open(staged, "rb").read() == payload


def test_expected_sha_mismatch_aborts_staging(env):
    with _trusted_client() as c:
        response = c.post(
            "/admin/upload-model-artifact?filename=bad.gguf&sha256_expected=" + "0" * 64,
            content=b"real-bytes", headers=ADMIN,
        )
    assert response.status_code == 400
    staged_files = os.listdir(env["staging"]) if os.path.exists(env["staging"]) else []
    assert staged_files == []          # nothing left behind


def test_filename_traversal_and_bad_extension_rejected(env):
    with _trusted_client() as c:
        bad_path = c.post("/admin/upload-model-artifact?filename=../evil.gguf",
                          content=b"x", headers=ADMIN)
        bad_ext = c.post("/admin/upload-model-artifact?filename=payload.exe",
                         content=b"x", headers=ADMIN)
    assert bad_path.status_code == 400
    assert bad_ext.status_code == 400
    staged_files = os.listdir(env["staging"]) if os.path.exists(env["staging"]) else []
    assert staged_files == []


def test_size_cap_enforced_early(env, monkeypatch):
    # cap patched to 10 bytes; declaring 100 → 413 before any byte lands
    monkeypatch.setattr("config.MAX_MODEL_UPLOAD_BYTES", 10)
    with _trusted_client() as c:
        big = b"x" * 100
        response = c.post("/admin/upload-model-artifact?filename=big.gguf",
                          content=big,
                          headers={**ADMIN, "Content-Length": str(len(big))})
    assert response.status_code == 413
    staged_files = os.listdir(env["staging"]) if os.path.exists(env["staging"]) else []
    assert staged_files == []


def test_install_moves_to_active_target_and_activates_only_on_call(env):
    payload = b"GGUF-fake-bytes" * 10
    with _trusted_client() as c:
        c.post("/admin/upload-model-artifact?filename=merged.gguf",
               content=payload, headers=ADMIN)
        # nothing in the active target yet — no activation without install
        assert not os.path.exists(env["llama"])
        response = c.post("/admin/install-model-artifact?staged_name=merged.gguf",
                          headers=ADMIN)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "installed"
    assert data["sha256"] == _sha(payload)
    assert os.path.exists(env["llama"]) and open(env["llama"], "rb").read() == payload
    # staging consumed; history carries both steps
    assert os.listdir(env["staging"]) == []
    with open(env["hist"], encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]
    assert any("staged" in e["message"] for e in entries)
    assert any("INSTALLED" in e["message"] for e in entries)
    assert any("merged.gguf" in e["message"] for e in entries)


def test_install_unknown_staged_returns_404(env):
    with _trusted_client() as c:
        response = c.post("/admin/install-model-artifact?staged_name=never.gguf",
                          headers=ADMIN)
    assert response.status_code == 404


def test_st_model_component_installs_into_model_dir(env):
    payload = json.dumps({"label_mapping": {"0": "Jarvis/FOCUS"}}).encode()
    with _trusted_client() as c:
        c.post("/admin/upload-model-artifact?filename=label_mapping.json",
               content=payload, headers=ADMIN)
        response = c.post("/admin/install-model-artifact?staged_name=label_mapping.json",
                          headers=ADMIN)
    assert response.status_code == 200
    installed = os.path.join(env["st_model"], "label_mapping.json")
    assert open(installed, "rb").read() == payload
