"""
tests/test_training_snapshot.py — GET /admin/training-snapshot
==============================================================

The trainer-side git replacement (design §5, 2026-10-29): one tar.gz with
the deployed brain code + the §16-validated splits + the training-data
repo contents, served TAILSCALE/LOCALHOST-PEER-ONLY. The forged-peer ASGI
client follows the test_model_upload.py convention.
"""

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config  # noqa: E402
import database  # noqa: E402
import jobs.update as upd  # noqa: E402
import main  # noqa: E402
from main import app  # noqa: E402

ADMIN = {"X-API-Key": "testkey"}


class _SyncASGI:
    """Sync wrapper over ASGITransport with a forged trusted peer — the
    async-only transport driven through asyncio.run per request."""

    def __init__(self, peer=("100.64.0.7", 55555)):
        from httpx import ASGITransport
        self._transport = ASGITransport(app=app, client=peer)

    def _run(self, method, url, **kw):
        import asyncio
        import httpx

        async def once():
            async with httpx.AsyncClient(transport=self._transport,
                                         base_url="http://testserver") as client:
                return await getattr(client, method.lower())(url, **kw)
        return asyncio.run(once())

    def get(self, url, **kw):
        return self._run("GET", url, **kw)


@pytest.fixture
def env(tmp_path, monkeypatch):
    data_dir = tmp_path / "training_data"
    data_dir.mkdir()
    hist = tmp_path / "history.json"
    hist.write_text("")
    db = tmp_path / "db.sqlite3"
    monkeypatch.setattr(database, "DB_FILE", str(db))
    monkeypatch.setattr(config, "TRAINING_DATA_DIR", str(data_dir))
    monkeypatch.setattr(upd, "HISTORY_FILE", str(hist))
    database.init_db()
    return {"data_dir": str(data_dir), "hist": str(hist), "tmp": tmp_path}


def _members(gz_bytes):
    with tarfile.open(fileobj=io.BytesIO(gz_bytes), mode="r:gz") as t:
        return set(t.getnames())


def _monorepo_head():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_untrusted_peer_with_valid_admin_key_is_403(env):
    """Key alone is not enough — peer must be in the Tailscale/localhost set."""
    from fastapi.testclient import TestClient
    with patch("database.DB_FILE", os.path.join(env["tmp"], "x.db")):
        database.init_db()
        with TestClient(app) as c:
            r = c.get("/admin/training-snapshot", headers=ADMIN)
    assert r.status_code == 403


def test_snapshot_serves_code_corpus_and_dir_fallback_data(env):
    """Trusted peer: full tar.gz — brain code, §16 splits, and the data dir
    under training_data/ (dir fallback when the tmp data dir is not git)."""
    os.makedirs(os.path.join(env["data_dir"], "Still/Needs/Moar"))
    with open(os.path.join(env["data_dir"], "Still/Needs/Moar/FINANCE.jsonl"), "w") as f:
        f.write('{"subject": "s", "body": "b", "from": "f", "to": "t", "cc": "", "mass_mail": false, "attachment_types": []}\n')

    r = _SyncASGI().get("/admin/training-snapshot", headers=ADMIN)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/gzip"
    assert r.headers["X-Snapshot-Data-Repo"] == "dir-fallback"
    assert r.headers["X-Snapshot-Data-Commit"] == ""

    code_sha = r.headers["X-Snapshot-Code-Commit"]
    assert len(code_sha) == 40 and all(ch in "0123456789abcdef" for ch in code_sha)
    assert code_sha == _monorepo_head()  # the snapshot certifies the deployed code

    members = _members(r.content)
    assert "email_classifier_brain/config.py" in members      # the training-code contract
    assert "email_classifier_brain/train.py" in members
    assert "retrain_2026-10-13/corpus/train.jsonl" in members  # the §16 gate inputs
    assert "retrain_2026-10-13/corpus/manifest.json" in members
    assert "retrain_2026-10-13/build_centroids.py" in members
    assert "training_data/Still/Needs/Moar/FINANCE.jsonl" in members
    # no runtime pollution: storage/ and .git never appear
    assert not any(m.startswith("email_classifier_brain/storage/") for m in members)
    assert not any("/.git/" in f"/{m}" for m in members)

    # every GET announces itself in the human log (corpus left the box)
    hist = open(env["hist"]).read()
    assert "Training snapshot served" in hist


def test_snapshot_from_real_git_data_repo(env):
    """When the data dir IS the git clone: payload-only archive (runtime
    clutter untracked = excluded by construction) + data commit header."""
    data = env["data_dir"]
    os.makedirs(os.path.join(data, "Jarvis"))
    with open(os.path.join(data, "Jarvis", "FOCUS.jsonl"), "w") as f:
        f.write('{"subject": "s", "body": "b", "from": "f", "to": "t", "cc": "", "mass_mail": false, "attachment_types": []}\n')
    def _git(*args):
        subprocess.run(["git", "-C", data, *args], check=True, capture_output=True)
    _git("init", "-q")
    _git("add", ".")
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "seed")
    # UNTRACKED runtime clutter, created AFTER the commit — the whole point
    # is that a git-archive payload excludes it by construction.
    with open(os.path.join(data, "runtime_clutter.db"), "wb") as f:
        f.write(b"untracked clutter")
    sha = subprocess.run(["git", "-C", data, "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()

    r = _SyncASGI().get("/admin/training-snapshot", headers=ADMIN)
    assert r.status_code == 200
    assert r.headers["X-Snapshot-Data-Repo"] == "git"
    assert r.headers["X-Snapshot-Data-Commit"] == sha

    members = _members(r.content)
    assert "training_data/Jarvis/FOCUS.jsonl" in members
    assert not any("runtime_clutter" in m for m in members)  # untracked ⇒ excluded


def test_missing_training_data_dir_is_500(env):
    """A missing corpus dir must fail LOUDLY, not hand back a partial snapshot."""
    import shutil
    shutil.rmtree(env["data_dir"])
    r = _SyncASGI().get("/admin/training-snapshot", headers=ADMIN)
    assert r.status_code == 500
    assert "Training data dir missing" in r.json()["detail"]
