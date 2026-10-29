"""
tests/test_receive_centroids.py — POST /admin/receive-centroids
===============================================================

Pipeline step 6 (design §5, 2026-10-29): the classifier owns git — it
validates the orchestrator's centroid artifact, commits it on branch
vocab/<run_id> from a TEMP CLONE (live working tree untouched) and pushes
to the monorepo origin. Tests run against a local fixture monorepo + a
local BARE origin — no real remote can ever be touched.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import database  # noqa: E402
import jobs.update as upd  # noqa: E402
from api.routes import admin as admin_routes  # noqa: E402
from main import app  # noqa: E402

ADMIN = {"X-API-Key": "testkey"}

REAL_SHA = "a" * 64


def _valid_artifact(**over):
    d = {
        "version": 3,
        "model": {"gguf": "merged_text_q8_0.gguf", "sha256": REAL_SHA, "server": "llama.cpp"},
        "provenance": {"adapter": "lora.bin", "corpus_commit": "b" * 40,
                        "train_rows": 45, "split": "lab v3 protocol 45/10/9 seed 7", "prefix": "task: classification | query: ", "dim": 4},
        "centroids_l2": {"Jarvis/FOCUS": [0.1, 0.2, 0.3, 0.4], "Jarvis/NOISE": [0.4, 0.3, 0.2, 0.1]},
        "unsure_gate": {"confidence_threshold": 0.63, "delta_threshold": 0.08,
                          "calibration": {"k": 1, "row_count": 55}},
    }
    d.update(over)
    return d


class _SyncASGI:
    """Forged Tailscale peer (trusted) — test_model_upload.py convention."""

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

    def post(self, url, **kw):
        return self._run("POST", url, **kw)

    def get(self, url, **kw):
        return self._run("GET", url, **kw)


def _git(*args, cwd):
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Mini monorepo (with the tracked vocab path + a committed placeholder)
    + a local BARE origin; pointless without DB/history redirects."""
    mini = tmp_path / "monorepo"
    (mini / "email_classifier_brain" / "model").mkdir(parents=True)
    (mini / "email_classifier_brain" / "model" / "label_centroids.json").write_text('{"placeholder": true}')
    (mini / "README.md").write_text("mini")
    _git("init", "-q", cwd=str(mini))
    _git("add", ".", cwd=str(mini))
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "seed", cwd=str(mini))

    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True, capture_output=True)
    _git("remote", "add", "origin", str(bare), cwd=str(mini))

    hist = tmp_path / "history.json"
    hist.write_text("")
    db = tmp_path / "db.sqlite3"
    monkeypatch.setattr(database, "DB_FILE", str(db))
    monkeypatch.setattr(upd, "HISTORY_FILE", str(hist))
    monkeypatch.setattr(admin_routes, "_monorepo_root", lambda: mini)
    database.init_db()
    return {"mini": str(mini), "bare": str(bare), "hist": str(hist), "tmp": tmp_path}


def _post(env, payload, run_id="run-20261029-1", sha=None, **kw):
    body = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
    url = f"/admin/receive-centroids?run_id={run_id}"
    if sha:
        url += f"&sha256_expected={sha}"
    return _SyncASGI().post(url, headers=ADMIN, content=body)


def _bare_branches(env):
    out = subprocess.run(["git", "-C", env["bare"], "for-each-ref", "--format=%(refname:short)"],
                         capture_output=True, text=True, check=True).stdout.split()
    return sorted(out)


def test_untrusted_peer_with_valid_admin_key_is_403(env):
    from fastapi.testclient import TestClient
    with patch("database.DB_FILE", os.path.join(env["tmp"], "x.db")):
        database.init_db()
        with TestClient(app) as c:
            r = c.post("/admin/receive-centroids?run_id=run-1", headers=ADMIN, content=b"{}")
    assert r.status_code == 403


def test_valid_artifact_committed_and_pushed(env):
    r = _post(env, _valid_artifact())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "committed"
    assert data["branch"] == "vocab/run-20261029-1"
    assert data["pushed"] is True
    assert "vocab/run-20261029-1" in _bare_branches(env)
    # committed content == posted bytes, byte for byte
    blob = subprocess.run(
        ["git", "-C", env["bare"], "show",
         f"{data['branch']}:email_classifier_brain/model/label_centroids.json"],
        capture_output=True, check=True).stdout
    assert blob == json.dumps(_valid_artifact()).encode()
    assert "Vocabulary artifact" in open(env["hist"]).read()
    assert "retrain-orchestrator" in subprocess.run(
        ["git", "-C", env["bare"], "log", "-1", "--format=%an <%ae>", data["branch"]],
        capture_output=True, text=True, check=True).stdout


def test_same_content_repost_is_already_recorded(env):
    first = _post(env, _valid_artifact()).json()
    again = _post(env, _valid_artifact()).json()
    assert first["status"] == "committed"
    assert again["status"] == "already_recorded"
    # The bare origin only ever receives pushed vocab branches — main is
    # never published by this endpoint (activation is the P4 pair).
    assert _bare_branches(env) == ["vocab/run-20261029-1"]


def test_same_run_id_different_content_is_409_first_record_wins(env):
    _post(env, _valid_artifact())
    drifted = _valid_artifact()
    drifted["unsure_gate"]["confidence_threshold"] = 0.9
    r = _post(env, drifted)
    assert r.status_code == 409
    blob = subprocess.run(
        ["git", "-C", env["bare"], "show",
         "vocab/run-20261029-1:email_classifier_brain/model/label_centroids.json"],
        capture_output=True, check=True).stdout
    assert json.loads(blob)["unsure_gate"]["confidence_threshold"] == 0.63  # first record stands


def test_strict_validation_rejects_drift(env):
    broken = _valid_artifact(); broken["mystery_key"] = 1
    assert _post(env, broken).status_code == 422                      # unknown key
    broken = _valid_artifact(); del broken["unsure_gate"]
    assert _post(env, broken).status_code == 422                      # missing key
    assert _post(env, b"{nope").status_code == 422                    # not JSON
    nan = json.dumps(_valid_artifact(), default=lambda o: o).replace("0.1", "NaN", 1)
    assert _post(env, json.loads(nan)).status_code == 422             # NaN rejected
    broken = _valid_artifact()
    broken["centroids_l2"]["Jarvis/NOISE"] = [0.1, 0.2]               # dim mismatch
    assert _post(env, broken).status_code == 422
    broken = _valid_artifact(); broken["provenance"]["dim"] = 999
    assert _post(env, broken).status_code == 422                      # provenance vs actual dim
    assert _post(env, _valid_artifact(), run_id="bad id!").status_code == 422
    assert _post(env, _valid_artifact(),
                 sha="f" * 64).status_code == 400                     # sha mismatch
    good_sha = __import__("hashlib").sha256(json.dumps(_valid_artifact()).encode()).hexdigest()
    ok = _post(env, _valid_artifact(), sha=good_sha)
    assert ok.status_code == 200 and ok.json()["status"] == "committed"


def test_merge_checkpoint_semantics_documented_in_write(env):
    """The human log must spell the inert-until-P4 nature of the vocab commit."""
    _post(env, _valid_artifact())
    hist = open(env["hist"]).read()
    assert "Inert until the paired GGUF install" in hist
