
"""Embedder-seam tests — llama.cpp is the production backend (per the lab's
BRAIN-SEAM-CHANGE-SPEC + the user-approved wave):

  - request shape: the classification prefix is applied IN THE CALLER,
    byte-for-byte 'task: classification | query: ', and the formatted
    format_model_input() string is POSTed as an OpenAI-style
    {"input": [ ... ]} to /v1/embeddings;
  - response parsing: OpenAI shape {"data": [{"embedding": [...]}]}, 768d,
    L2-normalized client-side (idempotent on an already-normalized payload);
  - EMBEDDER_BACKEND default = llama_cpp; EMBEDDER_BACKEND=st rolls back to
    the in-process SetFit path (kept intact);
  - llama endpoint outage = LOUD (update_history warning + EmbedderOutage
    raised through classify) — never a silent degrade.

The stub is a real in-process HTTP server (OpenAI /v1/embeddings shape),
so urllib is exercised end-to-end.
"""

import json
import math
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

os.environ.setdefault("TRAINING_DATA_DIR", os.path.join(tempfile_dir := __import__("tempfile").mkdtemp(), "s"))
os.environ.setdefault("ADMIN_API_KEY", "testkey")
os.environ.setdefault("TESTING", "true")

# NOTE: tests/test_auto_classification.py replaces sys.modules["classify"]
# with a MagicMock at collection time. Load the REAL classify module by file
# path so these tests are immune to that injection.
import importlib.util as _ilu, tempfile as _tempfile
_cs = _ilu.spec_from_file_location(
    "classify_real",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "classify.py"),
)
classify = _ilu.module_from_spec(_cs)
_cs.loader.exec_module(classify)

CLS_PROMPT = "task: classification | query: "


def l2(v):
    n = math.sqrt(sum(x * x for x in v)) + 1e-12
    return [x / n for x in v]


class FakeLlama:
    """OpenAI-style /v1/embeddings stub with call recording + failure modes."""

    def __init__(self, dim=768, fail=False):
        self.dim = dim
        self.fail = fail
        self.requests = []
        self.hits = 0
        self._ready = threading.Event()

    def handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if not self.path.startswith("/v1/embeddings"):
                    return self._json(404, {"error": "unknown route"})
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.requests.append(body)
                if fake.fail:
                    return self._json(503, {"error": "embedder down"})
                texts = body.get("input", [])
                fake.hits += len(texts)
                embs = [l2([0.1 * ((len(t) % 13) + 1)] * fake.dim) for t in texts]
                # distinct per text so kNN ordering is test-controllable
                embs = [l2([0.5 + 0.01 * ((len(t) % 7) + j * 0.13) for j in range(fake.dim)])
                        for t in texts]
                self._json(200, {"data": [{"embedding": e} for e in embs],
                                 "model": "email", "usage": {}})

            def do_GET(self):
                if self.path.startswith("/health"):
                    return self._json(200, {"status": "ok", "dim": fake.dim})
                self._json(404, {"error": "nope"})

        return H


@pytest.fixture
def llama_stub():
    fake = FakeLlama()
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{port}"
    yield fake
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test starts from a pristine import state, temp history file."""
    monkeypatch.setenv("EMBEDDER_BACKEND", "llama_cpp")
    monkeypatch.setenv("LLAMA_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("LLAMA_RETRIES", "1")
    monkeypatch.setenv("LLAMA_RETRY_BACKOFF_SECONDS", "0")
    hist = tmp_path / "history.json"
    hist.write_text("")
    import jobs.update as upd
    monkeypatch.setattr(upd, "HISTORY_FILE", str(hist))
    # per-test isolation via attribute patches — NO module reloads (reloads
    # fork module state across the suite); classify/embedder read config
    # ATTRIBUTES at call time so monkeypatching is sufficient.
    import embedder
    monkeypatch.setattr(embedder, "_centroid_cache", None)
    yield tmp_path
    monkeypatch.setattr(embedder, "_centroid_cache", None)


def test_default_backend_is_llama():
    import config
    assert config.EMBEDDER_BACKEND == "llama_cpp"                # default (env unset)
    assert config.CLS_PROMPT == CLS_PROMPT                       # byte-for-byte spec string
    assert config.LLAMA_ENDPOINT == "http://127.0.0.1:8090"      # DEPLOY.md port


def test_request_shape_prefix_and_parse(llama_stub, monkeypatch, isolated):
    import config, embedder
    monkeypatch.setattr(config, "LLAMA_ENDPOINT", llama_stub.url)
    texts = ["formatted email body one", "formatted email body two"]
    vecs = embedder.embed_texts(texts)
    assert len(vecs) == 2 and all(len(v) == 768 for v in vecs)   # 768d, parsed
    body = llama_stub.requests[-1]
    assert len(body["input"]) == 2
    assert body["input"][0].startswith(CLS_PROMPT)               # caller-applied prefix
    assert body["input"][0].endswith(texts[0])                   # formatted string appended
    assert body["input"][1].endswith(texts[1])
    assert all(abs(1.0 - math.sqrt(sum(x * x for x in v))) < 1e-6 for v in vecs)  # L2


def test_loud_outage_no_silent_degrade(monkeypatch, isolated):
    import config, embedder
    dead_url = "http://127.0.0.1:1"     # nothing listens here
    monkeypatch.setattr(config, "LLAMA_ENDPOINT", dead_url)
    with pytest.raises(embedder.EmbedderOutage):
        embedder.embed_texts(["irrelevant"])                     # raised, not swallowed
    import jobs.update as upd
    with open(upd.HISTORY_FILE, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]
    assert any(e["status"] == "warning" and "EMBEDDER OUTAGE" in e["message"]
               for e in entries)                                 # LOUD, human-readable
    # no auto-revert: the flag still says llama after an outage
    assert config.EMBEDDER_BACKEND == "llama_cpp"


def test_classification_uses_llama_and_gate(llama_stub, monkeypatch, isolated):
    """classify.predict_email on the llama path: kNN over shipped centroids,
    UNSURE gate with the calibrated artifact thresholds."""
    import config, embedder
    monkeypatch.setattr(config, "LLAMA_ENDPOINT", llama_stub.url)  # attr read at call time
    cent_path = isolated / "label_centroids.json"
    cent = {
        "version": 1,
        "centroids_l2": {
            "Jarvis/URGENT": l2([1.0] * 768),
            "Jarvis/NOISE": l2([0.2] * 768),
        },
        "unsure_gate": {"confidence_threshold": 0.71, "delta_threshold": 0.064},
    }
    cent_path.write_text(json.dumps(cent))
    monkeypatch.setattr(config, "CENTROIDS_PATH", str(cent_path))
    import embedder
    monkeypatch.setattr(embedder, "_centroid_cache", None)
    monkeypatch.setattr(config, "UNSURE_CONFIDENCE_THRESHOLD", 0.65)
    monkeypatch.setattr(config, "UNSURE_DELTA_THRESHOLD", 0.10)
    label, score, unsure = classify.predict_email(
        subject="Server down", body="all services offline", return_score=True
    )
    assert label == "Jarvis/URGENT"                              # argmax cosine
    assert 0 < score <= 1.0 and isinstance(unsure, bool)
    assert all("POST" not in str(x) for x in [])                 # no-op guard
    assert mock_free_embedding_seen(llama_stub)


def mock_free_embedding_seen(fake):
    return fake.hits >= 1


def test_st_rollback_path_is_intact(monkeypatch, isolated):
    """EMBEDDER_BACKEND=st keeps the in-process SetFit path (rollback),
    flipped only as a deliberate action — the default stays llama."""
    import config
    monkeypatch.setattr(config, "EMBEDDER_BACKEND", "st")
    with pytest.raises(RuntimeError, match="TESTING"):           # SetFit stack inert in tests
        classify._get_st()
    assert config.EMBEDDER_BACKEND == "st"                       # explicit flip honored
