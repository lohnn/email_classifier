"""Read-sweep debug round — category-source robustness (sweep root cause).

The read sweep skips with "no trained categories" when
classify.get_available_categories() resolves empty. These tests pin the
behaviour that makes an empty resolution diagnosable instead of silent:
  - a valid, non-empty centroids artifact → parsed label names;
  - an unreadable/missing artifact → explicit fallback + a LOUD warning
    naming paths and existence (the Pi's cycle-skip camouflage);
  - an empty-but-valid artifact / empty fallback → loud once-per-process
    vocabulary warning, never fabricated labels.
"""

import importlib.util
import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

os.environ.setdefault("TRAINING_DATA_DIR", tempfile.mkdtemp(prefix="catsrc-tests-"))
os.environ.setdefault("ADMIN_API_KEY", "testkey")
os.environ.setdefault("TESTING", "true")

import config as _config

# Load the REAL classify module under a private name. test_app.py swaps a
# MagicMock into sys.modules["classify"] (for main's deferred imports), so a
# plain `import classify` anywhere after its collection returns the mock —
# which is what made this file's assertions fail in full-suite runs while
# passing standalone. A spec-loaded private instance is immune.
_CLASSIFY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "classify.py"
)
_spec = importlib.util.spec_from_file_location(
    "classify_for_sweep_debug_tests", _CLASSIFY_PATH
)
classify = importlib.util.module_from_spec(_spec)
sys.modules["classify_for_sweep_debug_tests"] = classify
_spec.loader.exec_module(classify)


@pytest.fixture(autouse=True)
def reset_centroid_cache():
    """load_centroids caches process-wide; tests point it at their own files."""
    classify._reset_centroid_cache()
    yield
    classify._reset_centroid_cache()


@pytest.fixture
def llama_backend():
    with patch.object(_config, "EMBEDDER_BACKEND", "llama_cpp"):
        yield


def _write_centroids(tmp_path, payload):
    path = tmp_path / "label_centroids.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_nonempty_centroids_parse_to_label_names(tmp_path, llama_backend):
    path = _write_centroids(tmp_path, {
        "centroids_l2": {"WORK/URGENT": [1.0, 0.0], "NOISE": [0.0, 1.0]},
    })
    with patch.object(_config, "CENTROIDS_PATH", path):
        assert classify.get_available_categories() == ["NOISE", "WORK/URGENT"]


def test_missing_centroids_falls_back_and_warns_loud(tmp_path, llama_backend, caplog):
    path = _write_centroids(tmp_path, {"centroids_l2": {"NOISE": [1.0]}})
    missing = os.path.join(tmp_path, "gone", "label_centroids.json")
    with patch.object(_config, "CENTROIDS_PATH", missing), \
         patch.object(classify, "_label_mapping", {0: "NOISE"}), \
         caplog.at_level("WARNING"):
        categories = classify.get_available_categories()

    assert categories == ["NOISE"]  # the label_mapping fallback — when populated
    assert any("centroids artifact unreadable" in r.message for r in caplog.records)
    assert any(missing in r.message for r in caplog.records)


def test_empty_vocabulary_is_loud_never_fabricated(tmp_path, llama_backend, caplog):
    """Empty-but-valid artifact (or empty fallback) must produce a loud
    once-per-process vocabulary warning and [] — never invented labels."""
    path = _write_centroids(tmp_path, {"centroids_l2": {}})
    with patch.object(_config, "CENTROIDS_PATH", path), \
         patch.object(classify, "_label_mapping", {}), \
         patch.object(classify, "_vocabulary_warned", False), \
         caplog.at_level("ERROR"):
        first = classify.get_available_categories()
        second = classify.get_available_categories()

    assert first == []
    assert second == []
    warnings = [r for r in caplog.records if "EMPTY vocabulary" in r.message]
    assert len(warnings) == 1  # rate-limited to once per process
    assert any(str(path) in r.message for r in warnings)


def test_st_backend_uses_label_mapping_only(tmp_path):
    """backend=st: the vocabulary is the lazy SetFit-era mapping; empty until
    the first in-process prediction (_get_st) populates it — the sweep-time
    re-read (see test_read_sweep) closes exactly that race."""
    with patch.object(_config, "EMBEDDER_BACKEND", "st"), \
         patch.object(classify, "_label_mapping", {0: "FOCUS", 1: "URN"}):
        assert classify.get_available_categories() == ["FOCUS", "URN"]


def test_llama_backend_centroids_missing_and_mapping_empty_is_loud(
    tmp_path, llama_backend, caplog
):
    """The exact production trap: unreadable artifact + never-populated
    label_mapping → [] every call, but now LOUDLY diagnosable."""
    path = _write_centroids(tmp_path, {"centroids_l2": {"NOISE": [1.0]}})
    with patch.object(_config, "CENTROIDS_PATH",
                      os.path.join(tmp_path, "not_there.json")), \
         patch.object(classify, "_label_mapping", {}), \
         patch.object(classify, "_vocabulary_warned", False), \
         caplog.at_level("ERROR"):
        assert classify.get_available_categories() == []
    assert any("EMPTY vocabulary" in r.message for r in caplog.records)
