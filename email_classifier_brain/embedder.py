"""
embedder.py — llama.cpp embedding client + kNN centroid classifier
==================================================================

Production serving path (EMBEDDER_BACKEND=llama_cpp): POST the
classification-prefixed format_model_input() string to the local llama.cpp
server (/v1/embeddings, OpenAI-style response), L2-normalize the 768d
vector client-side, and classify by cosine against the shipped per-class
centroids (kNN). The UNSURE gate (is_unsure_classification in classify.py)
stays the single gate code — only its thresholds come from the centroids
artifact's calibration (cosine scale differs from SetFit probabilities).

Failure semantics (2026-10 wave rules): the llama.cpp endpoint is retried
(LLAMA_RETRIES × LLAMA_RETRY_BACKOFF_SECONDS), after which the failure is
surfaced LOUDLY — logger.error + one update_history.json warning per
cooldown — and EmbedderOutage is raised. There is NO silent fallback to
the ST path from inside this module; flipping EMBEDDER_BACKEND=st is a
deliberate ops action (the ST path is preserved in classify.py for it).
"""

import json
import logging
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_WARN_COOLDOWN_SECONDS = 30 * 60
_last_outage_warned_at = 0.0


class EmbedderOutage(RuntimeError):
    """The configured llama.cpp endpoint is unreachable or persistently failing."""


def l2_normalize(vec: list) -> list:
    norm = math.sqrt(sum(x * x for x in vec)) + 1e-12
    return [x / norm for x in vec]


def _post_embeddings(texts: list) -> list:
    """POST /v1/embeddings (OpenAI-style), returns list of L2-normalized 768d vectors."""
    body = json.dumps({"model": "email", "input": [config.CLS_PROMPT + t for t in texts]}).encode()
    req = urllib.request.Request(
        config.LLAMA_ENDPOINT.rstrip("/") + "/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(1, config.LLAMA_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=config.LLAMA_TIMEOUT_SECONDS) as resp:
                payload = json.load(resp)
            vectors = [item["embedding"] for item in payload["data"]]
            if not vectors or not isinstance(vectors[0], list):
                raise ValueError(f"unexpected /v1/embeddings payload shape: {payload}")
            return [l2_normalize(v) for v in vectors]
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, KeyError) as e:
            last_error = e
            logger.warning(
                "llama.cpp embed attempt %d/%d failed: %s: %s",
                attempt, config.LLAMA_RETRIES, type(e).__name__, e,
            )
            if attempt < config.LLAMA_RETRIES:
                time.sleep(config.LLAMA_RETRY_BACKOFF_SECONDS)
    raise EmbedderOutage(
        f"llama.cpp endpoint {config.LLAMA_ENDPOINT} unreachable/invalid after "
        f"{config.LLAMA_RETRIES} attempts: {type(last_error).__name__}: {last_error}"
    )


def embed_texts(texts: list) -> list:
    """Embed formatted inputs (single-flight callers only — the Pi
    serializes inference). Does NOT prepend the prompt: pass raw
    format_model_input() strings here; the classification prefix is applied
    inside _post_embeddings per spec."""
    if not texts:
        return []
    try:
        return _post_embeddings(texts)
    except EmbedderOutage:
        _surface_outage()
        raise


def _surface_outage() -> None:
    """LOUD outage signal (log + update_history warning, once per cooldown).
    The ST flip stays a deliberate ops action — never automatic."""
    global _last_outage_warned_at
    logger.error(
        "EMBEDDER OUTAGE: llama.cpp endpoint %s is down — classification fails "
        "until the server is restored (email-embedder.service) or ops flips "
        "EMBEDDER_BACKEND=st in .env (the rollback). NOT auto-reverting.",
        config.LLAMA_ENDPOINT,
    )
    now = time.time()
    if now - _last_outage_warned_at < _WARN_COOLDOWN_SECONDS:
        return
    _last_outage_warned_at = now
    try:
        from jobs.update import append_update_history
        append_update_history(
            "warning",
            f"EMBEDDER OUTAGE: llama.cpp endpoint {config.LLAMA_ENDPOINT} unreachable — "
            f"classification is failing until the embedder service recovers. Ops action: "
            f"check email-embedder.service, or deliberately flip EMBEDDER_BACKEND=st in "
            f".env to roll back (no auto-revert by design).",
        )
    except Exception as e:  # history write must never mask the outage
        logger.error("Could not write embedder-outage warning to update_history: %s", e)


# ---------------------------------------------------------------------------
# Centroid kNN
# ---------------------------------------------------------------------------

_centroid_cache = None  # {label: L2 vector}, None = not loaded


def load_centroids(path: str | None = None) -> dict:
    """Load the shipped per-class centroids (with provenance + gate
    calibration). Cached after first load.

    An empty-but-valid artifact parses to {} (WI-103 debug round: the
    previous `next(iter(...))` crashed with StopIteration on exactly that
    input — an empty vocabulary must resolve to [] + the loud
    get_available_categories() warning, not to a mysterious crash)."""
    global _centroid_cache
    if _centroid_cache is not None and path is None:
        return _centroid_cache
    path = path or config.CENTROIDS_PATH
    with open(path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    centroids = artifact["centroids_l2"]
    if not centroids:
        _centroid_cache = {}
        return {}
    dim = len(next(iter(centroids.values())))
    for label, vec in centroids.items():
        if len(vec) != dim:
            raise ValueError(f"centroid {label!r} has dim {len(vec)}, expected {dim}")
    _centroid_cache = centroids
    return centroids


def artifact_gate_thresholds(path: str | None = None) -> dict:
    """UNSURE gate thresholds calibrated in the centroids artifact
    (cosine scale). Falls back to the SetFit-era constants if absent."""
    path = path or config.CENTROIDS_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            gate = json.load(f).get("unsure_gate", {})
        return {
            "confidence_threshold": float(
                gate.get("confidence_threshold", config.UNSURE_CONFIDENCE_THRESHOLD)
            ),
            "delta_threshold": float(
                gate.get("delta_threshold", config.UNSURE_DELTA_THRESHOLD)
            ),
        }
    except (OSError, ValueError):
        return {
            "confidence_threshold": config.UNSURE_CONFIDENCE_THRESHOLD,
            "delta_threshold": config.UNSURE_DELTA_THRESHOLD,
        }


def cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def knn_scores(vec: list, centroids: dict) -> dict:
    """Per-class cosine scores; prediction = argmax."""
    return {label: cosine(vec, cent) for label, cent in centroids.items()}


def knn_classify(vec: list, centroids: dict) -> tuple:
    """Returns (label, top1_cosine, per-label cosine vector in fixed sorted order)."""
    scores = knn_scores(vec, centroids)
    order = sorted(scores)
    label = max(order, key=lambda c: scores[c])
    top = max(scores.values())
    return label, top, [scores[c] for c in order]


def is_outage_error(exc: BaseException) -> bool:
    return isinstance(exc, EmbedderOutage)


def llama_healthy(timeout: float = 2.0) -> bool:
    """Cheap liveness probe for the llama.cpp server (GET /health)."""
    import urllib.request
    try:
        with urllib.request.urlopen(
            config.LLAMA_ENDPOINT.rstrip("/") + "/health", timeout=timeout
        ) as resp:
            return resp.status == 200
    except Exception:
        return False
