"""
classify.py — Email Classification Inference (Raspberry Pi Optimized)
=====================================================================

Loads the fine-tuned SetFit model from `model/` and provides email
classification with rich metadata feature extraction.

Optimizations for Raspberry Pi 4 (4 GB RAM):
    - Explicit CPU-only inference (no GPU probing).
    - Model loaded once at module level to avoid repeated overhead.
    - The label mapping is loaded from `model/label_mapping.json`,
      so categories stay in sync with training automatically.

Usage as a library:
    from classify import predict_email

    label = predict_email(
        subject="Server is down!",
        body="All services are offline since 14:00.",
        sender="ops@company.com",
        to="me@company.com",
        mass_mail=False,
        attachment_types=["PDF"],
    )

    # Or from a raw email.message.Message:
    import email
    msg = email.message_from_file(open("email.eml"))
    label = predict_raw_email(msg)

Usage from the command line:
    python classify.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import logging

if TYPE_CHECKING:
    from setfit import SetFitModel  # type-checking only; never imported at runtime

logger = logging.getLogger(__name__)

import email
import email.message
import json
import mimetypes
import os

try:
    import setfit  # noqa: F401  (import check only — SetFit model load is LAZY)
except ImportError:  # pragma: no cover
    setfit = None

try:
    from config import (
        MODEL_OUTPUT_DIR, format_model_input, clean_subject, clean_body,
        UNSURE_CONFIDENCE_THRESHOLD, UNSURE_DELTA_THRESHOLD,
        EMBEDDER_BACKEND, CLS_PROMPT, CENTROIDS_PATH,
    )
except ImportError:
    from classifier_brain.config import (
        MODEL_OUTPUT_DIR, format_model_input, clean_subject, clean_body,
        UNSURE_CONFIDENCE_THRESHOLD, UNSURE_DELTA_THRESHOLD,
        EMBEDDER_BACKEND, CLS_PROMPT, CENTROIDS_PATH,
    )

import embedder as _embedder
import config as _config


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = MODEL_OUTPUT_DIR


# ---------------------------------------------------------------------------
# Load model and label mapping (once, at import time)
# ---------------------------------------------------------------------------

def _load_model() -> SetFitModel:
    """Load the SetFit model for CPU-only inference."""
    print(f"Loading model from '{MODEL_DIR}/'...")
    model = SetFitModel.from_pretrained(MODEL_DIR)
    # Ensure the model is on CPU (avoids accidental GPU probing)
    model.model_body = model.model_body.to("cpu")
    print("Model loaded successfully.")
    return model


def _load_label_mapping() -> dict[int, str]:
    """Load the integer-to-label mapping saved during training."""
    mapping_path = os.path.join(MODEL_DIR, "label_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # JSON keys are strings; convert to int
    return {int(k): v for k, v in raw.items()}


# Global singletons — loaded LAZILY on first ST-path use (the llama.cpp
# production path never allocates the 500 MB SetFit stack). Skip eager load
# during tests.
_model = None
_label_mapping = {}
_st_loaded = False
# once-per-process flag: the EMPTY-vocabulary ERROR (get_available_categories)
# fires a single time so a broken source is loud without spamming every cycle
_vocabulary_warned = False


def _get_st():
    """Lazily load the SetFit model + label mapping (rollback backend only)."""
    global _model, _label_mapping, _st_loaded
    if os.getenv("TESTING") == "true":
        raise RuntimeError("SetFit path not available under TESTING=true")
    if not _st_loaded:
        _model = _load_model()
        _label_mapping = _load_label_mapping()
        _st_loaded = True
    return _model, _label_mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_unsure_classification(probs, confidence_threshold: float | None = None,
                             delta_threshold: float | None = None) -> bool:
    """
    Determine whether a classification result should be flagged as uncertain.

    Returns True if either of the following conditions holds:
    - The top class probability is below the confidence threshold (low overall
      confidence), OR
    - The gap between the top-two class probabilities is below the delta
      threshold (the model cannot clearly distinguish between two categories).

    The gate logic is backend-independent: SetFit probabilities use the
    classic thresholds; llama.cpp cosine scores use the thresholds calibrated
    in the centroids artifact (the cosine scale differs slightly
    post-quantization — FINDINGS §14/§16).

    Args:
        probs: Array-like of per-class scores (probabilities or cosines).
        confidence_threshold / delta_threshold: backend-specific thresholds;
            None defaults to the SetFit-era constants from config.

    Returns:
        True if the classification is uncertain, False otherwise.
    """
    conf_thr = UNSURE_CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
    delta_thr = UNSURE_DELTA_THRESHOLD if delta_threshold is None else delta_threshold
    sorted_probs = sorted((float(p) for p in probs), reverse=True)
    top_score = sorted_probs[0]

    if top_score < conf_thr:
        return True

    if len(sorted_probs) >= 2:
        delta = top_score - sorted_probs[1]
        if delta < delta_thr:
            return True

    return False


def _predict_llama(model_input: str, return_score: bool):
    """llama.cpp production path: embed the prefix-applied input via the
    local server, kNN over the shipped L2 centroids, UNSURE gate on the
    calibrated cosine thresholds from the centroids artifact."""
    vec = _embedder.embed_texts([model_input])[0]
    centroids = _embedder.load_centroids(_config.CENTROIDS_PATH)
    label, score, scores_vec = _embedder.knn_classify(vec, centroids)
    if return_score:
        gate = _embedder.artifact_gate_thresholds(_config.CENTROIDS_PATH)
        unsure = is_unsure_classification(
            scores_vec,
            confidence_threshold=gate["confidence_threshold"],
            delta_threshold=gate["delta_threshold"],
        )
        return label, float(score), unsure
    return label


def predict_email(
    subject: str,
    body: str,
    sender: str = "",
    to: str = "",
    cc: str = "",
    mass_mail: bool = False,
    attachment_types: list[str] | None = None,
    return_score: bool = False,
) -> str | tuple[str, float, bool]:
    """
    Classify an email and return the predicted category label.

    Constructs the structured metadata input string, runs inference,
    and maps the integer prediction back to its label string.

    Args:
        subject: The email subject line.
        body: The email body text.
        sender: The "From" header value.
        to: The "To" header value.
        cc: The "CC" header value.
        mass_mail: Whether the email has a List-Unsubscribe header.
        attachment_types: List of file extensions (e.g. ["PDF", "ICS"]).
                          None or [] means no attachments.
        return_score: If True, returns a tuple (label, confidence_score, is_unsure).

    Returns:
        A category string (e.g. "URGENT", "FOCUS", "REFERENCE", "NOISE"),
        or any custom category discovered during training.
        If return_score is True, returns a tuple (category, score, is_unsure) where
        is_unsure indicates the classifier has low confidence or two categories are
        very close in probability.
    """
    model_input = format_model_input(
        subject=subject,
        body=body,
        sender=sender,
        to=to,
        cc=cc,
        mass_mail=mass_mail,
        attachment_types=attachment_types,
    )

    if _config.EMBEDDER_BACKEND == "llama_cpp":
        # Production path: local llama.cpp server (LoRA-merged GGUF) +
        # kNN over the shipped centroids. Outages raise EmbedderOutage
        # (surfaced LOUDLY in embedder._surface_outage) — flip
        # EMBEDDER_BACKEND=st only as a deliberate ops action.
        return _predict_llama(model_input, return_score)

    model, label_mapping = _get_st()
    if return_score:
        # Get probabilities for all classes
        probs = model.predict_proba([model_input])[0]
        # Find the index of the highest probability
        predicted_index = int(probs.argmax())
        score = float(probs[predicted_index])

        label = label_mapping.get(predicted_index, f"UNKNOWN({predicted_index})")
        unsure = is_unsure_classification(probs)
        return label, score, unsure

    prediction = model.predict([model_input])

    # SetFit returns a tensor or array; extract the integer label
    predicted_index = int(prediction[0])
    label = label_mapping.get(predicted_index, f"UNKNOWN({predicted_index})")
    return label


def extract_email_info(msg: email.message.Message) -> dict:
    """
    Extract relevant metadata and content from a raw email message.

    Returns a dictionary with keys: sender, to, cc, subject, body,
    mass_mail, and attachment_types.
    """
    sender = str(msg.get("From", ""))
    to = str(msg.get("To", ""))
    cc = str(msg.get("Cc", ""))
    subject = str(msg.get("Subject", ""))

    # Mass mail detection via List-Unsubscribe header
    mass_mail = msg.get("List-Unsubscribe") is not None

    # Extract body text and attachment types
    body = ""
    attachment_types: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Collect attachment file extensions (skip inline images)
            if "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    ext = os.path.splitext(filename)[1].lstrip(".").upper()
                    if ext:
                        attachment_types.append(ext)
                else:
                    # No filename — guess from MIME type
                    ext = mimetypes.guess_extension(content_type)
                    if ext:
                        attachment_types.append(ext.lstrip(".").upper())

            # Extract plain text body
            elif content_type == "text/plain" and not body:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")
    else:
        # Single-part message
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace")

    # Deduplicate attachment types while preserving order
    seen: set[str] = set()
    unique_types: list[str] = []
    for t in attachment_types:
        if t not in seen:
            seen.add(t)
            unique_types.append(t)

    return {
        "sender": sender,
        "to": to,
        "cc": cc,
        "subject": clean_subject(subject),
        "body": clean_body(body),
        "mass_mail": mass_mail,
        "attachment_types": unique_types,
    }


def predict_raw_email(msg: email.message.Message, return_score: bool = False) -> str | tuple[str, float, bool]:
    """
    Classify a raw email.message.Message by auto-extracting headers.

    Extracts From, To, CC, List-Unsubscribe, Subject, body text, and
    attachment types from the email message object.

    Args:
        msg: A Python email.message.Message (e.g. from email.message_from_file).
        return_score: If True, returns a tuple (label, confidence_score, is_unsure).

    Returns:
        A category string or tuple (category, score, is_unsure).
    """
    info = extract_email_info(msg)

    return predict_email(
        subject=info["subject"],
        body=info["body"],
        sender=info["sender"],
        to=info["to"],
        cc=info["cc"],
        mass_mail=info["mass_mail"],
        attachment_types=info["attachment_types"],
        return_score=return_score,
    )


def get_available_categories() -> list[str]:
    """Return the list of all category labels the model was trained on.

    llama.cpp backend: the shipped centroids' keys (the artifact that
    defines the production vocabulary). An unreadable/missing artifact
    falls back to the SetFit-era label_mapping — which is EMPTY on the
    llama path unless the ST stack has actually been lazily loaded in this
    process. That silent fallback is exactly what kept the read
    sweep skipped on the Pi ("no trained categories" every cycle while
    classification itself worked), so two vaccinations were added
    (2026-10-26 debug round):
      - the fallback logs a LOUD warning naming both candidate paths, their
        absolute resolution and existence (see /admin/sweep-state);
      - an EMPTY final resolution logs a once-per-process ERROR —
        predictions may still run per-email, but the label filter (UNSEEN
        fetch, read sweep) would match nothing.
    Never fabricates labels: an empty source resolution returns [].
    """
    global _vocabulary_warned
    categories: list[str] = []

    if _config.EMBEDDER_BACKEND == "llama_cpp":
        try:
            categories = sorted(_embedder.load_centroids(_config.CENTROIDS_PATH).keys())
        except OSError as e:
            mapping_path = os.path.join(MODEL_DIR, "label_mapping.json")
            logger.warning(
                "get_available_categories: centroids artifact unreadable "
                "(path=%r, abspath=%r, exists=%s; %s); falling back to the "
                "SetFit-era label_mapping (path=%r; exists=%s, entries=%d). "
                "That fallback is empty unless the ST path ran in this "
                "process — if classification works, check the model-artifact "
                "install / configured CENTROIDS_PATH.",
                _config.CENTROIDS_PATH, os.path.abspath(str(_config.CENTROIDS_PATH)),
                os.path.exists(_config.CENTROIDS_PATH), e,
                mapping_path, os.path.exists(mapping_path), len(_label_mapping),
            )
            categories = sorted(_label_mapping.values())
    else:
        categories = sorted(_label_mapping.values())

    if not categories and not _vocabulary_warned:
        _vocabulary_warned = True
        logger.error(
            "get_available_categories resolved an EMPTY vocabulary "
            "(backend=%s, centroids path=%r exists=%s, label_mapping "
            "entries=%d). Per-email predictions may still work, but the "
            "label filter (UNSEEN fetch, read sweep) matches nothing and "
            "the read sweep skips every cycle. Inspect "
            "GET /admin/sweep-state for the resolved source state.",
            _config.EMBEDDER_BACKEND, _config.CENTROIDS_PATH,
            os.path.exists(_config.CENTROIDS_PATH), len(_label_mapping),
        )
    return categories


def _reset_centroid_cache() -> None:
    """Test/diagnostic hook: forget the cached centroids artifact so the
    next load_centroids() re-reads the configured path."""
    _embedder._centroid_cache = None


# ---------------------------------------------------------------------------
# CLI Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Available categories: {get_available_categories()}\n")

    # Example emails with metadata
    examples = [
        {
            "subject": "Server down!",
            "body": "All production services are offline since 14:00.",
            "sender": "ops-alert@company.com",
            "to": "me@company.com",
            "cc": "",
            "mass_mail": False,
            "attachment_types": [],
        },
        {
            "subject": "Q3 Budget",
            "body": "Please prepare your department estimates by Thursday.",
            "sender": "finance-director@company.com",
            "to": "me@company.com",
            "cc": "",
            "mass_mail": False,
            "attachment_types": ["XLSX"],
        },
        {
            "subject": "Meeting notes",
            "body": "Attached are the minutes from yesterday's sync.",
            "sender": "architect@company.com",
            "to": "engineering@company.com",
            "cc": "me@company.com",
            "mass_mail": False,
            "attachment_types": ["PDF"],
        },
        {
            "subject": "Friday lunch",
            "body": "Who wants to order pizza for the team lunch?",
            "sender": "colleague@company.com",
            "to": "all-staff@company.com",
            "cc": "",
            "mass_mail": True,
            "attachment_types": [],
        },
    ]

    for ex in examples:
        result = predict_email(
            subject=ex["subject"],
            body=ex["body"],
            sender=ex["sender"],
            to=ex["to"],
            cc=ex["cc"],
            mass_mail=ex["mass_mail"],
            attachment_types=ex["attachment_types"],
            return_score=True,
        )
        print(f"Subject : {ex['subject']}")
        print(f"From    : {ex['sender']}")
        print(f"To      : {ex['to']}")
        att = ex["attachment_types"] or "None"
        print(f"Attach  : {att}")
        print(f"Category: {result}")
        print("-" * 50)
