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

import email
import email.message
import json
import mimetypes
import os

from setfit import SetFitModel

try:
    from config import MODEL_OUTPUT_DIR, format_model_input
except ImportError:
    from classifier_brain.config import MODEL_OUTPUT_DIR, format_model_input


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


# Global singletons — loaded once when the module is first imported
_model = _load_model()
_label_mapping = _load_label_mapping()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_email(
    subject: str,
    body: str,
    sender: str = "",
    to: str = "",
    cc: str = "",
    mass_mail: bool = False,
    attachment_types: list[str] | None = None,
    return_score: bool = False,
) -> str | tuple[str, float]:
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
        return_score: If True, returns a tuple (label, confidence_score).

    Returns:
        A category string (e.g. "URGENT", "FOCUS", "REFERENCE", "NOISE"),
        or any custom category discovered during training.
        If return_score is True, returns a tuple (category, score).
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

    if return_score:
        # Get probabilities for all classes
        probs = _model.predict_proba([model_input])[0]
        # Find the index of the highest probability
        predicted_index = int(probs.argmax())
        score = float(probs[predicted_index])

        label = _label_mapping.get(predicted_index, f"UNKNOWN({predicted_index})")
        return label, score

    prediction = _model.predict([model_input])

    # SetFit returns a tensor or array; extract the integer label
    predicted_index = int(prediction[0])
    label = _label_mapping.get(predicted_index, f"UNKNOWN({predicted_index})")
    return label


def predict_raw_email(msg: email.message.Message, return_score: bool = False) -> str | tuple[str, float]:
    """
    Classify a raw email.message.Message by auto-extracting headers.

    Extracts From, To, CC, List-Unsubscribe, Subject, body text, and
    attachment types from the email message object.

    Args:
        msg: A Python email.message.Message (e.g. from email.message_from_file).
        return_score: If True, returns a tuple (label, confidence_score).

    Returns:
        A category string or tuple (category, score).
    """
    sender = msg.get("From", "")
    to = msg.get("To", "")
    cc = msg.get("Cc", "")
    subject = msg.get("Subject", "")

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

    return predict_email(
        subject=subject,
        body=body,
        sender=sender,
        to=to,
        cc=cc,
        mass_mail=mass_mail,
        attachment_types=unique_types,
        return_score=return_score,
    )


def get_available_categories() -> list[str]:
    """Return the list of all category labels the model was trained on."""
    return sorted(_label_mapping.values())


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
        )
        print(f"Subject : {ex['subject']}")
        print(f"From    : {ex['sender']}")
        print(f"To      : {ex['to']}")
        att = ex["attachment_types"] or "None"
        print(f"Attach  : {att}")
        print(f"Category: {result}")
        print("-" * 50)
