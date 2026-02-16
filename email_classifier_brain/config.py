"""
config.py — Shared Configuration & Input Formatting
====================================================

Central module used by both train.py and classify.py to ensure
the model input format is always consistent between training and
inference.

Configuration is loaded from a `.env` file in the project root.
Copy `.env.example` to `.env` and fill in your values.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# User Configuration (from .env)
# ---------------------------------------------------------------------------

# Comma-separated list of your email addresses.
# Used to determine your role in an email:
#   - "Direct" if any of your addresses is in the "To" field
#   - "CC" if any is in the "CC" field
#   - "Hidden" otherwise (BCC, mailing list, etc.)
MY_EMAILS: list[str] = [
    addr.strip().lower()
    for addr in os.getenv("MY_EMAIL", "me@company.com").split(",")
    if addr.strip()
]

# ---------------------------------------------------------------------------
# Model & Paths
# ---------------------------------------------------------------------------

BASE_MODEL = "intfloat/multilingual-e5-small"
MODEL_OUTPUT_DIR = os.getenv("MODEL_DIR") or "../email_classifier_data/model"
TRAINING_DATA_DIR = os.getenv("TRAINING_DATA_DIR") or "TrainingData"
E5_PREFIX = "passage: "


# ---------------------------------------------------------------------------
# Role Detection
# ---------------------------------------------------------------------------

def determine_role(to: str, cc: str) -> str:
    """
    Determine the user's role in an email based on To/CC fields.

    Checks all addresses in MY_EMAILS against the To and CC headers.

    Args:
        to: The "To" header value (may contain multiple addresses).
        cc: The "CC" header value (may contain multiple addresses).

    Returns:
        "Direct" if any of MY_EMAILS is in To, "CC" if in CC, else "Hidden".
    """
    to_lower = to.lower()
    cc_lower = cc.lower()

    for addr in MY_EMAILS:
        if addr in to_lower:
            return "Direct"

    for addr in MY_EMAILS:
        if addr in cc_lower:
            return "CC"

    return "Hidden"


# ---------------------------------------------------------------------------
# Attachment Types Formatting
# ---------------------------------------------------------------------------

def format_attachment_types(attachment_types: list[str]) -> str:
    """
    Format attachment types for the model input string.

    Args:
        attachment_types: List of file extensions, e.g. ["PDF", "DOCX"].
                          Empty list means no attachments.

    Returns:
        "None" if empty, otherwise "[PDF, DOCX, ...]".
    """
    if not attachment_types:
        return "None"
    return f"[{', '.join(attachment_types)}]"


# ---------------------------------------------------------------------------
# Model Input Formatting
# ---------------------------------------------------------------------------

def format_model_input(
    subject: str,
    body: str,
    sender: str = "",
    to: str = "",
    cc: str = "",
    mass_mail: bool = False,
    attachment_types: list[str] | None = None,
) -> str:
    """
    Build the structured input string for the E5/SetFit model.

    This function is the single source of truth for input formatting.
    Both train.py and classify.py must use this to ensure consistency.

    Args:
        subject: Email subject line.
        body: Email body text.
        sender: "From" header value.
        to: "To" header value.
        cc: "CC" header value.
        mass_mail: Whether the email has List-Unsubscribe header.
        attachment_types: List of file extensions (e.g. ["PDF", "ICS"]).
                          None or [] means no attachments.

    Returns:
        A prefixed, structured string ready for model input.
    """
    role = determine_role(to, cc)
    mass_mail_str = "Yes" if mass_mail else "No"
    att_types_str = format_attachment_types(attachment_types or [])

    structured = (
        f"Role: {role} | "
        f"Mass Mail: {mass_mail_str} | "
        f"Attachment Types: {att_types_str} | "
        f"From: {sender} | "
        f"To: {to} | "
        f"Subject: {subject} | "
        f"Body: {body}"
    )

    return f"{E5_PREFIX}{structured}"
