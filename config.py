"""
config.py — Shared Configuration & Input Formatting
====================================================

Central module used by both train.py and classify.py to ensure
the model input format is always consistent between training and
inference.
"""

# ---------------------------------------------------------------------------
# User Configuration
# ---------------------------------------------------------------------------

# Set this to your email address. Used to determine your role in an email:
#   - "Direct" if MY_EMAIL is in the "To" field
#   - "CC" if MY_EMAIL is in the "CC" field
#   - "Hidden" otherwise (BCC, mailing list, etc.)
MY_EMAIL = "me@company.com"

# ---------------------------------------------------------------------------
# Model & Paths
# ---------------------------------------------------------------------------

BASE_MODEL = "intfloat/multilingual-e5-small"
MODEL_OUTPUT_DIR = "model"
TRAINING_DATA_DIR = "TrainingData"
E5_PREFIX = "passage: "


# ---------------------------------------------------------------------------
# Role Detection
# ---------------------------------------------------------------------------

def determine_role(to: str, cc: str) -> str:
    """
    Determine the user's role in an email based on To/CC fields.

    Args:
        to: The "To" header value (may contain multiple addresses).
        cc: The "CC" header value (may contain multiple addresses).

    Returns:
        "Direct" if MY_EMAIL is in To, "CC" if in CC, else "Hidden".
    """
    my_email_lower = MY_EMAIL.lower()
    if my_email_lower in to.lower():
        return "Direct"
    if my_email_lower in cc.lower():
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
