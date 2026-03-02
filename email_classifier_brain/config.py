"""
config.py — Shared Configuration & Input Formatting
====================================================

Central module used by both train.py and classify.py to ensure
the model input format is always consistent between training and
inference.

Configuration is loaded from a `.env` file in the project root.
Copy `.env.example` to `.env` and fill in your values.
"""

import email.header
import os
import re
from html.parser import HTMLParser
from io import StringIO

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# User Configuration (from .env)
# ---------------------------------------------------------------------------

# Automatic classification toggle.
# If "false", the background job won't be scheduled.
ENABLE_AUTO_CLASSIFICATION = (os.getenv("ENABLE_AUTO_CLASSIFICATION") or "true").lower() == "true"

# Enable the background job that checks for label corrections from the server
ENABLE_RECHECK_JOB = (os.getenv("ENABLE_RECHECK_JOB") or "true").lower() == "true"

# Interval in hours for the recheck job (default: 12)
try:
    RECHECK_INTERVAL_HOURS = int(os.getenv("RECHECK_INTERVAL_HOURS") or "12")
except ValueError:
    RECHECK_INTERVAL_HOURS = 12

# Enable the background job that periodically re-classifies existing emails
ENABLE_RECLASSIFY_JOB = (os.getenv("ENABLE_RECLASSIFY_JOB") or "true").lower() == "true"

# Interval in hours for the reclassify job (default: same as RECHECK_INTERVAL_HOURS)
try:
    RECLASSIFY_INTERVAL_HOURS = int(os.getenv("RECLASSIFY_INTERVAL_HOURS") or str(RECHECK_INTERVAL_HOURS))
except ValueError:
    RECLASSIFY_INTERVAL_HOURS = RECHECK_INTERVAL_HOURS

# Label used to explicitly verify a classification
VERIFICATION_LABEL = os.getenv("VERIFICATION_LABEL") or "VERIFIED"

# Label applied alongside the primary category when the classifier is unsure
UNSURE_LABEL = os.getenv("UNSURE_LABEL") or "UNSURE_CLASSIFICATION"

# Confidence score below which classification is considered unsure (0.0–1.0)
# If the top class probability is below this threshold, the UNSURE_LABEL is applied.
try:
    UNSURE_CONFIDENCE_THRESHOLD = float(os.getenv("UNSURE_CONFIDENCE_THRESHOLD") or "0.65")
except ValueError:
    UNSURE_CONFIDENCE_THRESHOLD = 0.65

# Probability gap between top-2 predictions below which classification is considered unsure
# If top1_prob - top2_prob < this delta, the model cannot clearly distinguish between categories.
try:
    UNSURE_DELTA_THRESHOLD = float(os.getenv("UNSURE_DELTA_THRESHOLD") or "0.10")
except ValueError:
    UNSURE_DELTA_THRESHOLD = 0.10

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
STORAGE_DIR = os.getenv("STORAGE_DIR") or "storage"
DB_PATH = os.getenv("DB_PATH") or os.path.join(STORAGE_DIR, "email_history.db")
E5_PREFIX = "passage: "

# Regex to match long tracking/marketing URLs (common in newsletter emails)
_TRACKING_URL_PATTERN = re.compile(
    r'https?://\S{100,}',  # URLs longer than 100 chars are almost always tracking
)

# Regex to collapse multiple blank lines / excessive whitespace
_MULTI_NEWLINE_PATTERN = re.compile(r'\n\s*\n\s*\n+')
_MULTI_SPACE_PATTERN = re.compile(r'[ \t]{2,}')


class _HTMLTextExtractor(HTMLParser):
    """
    Minimal stdlib-based HTML-to-text extractor.

    Strips all HTML tags and extracts visible text content.
    Ignores content inside <style> and <script> elements.
    """

    def __init__(self):
        super().__init__()
        self._result = StringIO()
        self._skip_depth = 0
        self._skip_tags = {"style", "script", "head"}

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() in self._skip_tags:
            self._skip_depth += 1
        elif tag.lower() in ("br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._result.write("\n")

    def handle_endtag(self, tag: str):
        if tag.lower() in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            self._result.write(data)

    def get_text(self) -> str:
        return self._result.getvalue()


def clean_subject(raw_subject: str) -> str:
    """
    Decode MIME-encoded email subjects to plain text.

    Handles both Quoted-Printable (=?UTF-8?Q?...?=) and
    Base64 (=?UTF-8?B?...?=) encodings. Already-decoded text
    passes through unchanged.

    Args:
        raw_subject: The raw subject string (possibly MIME-encoded).

    Returns:
        Decoded plain-text subject string.
    """
    if not raw_subject:
        return ""

    try:
        decoded_parts = email.header.decode_header(raw_subject)
        parts = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                parts.append(part.decode(charset or "utf-8", errors="replace"))
            else:
                parts.append(part)
        return " ".join(parts).strip()
    except Exception:
        # If decoding fails for any reason, return the original
        return raw_subject.strip()


def clean_body(raw_body: str) -> str:
    """
    Clean email body text for use in classification.

    If the body contains HTML tags, strips them and extracts visible
    text. Also removes tracking URLs and collapses excessive whitespace.
    Already-clean plain text passes through with minimal changes
    (only whitespace normalization).

    Args:
        raw_body: The raw body string (may be HTML or plain text).

    Returns:
        Cleaned plain-text body.
    """
    if not raw_body:
        return ""

    text = raw_body

    # If it looks like HTML, extract text
    if "<" in text and (">" in text) and any(
        tag in text.lower() for tag in ("<html", "<body", "<div", "<table", "<p>", "<!doctype", "<br")
    ):
        extractor = _HTMLTextExtractor()
        try:
            extractor.feed(text)
            text = extractor.get_text()
        except Exception:
            pass  # Fall back to original text if parsing fails

    # Remove tracking URLs (very long URLs, typically from marketing emails)
    text = _TRACKING_URL_PATTERN.sub("", text)

    # Collapse excessive whitespace
    text = _MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text)

    # Remove zero-width spaces and other invisible Unicode chars
    text = text.replace("\u200c", "").replace("\u200b", "").replace("\u200d", "")

    return text.strip()

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
