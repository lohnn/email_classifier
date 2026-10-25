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
import logging
import os
import re
import threading
from html.parser import HTMLParser
from io import StringIO

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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

# ---------------------------------------------------------------------------
# Read-mail backfill sweep (WI-103)
# ---------------------------------------------------------------------------
# Once the unread (UNSEEN) queue is fully drained, the classifier may label
# older already-read (\Seen) mail that was never classified — gradually, a
# bounded batch per classification cycle, until mailbox history is done.
#
# Hard gate (enforced in jobs/classification.py, not by convention): read-mail
# work only runs when has_unprocessed_unseen_emails() is False — no UNSEEN
# email in the INBOX selection lacks all trained-category labels. Unread
# always wins; the two phases never interleave.
#
# Default: ENABLED. Turning it on flips no safety rail — the hard unread-first
# gate above, the trained-label filter, and the processed-marker skip_ids all
# apply unchanged — and the batch default (5) keeps the gradual per-cycle pace.
# Set ENABLE_READ_SWEEP=false to opt out entirely.
ENABLE_READ_SWEEP = (os.getenv("ENABLE_READ_SWEEP") or "true").lower() == "true"

# Maximum read emails labelled per classification cycle once the sweep is
# enabled and the hard gate passes. 5 per 5-minute cycle ≈ 1440/day, so a
# multi-thousand-email backlog sweeps in days, not in one blast.
READ_SWEEP_BATCH_SIZE = int(os.getenv("READ_SWEEP_BATCH_SIZE") or "5")
if READ_SWEEP_BATCH_SIZE < 1:
    READ_SWEEP_BATCH_SIZE = 1

# Label used to explicitly verify a classification
VERIFICATION_LABEL = os.getenv("VERIFICATION_LABEL") or "VERIFIED"

# Label applied alongside the primary category when the classifier is unsure
UNSURE_LABEL = os.getenv("UNSURE_LABEL") or "UNSURE_CLASSIFICATION"

def _get_float_env(var_name: str, default: float) -> float:
    try:
        return float(os.getenv(var_name) or str(default))
    except ValueError:
        return default


# Confidence score below which classification is considered unsure (0.0–1.0)
# If the top class probability is below this threshold, the UNSURE_LABEL is applied.
UNSURE_CONFIDENCE_THRESHOLD = _get_float_env("UNSURE_CONFIDENCE_THRESHOLD", 0.65)

# Probability gap between top-2 predictions below which classification is considered unsure
# If top1_prob - top2_prob < this delta, the model cannot clearly distinguish between categories.
UNSURE_DELTA_THRESHOLD = _get_float_env("UNSURE_DELTA_THRESHOLD", 0.10)

# Daily-update health knob: how many days without any recorded label correction
# (while mail volume is non-zero) before the auto-update job warns about a
# possibly-stalled correction→training-data loop in update_history.json.
# The recheck cadence itself stays at RECHECK_INTERVAL_HOURS below.
ZERO_CORRECTION_WARN_DAYS = int(os.getenv("ZERO_CORRECTION_WARN_DAYS") or 14)

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

# ---------------------------------------------------------------------------
# Embedder backend — production serving path is the llama.cpp endpoint
# serving the LoRA-merged GGUF (embeddinggemma-2 text tower, 768d, mean
# pooling). 'st' is the full rollback: the in-process SetFit path, flipped
# only as a deliberate ops action. An llama.cpp outage is surfaced LOUDLY
# (error log + update_history warning) and never silently degraded —
# 2026-10 wave lessons: silent degrade is how drift ships.
# ---------------------------------------------------------------------------
EMBEDDER_BACKEND = (os.getenv("EMBEDDER_BACKEND") or "llama_cpp").strip().lower()
LLAMA_ENDPOINT = os.getenv("LLAMA_ENDPOINT") or "http://127.0.0.1:8090"
LLAMA_TIMEOUT_SECONDS = _get_float_env("LLAMA_TIMEOUT_SECONDS", 30.0)
LLAMA_RETRIES = int(os.getenv("LLAMA_RETRIES") or "3")
LLAMA_RETRY_BACKOFF_SECONDS = _get_float_env("LLAMA_RETRY_BACKOFF_SECONDS", 2.0)
# Classification task prefix applied IN THE CALLER (llama.cpp takes raw
# text) — byte-for-byte the string the lab validated (FINDINGS §16: dev
# 0.900 / test ≥0.889 with caller-applied prefix on served Q8).
CLS_PROMPT = os.getenv("CLS_PROMPT", "task: classification | query: ")
# kNN centroids shipped with the model artifact (alongside label_mapping.json;
# built by the retrain pipeline's build_centroids.py against the exact GGUF).
CENTROIDS_PATH = os.getenv("CENTROIDS_PATH") or os.path.join(
    MODEL_OUTPUT_DIR, "label_centroids.json"
)

# Model-artifact upload/install (POST /admin/upload-model-artifact +
# /admin/install-model-artifact — TAILSCALE-PEER-ONLY routes). The staging
# dir lives under STORAGE_DIR so staging→install stays on one filesystem
# (atomic os.replace). The GGUF install target is the path llama.cpp loads
# (email-embedder.service -m path on the Pi); the ST set installs into
# MODEL_OUTPUT_DIR.
MODEL_STAGING_DIR = os.getenv("MODEL_STAGING_DIR") or os.path.join(
    STORAGE_DIR, "model_staging"
)
LLAMA_MODEL_FILE = os.getenv("LLAMA_MODEL_FILE") or os.path.join(
    "llama_model", "merged_text_q8_0.gguf"
)
MAX_MODEL_UPLOAD_BYTES = int(os.getenv("MAX_MODEL_UPLOAD_BYTES") or 2 * 1024 * 1024 * 1024)

# ---------------------------------------------------------------------------
# Context-window budget (intfloat/multilingual-e5-small)
# ---------------------------------------------------------------------------
#
# multilingual-e5-small accepts at most 512 tokens (including special tokens).
# The tokenizer truncates silently from the end, which would silently eat body
# text for long emails (newsletters, reply chains). To make truncation
# deterministic and identical between training and inference, the body is
# capped HERE, inside format_model_input(), before the string is built.
#
# Measured with AutoTokenizer.from_pretrained(BASE_MODEL) against the real
# training data (2026-08-24):
#   - bare template (all fields empty):           41 tokens
#   - typical metadata prefix (real data, p90):   ~65 tokens
#   - worst-case plausible prefix (long From/To/CC/Subject/attachments): 161 tokens
#   - English prose ratio: ~4.7 chars/token (UTF-8 bytes ≈ chars here)
#
# Budget: 512 - 161 (worst prefix) - 31 (safety margin for tokenization
# variance and multilingual expansion) = 320 tokens for the body.
# At a conservative 4.0 chars/token (vs. measured 4.7) that is 1280 chars.
MODEL_MAX_TOKENS = 512
BODY_TOKEN_BUDGET = 320

# Conservative characters-per-token ratio used to convert the token budget
# into a character cap without shipping a tokenizer in the formatting path.
_CHARS_PER_TOKEN = 4.0

# Marker inserted between the kept head and tail when a body is truncated.
# Pure ASCII so it never interacts with encoding edge cases.
TRUNCATION_MARKER = " [...] "

# Maximum body length in characters (derived from the token budget above).
MAX_BODY_CHARS = int(BODY_TOKEN_BUDGET * _CHARS_PER_TOKEN)


class _TruncationStats:
    """Process-global counters for body truncation events.

    Thread-safe: the classification/reclassification jobs run on the
    single JobQueue worker thread while the API is served from the main
    thread, so both can touch these counters.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._total = 0
        self._truncated = 0

    def record(self, truncated: bool) -> None:
        with self._lock:
            self._total += 1
            if truncated:
                self._truncated += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "model_inputs_total": self._total,
                "model_inputs_truncated": self._truncated,
            }

    def reset(self) -> None:
        with self._lock:
            self._total = 0
            self._truncated = 0


truncation_stats = _TruncationStats()


class _SweepStats:
    """Process-global counters for the read-mail backfill sweep (WI-103).

    Observable signal, per the W-022 rule that a prevalence count that only
    lands in a log file is invisible: gated/candidate/labelled counts surface
    via GET /stats. Thread-safe for the same reason as _TruncationStats (the
    classification job runs on the JobQueue worker thread while /stats is
    served from the main thread).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._attempts = 0
        self._gated = 0
        self._candidates = 0
        self._labeled = 0

    def record(self, attempt: bool = False, gated: bool = False,
               candidates: int = 0, labeled: int = 0) -> None:
        """Record sweep events. `attempt=True` marks one sweep-phase execution;
        gated/candidates/labeled are the outcome deltas of that execution."""
        with self._lock:
            if attempt:
                self._attempts += 1
            if gated:
                self._gated += 1
            self._candidates += candidates
            self._labeled += labeled

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "read_sweep_attempts_total": self._attempts,
                "read_sweep_gated_total": self._gated,
                "read_sweep_candidates_total": self._candidates,
                "read_sweep_labeled_total": self._labeled,
            }

    def reset(self) -> None:
        with self._lock:
            self._attempts = 0
            self._gated = 0
            self._candidates = 0
            self._labeled = 0


sweep_stats = _SweepStats()


def _truncate_body(body: str, max_chars: int = MAX_BODY_CHARS) -> tuple[str, bool]:
    """
    Deterministically cap a body to fit the E5 context-window budget.

    Keeps the first half and the last half of the allowed character window,
    joined by TRUNCATION_MARKER, so signatures and closing paragraphs survive
    (they often carry class-relevant signal). Purely character-based and
    side-effect-free, so training and inference always produce byte-identical
    output for the same input.

    Args:
        body: The cleaned email body.
        max_chars: Character cap derived from BODY_TOKEN_BUDGET.

    Returns:
        (possibly_truncated_body, was_truncated)
    """
    if len(body) <= max_chars:
        return body, False

    marker_len = len(TRUNCATION_MARKER)
    head_len = (max_chars - marker_len) // 2
    tail_len = max_chars - marker_len - head_len
    return body[:head_len] + TRUNCATION_MARKER + body[-tail_len:], True

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
        A prefixed, structured string ready for model input. The body is
        deterministically capped (see _truncate_body) so the result fits
        within the E5 context window (MODEL_MAX_TOKENS tokens).
    """
    role = determine_role(to, cc)
    mass_mail_str = "Yes" if mass_mail else "No"
    att_types_str = format_attachment_types(attachment_types or [])

    original_body_len = len(body)
    body, was_truncated = _truncate_body(body)
    truncation_stats.record(was_truncated)
    if was_truncated:
        logger.info(
            "Body truncated to fit E5 context window: subject=%r, original_chars=%d, cap=%d",
            subject, original_body_len, MAX_BODY_CHARS,
        )

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
