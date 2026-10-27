"""
test_truncation.py — Tests for E5 context-window body truncation
================================================================

Tests the deterministic body cap in config.format_model_input()
(WI-021): model inputs must never silently exceed the e5-small
512-token window, and truncation must be identical between the
training and inference paths (both call config.format_model_input).
"""

import os
import sys

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from config import (
    BODY_TOKEN_BUDGET,
    MAX_BODY_CHARS,
    MODEL_MAX_TOKENS,
    TRUNCATION_MARKER,
    _truncate_body,
    format_model_input,
)


@pytest.fixture(autouse=True)
def reset_truncation_stats():
    """Isolate the process-global truncation counter between tests."""
    config.truncation_stats.reset()
    yield
    config.truncation_stats.reset()


# ---------------------------------------------------------------------------
# _truncate_body unit tests
# ---------------------------------------------------------------------------

class TestTruncateBody:
    """Unit tests for the deterministic character cap."""

    def test_short_body_unchanged(self):
        body = "a" * (MAX_BODY_CHARS - 1)
        out, truncated = _truncate_body(body)
        assert truncated is False
        assert out == body

    def test_exactly_at_cap_unchanged(self):
        body = "a" * MAX_BODY_CHARS
        out, truncated = _truncate_body(body)
        assert truncated is False
        assert out == body

    def test_over_cap_is_truncated(self):
        body = "a" * (MAX_BODY_CHARS + 1)
        out, truncated = _truncate_body(body)
        assert truncated is True
        assert len(out) == MAX_BODY_CHARS

    def test_empty_body_unchanged(self):
        out, truncated = _truncate_body("")
        assert truncated is False
        assert out == ""

    def test_truncated_output_is_deterministic(self):
        body = "x" * (MAX_BODY_CHARS * 3)
        out1, _ = _truncate_body(body)
        out2, _ = _truncate_body(body)
        assert out1 == out2

    def test_truncation_keeps_head_and_tail(self):
        head = "H" * MAX_BODY_CHARS
        tail = "T" * MAX_BODY_CHARS
        body = head + "M" * 500 + tail
        out, truncated = _truncate_body(body)
        assert truncated is True
        assert TRUNCATION_MARKER in out
        # Head survives at the start, tail at the end
        assert out.startswith("H" * 100)
        assert out.endswith("T" * 100)
        # The middle is gone
        assert "M" not in out

    def test_marker_budget_math(self):
        """head + marker + tail must land exactly on the cap."""
        body = "b" * (MAX_BODY_CHARS + 123)
        out, truncated = _truncate_body(body)
        assert truncated is True
        assert len(out) == MAX_BODY_CHARS
        head, _, tail = out.partition(TRUNCATION_MARKER)
        assert len(head) + len(TRUNCATION_MARKER) + len(tail) == MAX_BODY_CHARS


# ---------------------------------------------------------------------------
# format_model_input integration tests
# ---------------------------------------------------------------------------

class TestFormatModelInputTruncation:
    """The cap must be applied inside format_model_input (single source of truth)."""

    def test_short_input_passes_through(self):
        result = format_model_input(
            subject="Hi", body="Short body.", sender="a@b.com", to="me@company.com",
        )
        assert "Body: Short body." in result
        assert TRUNCATION_MARKER not in result

    def test_long_body_capped_with_marker(self):
        body = "word " * 5000  # ~25k chars, way over the cap
        result = format_model_input(
            subject="Newsletter", body=body, sender="news@x.com", to="me@company.com",
        )
        assert TRUNCATION_MARKER in result
        body_part = result.split("Body: ", 1)[1]
        assert len(body_part) == MAX_BODY_CHARS

    def test_deterministic_identical_output(self):
        """Same input → byte-identical output, every time (train == inference)."""
        kwargs = dict(
            subject="Re: long thread",
            body="First para. " + "filler " * 4000 + "Last para, signature.",
            sender='"Some One" <someone@example.com>',
            to="me@company.com",
            cc="other@example.com",
            mass_mail=True,
            attachment_types=["PDF"],
        )
        first = format_model_input(**kwargs)
        second = format_model_input(**kwargs)
        assert first == second

    def test_truncated_input_fits_e5_window(self):
        """
        With the real tokenizer, a fully-truncated worst-case input must
        fit within MODEL_MAX_TOKENS. Skipped if transformers is absent.
        """
        transformers = pytest.importorskip("transformers")
        tok = transformers.AutoTokenizer.from_pretrained(config.BASE_MODEL)
        result = format_model_input(
            subject="Re: " + "quite a long subject line with many words " * 4,
            body="word " * 20000,
            sender='"A Very Long Display Name" <averylongaddress@some-domain.example.com>',
            to="a@example.com, b@example.com, c@example.com, me@company.com",
            cc="d@example.com, e@example.com",
            mass_mail=True,
            attachment_types=["PDF", "DOCX", "XLSX", "ICS", "PNG", "ZIP"],
        )
        n_tokens = len(tok.encode(result, add_special_tokens=True))
        assert n_tokens <= MODEL_MAX_TOKENS, (
            f"Truncated input is {n_tokens} tokens, exceeds {MODEL_MAX_TOKENS}"
        )

    def test_typical_input_well_under_window(self):
        """A typical short email must be nowhere near the cap."""
        transformers = pytest.importorskip("transformers")
        tok = transformers.AutoTokenizer.from_pretrained(config.BASE_MODEL)
        result = format_model_input(
            subject="Server is down",
            body="All services are offline! Engineers have been paged.",
            sender="ops-alert@company.com",
            to="me@company.com",
        )
        n_tokens = len(tok.encode(result, add_special_tokens=True))
        assert n_tokens <= MODEL_MAX_TOKENS // 4


# ---------------------------------------------------------------------------
# Truncation prevalence counter (observable signal)
# ---------------------------------------------------------------------------

class TestTruncationStats:
    """Prevalence must be an observable counter, not just a log line."""

    def test_counts_total_inputs(self):
        format_model_input(subject="a", body="short", to="me@company.com")
        format_model_input(subject="b", body="also short", to="me@company.com")
        snap = config.truncation_stats.snapshot()
        assert snap["model_inputs_total"] == 2
        assert snap["model_inputs_truncated"] == 0

    def test_counts_truncated_inputs(self):
        format_model_input(subject="a", body="short", to="me@company.com")
        format_model_input(subject="b", body="word " * 5000, to="me@company.com")
        snap = config.truncation_stats.snapshot()
        assert snap["model_inputs_total"] == 2
        assert snap["model_inputs_truncated"] == 1

    def test_reset_clears_counters(self):
        format_model_input(subject="a", body="word " * 5000, to="me@company.com")
        config.truncation_stats.reset()
        snap = config.truncation_stats.snapshot()
        assert snap["model_inputs_total"] == 0
        assert snap["model_inputs_truncated"] == 0


# ---------------------------------------------------------------------------
# Budget constants sanity
# ---------------------------------------------------------------------------

class TestBudgetConstants:
    def test_cap_leaves_room_for_metadata(self):
        """
        The body budget plus the worst measured metadata prefix (161 tokens,
        see config.py comment) plus a safety margin must fit in the window.
        """
        WORST_MEASURED_PREFIX_TOKENS = 161
        assert BODY_TOKEN_BUDGET + WORST_MEASURED_PREFIX_TOKENS < MODEL_MAX_TOKENS

    def test_marker_is_ascii(self):
        assert TRUNCATION_MARKER.isascii()
