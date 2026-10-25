"""WI-103 — read-mail backfill sweep.

Hard gating: read-email work may only run when the unread (UNSEEN) queue is
fully drained. Coverage here:
  - imap layer: fetch_read_unprocessed_emails (SEEN + label filter + skip_ids,
    server-side negation with fallback), has_unprocessed_unseen_emails probe
    (metadata-only, no body downloads)
  - job layer: gating (sweep skipped until drained), pacing (per-cycle batch
    cap from config), resumability (DB processed-marker as skip_ids; a failed
    label application leaves the email unmarked/retryable), label respect
    (never removes or overwrites known labels; known-labeled email never
    becomes a sweep candidate)
  - config: conservative defaults; /stats observability counters
"""

import imaplib
import os
import re
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

os.environ.setdefault("TRAINING_DATA_DIR", tempfile.mkdtemp(prefix="sweep-tests-"))
os.environ.setdefault("ADMIN_API_KEY", "testkey")
os.environ.setdefault("TESTING", "true")

import main  # noqa: F401  (same import-time contract as the other suites)
import config
import database
import imap_client
from jobs.classification import classification_job


# ---------------------------------------------------------------------------
# Fake INBOX connection (search + batched fetch surface used by the sweep)
# ---------------------------------------------------------------------------

class FakeMailboxConn:
    """
    Scripted INBOX for fetch paths. Messages are ordered by sequence number
    (list order = ascending seq = oldest..newest), like real IMAP SEARCH
    output. Supports UNSEEN/SEEN searches, honors NOT X-GM-LABELS clauses,
    and can be told to reject (tagged BAD) or ignore (server quirk →
    superset result) those clauses to exercise the client-side fallbacks.
    """

    def __init__(self, messages=None, reject_negation=False, ignore_negation=False):
        self.messages = messages or []
        self.reject_negation = reject_negation
        self.ignore_negation = ignore_negation
        self.search_calls = []
        self.fetch_calls = []

    def noop(self):
        return ("OK", [b""])

    @staticmethod
    def _negated_labels(criteria):
        quoted_or_atom = re.findall(r'NOT X-GM-LABELS\s+("[^"]*"|\S+)', criteria)
        return {m.strip('"') for m in quoted_or_atom}

    def search(self, charset, criteria):
        self.search_calls.append(criteria)
        if self.reject_negation and "NOT X-GM-LABELS" in criteria:
            raise imaplib.IMAP4.error("BAD Could not parse command")
        neg = set() if self.ignore_negation else self._negated_labels(criteria)
        ids = []
        for seq, msg in enumerate(self.messages, start=1):
            if "UNSEEN" in criteria.split() and msg["seen"]:
                continue
            if "SEEN" in criteria.split() and not msg["seen"]:
                continue
            if neg and any(lbl in neg for lbl in msg["labels"]):
                continue
            ids.append(str(seq).encode())
        return ("OK", [b" ".join(ids)])

    def fetch(self, ids_bytes, query):
        self.fetch_calls.append((ids_bytes, query))
        out = []
        for raw_id in ids_bytes.split(b","):
            seq = int(raw_id)
            if seq < 1 or seq > len(self.messages):
                continue
            msg = self.messages[seq - 1]
            if "BODY.PEEK" in query:
                raw_email = f'Subject: {msg["subject"]}\r\n\r\nbody-{msg["gid"]}'.encode()
                header = (
                    f'{seq} (FLAGS (\\Seen) BODY[] {{{len(raw_email)}}} '
                    f'X-GM-MSGID {msg["gid"]})'
                ).encode()
                out.append((header, raw_email))
            else:
                labels = " ".join(msg["labels"])
                out.append(f'{seq} (X-GM-LABELS ({labels}) X-GM-MSGID {msg["gid"]})'.encode())
        return ("OK", out)


def _client(conn):
    """A GmailClient bound to the scripted connection (bypasses credentials)."""
    client = imap_client.GmailClient()
    client.user = "test@example.com"
    client.password = "not-a-real-password"
    client.connection = conn
    return client


def _msg(gid, seen, labels, subject):
    return {"gid": gid, "seen": seen, "labels": labels, "subject": subject}


def _body_fetch_ids(conn):
    return [
        {int(b) for b in ids.split(b",")}
        for ids, query in conn.fetch_calls
        if "BODY.PEEK" in query
    ]


# ---------------------------------------------------------------------------
# imap layer — fetch_read_unprocessed_emails
# ---------------------------------------------------------------------------

def test_read_sweep_fetches_only_unlabeled_read_mail():
    conn = FakeMailboxConn([
        _msg("1001", True, [], "Old unprocessed"),          # sweep candidate
        _msg("1002", True, ["URGENT"], "Labeled by user"),  # labeled read — never touched
        _msg("1003", False, [], "Fresh unread"),            # unread — unread path's job
        _msg("1004", True, [], "Another candidate"),
    ])
    result = _client(conn).fetch_read_unprocessed_emails(known_labels=["URGENT"])

    assert [gid for gid, _msg_obj in result] == ["1004", "1001"]  # newest first
    # Server-side reduction: SEEN + NOT of the known labels
    assert conn.search_calls[-1] == 'SEEN NOT X-GM-LABELS "URGENT"'
    # Bodies downloaded only for the two surviving candidates (seqs 1 and 4)
    body_ids = set().union(*_body_fetch_ids(conn)) if _body_fetch_ids(conn) else set()
    assert body_ids == {1, 4}


def test_read_sweep_negation_rejected_falls_back_to_plain_seen():
    conn = FakeMailboxConn(
        [
            _msg("2001", True, [], "Unlabeled read"),
            _msg("2002", True, ["URGENT"], "Labeled read"),
            _msg("2003", False, [], "Unread"),
        ],
        reject_negation=True,
    )
    result = _client(conn).fetch_read_unprocessed_emails(known_labels=["URGENT"])

    assert [gid for gid, _m in result] == ["2001"]
    # First attempt used the negated criteria, then the fallback used plain SEEN
    assert conn.search_calls[0] == 'SEEN NOT X-GM-LABELS "URGENT"'
    assert conn.search_calls[1] == "SEEN"
    # Client-side label filter did the real work on the fallback path
    body_ids = set().union(*_body_fetch_ids(conn)) if _body_fetch_ids(conn) else set()
    assert body_ids == {1}


def test_read_sweep_negation_superset_still_filtered_client_side():
    # Server "honors" the command but returns ALL read mail (quirk / missing
    # NOT support): the client-side label filter must still protect labels.
    conn = FakeMailboxConn(
        [
            _msg("3001", True, [], "Unlabeled read"),
            _msg("3002", True, ["URGENT"], "Labeled read"),
        ],
        ignore_negation=True,
    )
    result = _client(conn).fetch_read_unprocessed_emails(known_labels=["URGENT"])
    assert [gid for gid, _m in result] == ["3001"]


def test_read_sweep_skip_ids_exclude_processed_without_body_fetch():
    conn = FakeMailboxConn([
        _msg("4001", True, [], "Processed but label-stripped"),  # in DB → skip
        _msg("4002", True, [], "Unprocessed"),
        _msg("4003", True, [], "Also processed"),                # in DB → skip
    ])
    result = _client(conn).fetch_read_unprocessed_emails(
        known_labels=["URGENT"], skip_ids={"4001", "4003"}
    )

    assert [gid for gid, _m in result] == ["4002"]
    # Skipped mails were scanned in the metadata phase but never body-fetched
    all_fetch_ids = set().union(*[{int(b) for b in ids.split(b",")} for ids, _ in conn.fetch_calls])
    metadata_only = {2, 3}  # seqs seen in phase 1
    body_ids = set().union(*_body_fetch_ids(conn)) if _body_fetch_ids(conn) else set()
    assert metadata_only.issubset(all_fetch_ids)
    assert body_ids == {2}


def test_read_sweep_limit_paces_batch():
    conn = FakeMailboxConn(
        [_msg(str(5000 + i), True, [], f"Old {i}") for i in range(10)]
    )
    result = _client(conn).fetch_read_unprocessed_emails(known_labels=[], limit=3)

    assert len(result) == 3
    # Newest-first pacing: the three newest candidates win
    assert [gid for gid, _m in result] == ["5009", "5008", "5007"]


def test_read_sweep_huge_criteria_falls_back_to_plain_seen():
    # Absurd category counts must not ship a multi-kilobyte SEARCH line.
    with patch.object(imap_client, "MAX_NEGATED_SEARCH_CHARS", 10):
        conn = FakeMailboxConn([_msg("6001", True, [], "R")])
        result = _client(conn).fetch_read_unprocessed_emails(known_labels=["LONG/LABEL"])
    assert conn.search_calls == ["SEEN"]
    assert [gid for gid, _m in result] == ["6001"]


def test_read_sweep_no_dangerous_server_response_is_tolerated():
    # typ != 'OK' (instead of raising) must also trigger the plain-SEEN fallback.
    conn = FakeMailboxConn([_msg("7001", True, [], "R")])
    original_search = conn.search

    def flaky_search(charset, criteria):
        if "NOT X-GM-LABELS" in criteria:
            conn.search_calls.append(criteria)
            return ("NO", [b"mailbox unavailable"])
        return original_search(charset, criteria)  # standalone server appends internally

    conn.search = flaky_search
    result = _client(conn).fetch_read_unprocessed_emails(known_labels=["URGENT"])
    assert conn.search_calls == ['SEEN NOT X-GM-LABELS "URGENT"', "SEEN"]
    assert [gid for gid, _m in result] == ["7001"]


# ---------------------------------------------------------------------------
# imap layer — has_unprocessed_unseen_emails (the hard-gate probe)
# ---------------------------------------------------------------------------

def test_gate_probe_true_when_unprocessed_unread_exists():
    conn = FakeMailboxConn([
        _msg("8001", False, ["URGENT"], "Labeled unread"),
        _msg("8002", False, [], "Unprocessed unread"),
    ])
    client = _client(conn)
    assert client.has_unprocessed_unseen_emails(known_labels=["URGENT"]) is True
    # Metadata-only: never downloads bodies
    assert all("BODY.PEEK" not in q for _, q in conn.fetch_calls)


def test_gate_probe_false_when_read_or_labeled():
    conn = FakeMailboxConn([
        _msg("8101", False, ["URGENT"], "Labeled unread"),
        _msg("8102", True, [], "Read unprocessed (sweep's job, not the gate's)"),
    ])
    assert _client(conn).has_unprocessed_unseen_emails(known_labels=["URGENT"]) is False


def test_gate_probe_empty_inbox_is_drained():
    conn = FakeMailboxConn([])
    assert _client(conn).has_unprocessed_unseen_emails(known_labels=["URGENT"]) is False


def test_gate_probe_error_fails_closed():
    """A probe failure must never open the gate (fail closed)."""
    # One unread message so the probe reaches the metadata scan (the
    # fault-injection point); backoff sleeps are stubbed out.
    client = _client(FakeMailboxConn([_msg("9901", False, [], "Unread")]))
    with patch.object(client, "_scan_qualifying_metadata", side_effect=OSError("boom")), \
         patch("retry.time.sleep"):
        assert client.has_unprocessed_unseen_emails(known_labels=["URGENT"]) is True


# ---------------------------------------------------------------------------
# job layer — hard gating inside classification_job
# ---------------------------------------------------------------------------

def _payload(msg_obj):
    return {
        "sender": "s@t.com", "to": "r@t.com", "cc": "",
        "subject": msg_obj.get("Subject") or "S", "body": "B",
        "mass_mail": False, "attachment_types": [],
    }


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("database.DB_FILE", tmp.name):
            database.init_db()
            yield tmp.name


@pytest.fixture
def sweep_config():
    with patch("config.ENABLE_READ_SWEEP", True), \
         patch("config.READ_SWEEP_BATCH_SIZE", 5), \
         patch("config.truncation_stats"), \
         patch.object(config, "sweep_stats", config._SweepStats()):
        yield config


def _make_classify_mock():
    mock = MagicMock()
    mock.get_available_categories.return_value = ["URGENT", "FOCUS"]
    side = {}

    def extract(_msg):
        payload = {
            "sender": "s@t.com", "to": "r@t.com", "cc": "", "subject": "S",
            "body": "B", "mass_mail": False, "attachment_types": [],
        }
        return side.get("extract", payload)

    def predict(**kwargs):
        return side.get("predict", ("URGENT", 0.95, False))

    mock.extract_email_info.side_effect = extract
    mock.predict_email.side_effect = predict
    mock.sides = side
    return mock


def _run_job(client_mock, classify_mock):
    with patch("imap_client.gmail_client", client_mock), \
         patch("jobs.classification.classify", classify_mock):
        return classification_job(limit=20, trigger="manual")


def test_sweep_gated_while_unprocessed_unread_exists(db, sweep_config):
    """Read work must never run while unread backlog remains — even when the
    unread phase itself finished its own batch (backlog larger than limit)."""
    unread_msg = MagicMock()
    unread_msg.get.return_value = None

    client = MagicMock()
    client.fetch_unprocessed_emails.return_value = [("4001", unread_msg)]
    client.has_unprocessed_unseen_emails.return_value = True  # more backlog left
    client.apply_label.return_value = True

    result = _run_job(client, _make_classify_mock())

    # Unread phase did its job…
    assert {r["id"] for r in result} == {"4001"}
    assert "4001" in database.get_all_log_ids()
    # …and the sweep never started.
    client.fetch_read_unprocessed_emails.assert_not_called()


def test_sweep_runs_when_unread_fully_drained(db, sweep_config):
    read1, read2 = MagicMock(), MagicMock()
    read1.get.return_value = None
    read2.get.return_value = None

    client = MagicMock()
    client.fetch_unprocessed_emails.return_value = []
    client.has_unprocessed_unseen_emails.return_value = False
    client.fetch_read_unprocessed_emails.return_value = [("9001", read1), ("9002", read2)]
    client.apply_label.return_value = True

    result = _run_job(client, _make_classify_mock())

    sweep_call = client.fetch_read_unprocessed_emails.call_args
    assert sweep_call.args[0] == ["URGENT", "FOCUS"]
    assert sweep_call.kwargs["limit"] == 5
    assert sweep_call.kwargs["skip_ids"] == set()
    assert {r["id"] for r in result} == {"9001", "9002"}
    assert database.get_all_log_ids() == {"9001", "9002"}
    assert client.apply_label.call_args_list == [
        (("9001", "URGENT"),), (("9002", "URGENT"),)
    ]


def test_sweep_disabled_by_config_does_nothing(db):
    """Default (conservative) config: sweep off → no read-mail work at all."""
    client = MagicMock()
    client.fetch_unprocessed_emails.return_value = []
    client.has_unprocessed_unseen_emails.return_value = False

    with patch("config.ENABLE_READ_SWEEP", False):
        _run_job(client, _make_classify_mock())

    client.has_unprocessed_unseen_emails.assert_not_called()
    client.fetch_read_unprocessed_emails.assert_not_called()


def test_sweep_paces_by_config_batch_size(db):
    client = MagicMock()
    client.fetch_unprocessed_emails.return_value = []
    client.has_unprocessed_unseen_emails.return_value = False
    client.fetch_read_unprocessed_emails.return_value = []
    client.apply_label.return_value = True

    with patch("config.ENABLE_READ_SWEEP", True), patch("config.READ_SWEEP_BATCH_SIZE", 2):
        _run_job(client, _make_classify_mock())

    assert client.fetch_read_unprocessed_emails.call_args.kwargs["limit"] == 2


def test_sweep_uses_db_processed_marker_as_skip_ids(db):
    """Zero-label read emails that already have a DB row are processed-state;
    they must be handed to the sweep as skips, not re-fetched forever."""
    now = database.datetime.datetime.now()
    database.add_log(
        id="9000", sender="s@t.com", recipient="r@t.com", subject="History",
        predicted_category="URGENT", confidence_score=0.9, timestamp=now,
    )

    client = MagicMock()
    client.fetch_unprocessed_emails.return_value = []
    client.has_unprocessed_unseen_emails.return_value = False
    client.fetch_read_unprocessed_emails.return_value = []
    client.apply_label.return_value = True

    with patch("config.ENABLE_READ_SWEEP", True):
        _run_job(client, _make_classify_mock())

    assert client.fetch_read_unprocessed_emails.call_args.kwargs["skip_ids"] == {"9000"}


def test_sweep_apply_failure_leaves_email_unmarked_and_retryable(db, sweep_config):
    """If the Gmail label never lands, no DB processed-marker is written —
    the email stays eligible for the next cycle instead of being marked
    processed while unlabeled."""
    read_msg = MagicMock()
    read_msg.get.return_value = None

    client = MagicMock()
    client.fetch_unprocessed_emails.return_value = []
    client.has_unprocessed_unseen_emails.return_value = False
    client.fetch_read_unprocessed_emails.return_value = [("9100", read_msg)]
    client.apply_label.return_value = False  # STORE failed after retries

    result = _run_job(client, _make_classify_mock())

    assert client.apply_label.called
    assert "9100" not in database.get_all_log_ids()
    assert result == []


def test_sweep_never_removes_or_rewrites_known_labels(db, sweep_config):
    client = MagicMock()
    client.fetch_unprocessed_emails.return_value = []
    client.has_unprocessed_unseen_emails.return_value = False
    read_msg = MagicMock()
    read_msg.get.return_value = None
    client.fetch_read_unprocessed_emails.return_value = [("9200", read_msg)]
    client.apply_label.return_value = True

    _run_job(client, _make_classify_mock())

    # Corrections are moves owned by the recheck job — the sweep only adds.
    client.remove_label.assert_not_called()
    assert client.apply_label.call_args_list == [(("9200", "URGENT"),)]


def test_sweep_respects_job_cancellation(db, sweep_config):
    client = MagicMock()
    client.fetch_unprocessed_emails.return_value = []
    client.has_unprocessed_unseen_emails.return_value = False
    client.fetch_read_unprocessed_emails.return_value = [("9300", MagicMock())]
    client.apply_label.return_value = True

    from job_queue import job_queue
    job_queue._cancel.set()
    try:
        with patch("imap_client.gmail_client", client), \
             patch("jobs.classification.classify", _make_classify_mock()):
            classification_job(limit=20, trigger="manual")
    finally:
        job_queue._cancel.clear()

    # Cancelled before the sweep loop touched anything.
    client.apply_label.assert_not_called()
    assert database.get_all_log_ids() == set()


# ---------------------------------------------------------------------------
# config defaults + /stats observability
# ---------------------------------------------------------------------------

def test_sweep_config_conservative_defaults():
    assert config.ENABLE_READ_SWEEP is False
    assert config.READ_SWEEP_BATCH_SIZE == 5


def test_stats_surfaces_read_sweep_counters(db, sweep_config):
    config.sweep_stats.reset()

    client = MagicMock()
    client.fetch_unprocessed_emails.return_value = []
    client.has_unprocessed_unseen_emails.return_value = True  # gated attempt

    read_msg = MagicMock()
    read_msg.get.return_value = None
    client2 = MagicMock()
    client2.fetch_unprocessed_emails.return_value = []
    client2.has_unprocessed_unseen_emails.return_value = False
    client2.fetch_read_unprocessed_emails.return_value = [("9400", read_msg)]
    client2.apply_label.return_value = True

    _run_job(client, _make_classify_mock())   # gated
    _run_job(client2, _make_classify_mock())  # runs: 1 candidate, 1 labeled

    from api.routes.health import get_stats
    from api.models import StatsResponse
    data = get_stats()
    assert data["read_sweep_attempts_total"] == 2
    assert data["read_sweep_gated_total"] == 1
    assert data["read_sweep_candidates_total"] == 1
    assert data["read_sweep_labeled_total"] == 1
    # The response model must accept the additive fields (API stays valid).
    StatsResponse(**data)
