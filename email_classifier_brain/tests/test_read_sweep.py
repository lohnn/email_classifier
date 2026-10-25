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
  - config: default-enabled knob (opt-out); /stats observability counters
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
# Fake mailbox connection (search + batched fetch surface used by the sweep)
# ---------------------------------------------------------------------------

class FakeMailboxConn:
    """
    Scripted two-mailbox IMAP account (INBOX + All Mail), the surface the
    sweep and gate touch. Message lists are ordered by sequence number
    (list order = ascending seq = oldest..newest), like real IMAP SEARCH
    output. Supports UNSEEN/SEEN searches, honors NOT X-GM-LABELS clauses,
    and can be told to reject (tagged BAD) or ignore (server quirk →
    superset result) those clauses to exercise the client-side fallbacks.

    select()/search() calls are recorded together in `timeline` (in order)
    so tests can assert that All Mail is selected BEFORE the sweep searches
    and INBOX is restored afterwards.
    """

    ALL_NAME = "[Gmail]/All Mail"

    def __init__(self, messages=None, reject_negation=False, ignore_negation=False,
                 all_mail=None, no_all_mail=False, all_select_fails=False,
                 list_raises=False):
        # `messages` = INBOX view; `all_mail` = account-wide archive view.
        self.stores = {
            "INBOX": list(messages or []),
            "ALL": list(all_mail if all_mail is not None else []),
        }
        self.reject_negation = reject_negation
        self.ignore_negation = ignore_negation
        self.no_all_mail = no_all_mail
        self.all_select_fails = all_select_fails
        self.list_raises = list_raises
        self.mailbox = "INBOX"
        self.search_calls = []
        self.select_calls = []
        self.timeline = []
        self.list_calls = 0
        self.fetch_calls = []

    def noop(self):
        return ("OK", [b""])

    def list(self, *args, **kwargs):
        self.list_calls += 1
        if self.list_raises:
            raise imaplib.IMAP4.error("LIST exploded")
        if self.no_all_mail:
            return ("OK", [b'(\\HasNoChildren \\Trash) "/" "[Gmail]/Trash"'])
        return (
            "OK",
            [
                b'(\\HasNoChildren \\All) "/" "[Gmail]/All Mail"',
                b'(\\HasNoChildren \\Trash) "/" "[Gmail]/Trash"',
            ],
        )

    def select(self, name, *args, **kwargs):
        self.select_calls.append(name)
        self.timeline.append(("select", name))
        if self.all_select_fails and self.ALL_NAME in str(name):
            return ("NO", [b"No such mailbox"])
        self.mailbox = "ALL" if self.ALL_NAME in str(name) else "INBOX"
        return ("OK", [b"42"])

    def _store(self):
        return self.stores["ALL"] if self.mailbox == "ALL" else self.stores["INBOX"]

    @staticmethod
    def _negated_labels(criteria):
        quoted_or_atom = re.findall(r'NOT X-GM-LABELS\s+("[^"]*"|\S+)', criteria)
        return {m.strip('"') for m in quoted_or_atom}

    def search(self, charset, criteria):
        self.search_calls.append(criteria)
        self.timeline.append(("search", criteria))
        if self.reject_negation and "NOT X-GM-LABELS" in criteria:
            raise imaplib.IMAP4.error("BAD Could not parse command")
        neg = set() if self.ignore_negation else self._negated_labels(criteria)
        ids = []
        for seq, msg in enumerate(self._store(), start=1):
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
        store = self._store()
        out = []
        for raw_id in ids_bytes.split(b","):
            seq = int(raw_id)
            if seq < 1 or seq > len(store):
                continue
            msg = store[seq - 1]
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
# imap layer — fetch_read_unprocessed_emails (sweep scope: All Mail)
# ---------------------------------------------------------------------------

def test_read_sweep_fetches_only_unlabeled_read_mail():
    conn = FakeMailboxConn(all_mail=[
        _msg("1001", True, [], "Old unprocessed"),          # sweep candidate
        _msg("1002", True, ["URGENT"], "Labeled by user"),  # labeled read — never touched
        _msg("1003", False, [], "Fresh unread"),            # unread — unread path's job
        _msg("1004", True, [], "Another candidate"),
    ])
    result = _client(conn).fetch_read_unprocessed_emails(known_labels=["URGENT"])

    assert [gid for gid, _msg_obj in result] == ["1004", "1001"]  # newest first
    # Server-side reduction: SEEN + NOT of the known labels
    assert conn.search_calls[-1] == 'SEEN NOT X-GM-LABELS "URGENT"'
    # All Mail was selected BEFORE the search, INBOX restored afterwards
    assert conn.timeline == [
        ("select", FakeMailboxConn.ALL_NAME),
        ("search", 'SEEN NOT X-GM-LABELS "URGENT"'),
        ("select", "INBOX"),
    ]
    # Bodies downloaded only for the two surviving candidates (seqs 1 and 4)
    body_ids = set().union(*_body_fetch_ids(conn)) if _body_fetch_ids(conn) else set()
    assert body_ids == {1, 4}


def test_read_sweep_negation_rejected_falls_back_to_plain_seen():
    conn = FakeMailboxConn(
        all_mail=[
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
    # All Mail was selected before BOTH searches; INBOX restored at the end
    assert conn.timeline == [
        ("select", FakeMailboxConn.ALL_NAME),
        ("search", 'SEEN NOT X-GM-LABELS "URGENT"'),
        ("search", "SEEN"),
        ("select", "INBOX"),
    ]
    # Client-side label filter did the real work on the fallback path
    body_ids = set().union(*_body_fetch_ids(conn)) if _body_fetch_ids(conn) else set()
    assert body_ids == {1}


def test_read_sweep_negation_superset_still_filtered_client_side():
    # Server "honors" the command but returns ALL read mail (quirk / missing
    # NOT support): the client-side label filter must still protect labels.
    conn = FakeMailboxConn(
        all_mail=[
            _msg("3001", True, [], "Unlabeled read"),
            _msg("3002", True, ["URGENT"], "Labeled read"),
        ],
        ignore_negation=True,
    )
    result = _client(conn).fetch_read_unprocessed_emails(known_labels=["URGENT"])
    assert [gid for gid, _m in result] == ["3001"]


def test_read_sweep_skip_ids_exclude_processed_without_body_fetch():
    conn = FakeMailboxConn(all_mail=[
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
        all_mail=[_msg(str(5000 + i), True, [], f"Old {i}") for i in range(10)]
    )
    result = _client(conn).fetch_read_unprocessed_emails(known_labels=[], limit=3)

    assert len(result) == 3
    # Newest-first pacing: the three newest candidates win
    assert [gid for gid, _m in result] == ["5009", "5008", "5007"]


def test_read_sweep_huge_criteria_falls_back_to_plain_seen():
    # Absurd category counts must not ship a multi-kilobyte SEARCH line.
    with patch.object(imap_client, "MAX_NEGATED_SEARCH_CHARS", 10):
        conn = FakeMailboxConn(all_mail=[_msg("6001", True, [], "R")])
        result = _client(conn).fetch_read_unprocessed_emails(known_labels=["LONG/LABEL"])
    assert conn.search_calls == ["SEEN"]
    assert [gid for gid, _m in result] == ["6001"]


def test_read_sweep_no_dangerous_server_response_is_tolerated():
    # typ != 'OK' (instead of raising) must also trigger the plain-SEEN fallback.
    conn = FakeMailboxConn(all_mail=[_msg("7001", True, [], "R")])
    original_search = conn.search

    def flaky_search(charset, criteria):
        if "NOT X-GM-LABELS" in criteria:
            conn.search_calls.append(criteria)
            conn.timeline.append(("search", criteria))
            return ("NO", [b"mailbox unavailable"])
        return original_search(charset, criteria)  # standalone server appends internally

    conn.search = flaky_search
    result = _client(conn).fetch_read_unprocessed_emails(known_labels=["URGENT"])
    assert conn.search_calls == ['SEEN NOT X-GM-LABELS "URGENT"', "SEEN"]
    assert [gid for gid, _m in result] == ["7001"]
    # Selection still bracketed the whole cycle, including the fallback path
    assert conn.timeline[0] == ("select", FakeMailboxConn.ALL_NAME)
    assert conn.timeline[-1] == ("select", "INBOX")


def test_read_sweep_all_mail_discovered_via_list_special_use():
    """Localized Gmail: the \\All special-use attribute resolves the mailbox
    name (not the literal); the sweep searches that selection."""
    localized_all = "[Gmail]/Todos"
    conn = FakeMailboxConn(all_mail=[_msg("801", True, [], "R")])

    def localized_list(*args, **kwargs):
        conn.list_calls += 1
        return ("OK", [
            f'(\\HasNoChildren \\All) "/" "{localized_all}"'.encode(),
            b'(\\HasNoChildren \\Trash) "/" "[Gmail]/Trash"',
        ])

    conn.list = localized_list
    # The stock select() only maps names containing "All Mail" to the ALL
    # store; make the localized name resolve too.
    original_select = conn.select

    def select_any(name, *args, **kwargs):
        if name == localized_all:
            conn.select_calls.append(name)
            conn.timeline.append(("select", name))
            conn.mailbox = "ALL"
            return ("OK", [b"42"])
        return original_select(name, *args, **kwargs)

    conn.select = select_any

    client = _client(conn)
    result = client.fetch_read_unprocessed_emails(known_labels=["URGENT"])
    assert [gid for gid, _m in result] == ["801"]
    assert client._all_mail_name == localized_all


def test_read_sweep_all_mail_name_cached_across_cycles():
    """The resolved name is cached per client: the second sweep must not
    re-run mailbox discovery (LIST)."""
    conn = FakeMailboxConn(all_mail=[
        _msg("9001", True, [], "First batch"),
        _msg("9002", True, [], "Second batch"),
    ])
    client = _client(conn)
    first = client.fetch_read_unprocessed_emails(known_labels=["U"], limit=1)
    list_calls_after_first = conn.list_calls
    second = client.fetch_read_unprocessed_emails(known_labels=["U"], limit=1)

    # Same visible candidates (the sweep consumes no mailbox state); the
    # point is that the second cycle re-uses the cached resolution.
    assert [gid for gid, _m in first] == ["9002"]
    assert [gid for gid, _m in second] == ["9002"]
    assert conn.list_calls == list_calls_after_first == 1
    # Both cycles still selected All Mail and restored INBOX
    assert conn.select_calls == [
        FakeMailboxConn.ALL_NAME, "INBOX",
        FakeMailboxConn.ALL_NAME, "INBOX",
    ]


def test_read_sweep_falls_back_to_literal_all_mail_when_list_fails():
    """No LIST discovery (server quirk / transient failure): the conventional
    Gmail name is tried, trusted only once its SELECT succeeds, then cached."""
    conn = FakeMailboxConn(all_mail=[_msg("9101", True, [], "R")], list_raises=True)
    client = _client(conn)

    result = client.fetch_read_unprocessed_emails(known_labels=["URGENT"])

    assert [gid for gid, _m in result] == ["9101"]
    assert client._all_mail_name == FakeMailboxConn.ALL_NAME
    assert conn.timeline == [
        ("select", FakeMailboxConn.ALL_NAME),
        ("search", 'SEEN NOT X-GM-LABELS "URGENT"'),
        ("select", "INBOX"),
    ]


def test_read_sweep_no_all_mail_stays_on_inbox():
    """Final fallback: non-Gmail-style server (no \\All in LIST, literal name
    unselectable) → warn and sweep the selected INBOX instead of breaking."""
    conn = FakeMailboxConn(
        messages=[_msg("9201", True, [], "Inbox read, never labeled")],
        no_all_mail=True,
        all_select_fails=True,
    )
    client = _client(conn)

    result = client.fetch_read_unprocessed_emails(known_labels=["URGENT"])

    # Sweep ran INBOX-scoped: the INBOX candidate was found and returned
    assert [gid for gid, _m in result] == ["9201"]
    assert client._all_mail_name is None
    # Attempted the literal fallback name, never selected All Mail; restored INBOX
    assert conn.select_calls == [FakeMailboxConn.ALL_NAME, "INBOX"]
    assert conn.timeline == [
        ("select", FakeMailboxConn.ALL_NAME),
        ("search", 'SEEN NOT X-GM-LABELS "URGENT"'),
        ("select", "INBOX"),
    ]


def test_read_sweep_broken_inbox_restore_drops_connection():
    """If restoring INBOX fails after the sweep, the cached connection is
    dropped so the next call reconnects fresh (never sweeps the wrong
    mailbox with a stale selection)."""
    conn = FakeMailboxConn(all_mail=[_msg("9301", True, [], "R")])
    client = _client(conn)
    original_select = conn.select

    def select_fails_on_inbox(name, *args, **kwargs):
        if name == "INBOX":
            conn.select_calls.append(name)
            conn.timeline.append(("select", name))
            raise imaplib.IMAP4.error("connection dead")
        return original_select(name, *args, **kwargs)

    conn.select = select_fails_on_inbox
    client.fetch_read_unprocessed_emails(known_labels=["URGENT"])
    assert client.connection is None


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


def test_gate_probe_is_inbox_scoped_even_when_archive_has_unread():
    """Priority rule stays bound to INBOX: archived unseen mail (invisible to
    the UNSEEN phase) must neither block the gate nor trigger a mailbox
    switch — only unseen mail in the INBOX selection governs."""
    conn = FakeMailboxConn(
        messages=[_msg("8501", False, ["URGENT"], "Labeled unread in INBOX")],
        all_mail=[_msg("8502", False, [], "Archived UNSEEN — nobody's job yet")],
    )
    client = _client(conn)
    assert client.has_unprocessed_unseen_emails(known_labels=["URGENT"]) is False
    # The probe never left INBOX: no mailbox switches at all
    assert conn.select_calls == []


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
    """The opt-out knob: ENABLE_READ_SWEEP=false → no read-mail work at all,
    even with the unread queue fully drained."""
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

def test_sweep_config_default_enabled():
    """WI-103 follow-up: the sweep ships ENABLED by default (gradual pacing
    unchanged); disabling is the explicit opt-out."""
    assert config.ENABLE_READ_SWEEP is True
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
