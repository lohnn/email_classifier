
"""Gmail label-write reliability wave — root cause + fixes for the
2026-10 live bug (DB said corrected, Gmail kept the old label; VERIFIED
applied fine in the same minute).

Mechanisms that could produce that evidence, all closed here:
  M1 search-miss  -> used to silently return (with_retry never retried a
                     clean `return`; only exceptions).
  M2 STORE error  -> retried 3x then swallowed by logger.error.
  M3 STORE no-op  -> server said OK but the label survived; undetectable.
Now: removal is verified server-side (FETCH X-GM-LABELS after STORE),
uid resolution reaches All Mail/Trash (archived mail stays manageable),
and cleanup failures surface into update_history.json.
"""

import json
import os

import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

temp_data_dir = tempfile.mkdtemp()
os.environ["TRAINING_DATA_DIR"] = temp_data_dir
os.environ["ADMIN_API_KEY"] = "testkey"
os.environ["TESTING"] = "true"

import main  # noqa: F401
from main import app
import database
import imap_client
from imap_client import X_GM_MSGID_PATTERN


class LabelFakeServer:
    """
    Fake IMAP backend for label-write tests. Labels live per mailbox;
    slash-labels supported (the deployment's labels are 'Jarvis/NOISE',
    'Jarvis/URGENT'). store_ignores_removal models the M3 failure mode:
    Gmail returns OK but the label survives.
    """

    def __init__(self, mailboxes=None, store_ignores_removal=False):
        self.stores = {"INBOX": {}, "ALL": {}, "TRASH": {}}
        if mailboxes:
            self.stores.update(mailboxes)
        self.mailbox = "INBOX"
        self.select_calls = []
        self.store_calls = []
        self.calls = []
        self.store_ignores_removal = store_ignores_removal

    def noop(self):
        return ("OK", [b""])

    def list(self, *a, **k):
        return ("OK", [
            b'(\\HasNoChildren \\All) "/" "[Gmail]/All Mail"',
            b'(\\HasNoChildren \\Trash) "/" "[Gmail]/Trash"',
        ])

    def select(self, mailbox, *a, **k):
        self.mailbox = str(mailbox)
        self.select_calls.append(str(mailbox))
        return ("OK", [b""])

    def _current(self):
        if self.mailbox == "INBOX":
            return self.stores["INBOX"]
        if "Trash" in self.mailbox:
            return self.stores["TRASH"]
        return self.stores["ALL"]

    def uid(self, command, *args):
        self.calls.append((command,) + tuple(str(x) for x in args))
        if command == "SEARCH":
            m = X_GM_MSGID_PATTERN.search(str(args[1]))
            gid = int(m.group(1)) if m else None
            store = self._current()
            if gid is not None and gid in store:
                return ("OK", [str(gid).encode()])
            return ("OK", [b""])
        if command == "STORE":
            uid = bytes(args[0]).decode()
            gid = int(uid)
            mode = str(args[1])           # '+X-GM-LABELS' / '-X-GM-LABELS'
            body = str(args[2]).strip("()").strip('"')   # one label per call
            store = self._current()
            current = list(store.get(gid, []))
            if not self.store_ignores_removal:
                if mode.startswith("-") and body in current:
                    current.remove(body)
                elif mode.startswith("+") and body not in current:
                    current.append(body)
                store[gid] = current
            return ("OK", [f"{uid} (X-GM-LABELS ({' '.join(current)}))".encode()])
        if command == "FETCH" and "X-GM-LABELS" in str(args[1]):
            uids = bytes(args[0]).split(b",")
            out = []
            for uid_bytes in uids:
                gid = int(uid_bytes.decode())
                current = self._current().get(gid, [])
                out.append((f"1 (X-GM-MSGID {gid} X-GM-LABELS ({' '.join(current)}))".encode(), b""))
            return ("OK", out)
        raise AssertionError(f"unexpected uid command {command} {args}")


def _client(server):
    client = imap_client.GmailClient()
    client.user = "test@example.com"
    client.password = "not-a-real-password"
    client.connection = server
    return client


def test_slash_label_removed_and_verified_server_side():
    server = LabelFakeServer({"INBOX": {7: ["Jarvis/NOISE", "Jarvis/URGENT"]}})
    assert _client(server).remove_label("7", "Jarvis/NOISE") is True
    assert server.stores["INBOX"][7] == ["Jarvis/URGENT"]           # effect is real
    assert any(a[2] == "-X-GM-LABELS" for a in server.calls)        # minus variant used
    assert any(a[0] == "FETCH" for a in server.calls)               # effect verified


def test_store_no_effect_detected_return_false():
    server = LabelFakeServer({"INBOX": {7: ["Jarvis/NOISE"]}}, store_ignores_removal=True)
    with patch.object(imap_client, "with_retry", lambda fn, **k: fn()):
        ok = _client(server).remove_label("7", "Jarvis/NOISE")
    assert ok is False                                              # M3 is now visible
    assert server.stores["INBOX"][7] == ["Jarvis/NOISE"]            # server state untouched


def test_search_miss_everywhere_returns_false_not_silence():
    server = LabelFakeServer()  # message gone
    assert _client(server).remove_label("999", "Jarvis/NOISE") is False   # M1 surfaced
    assert _client(server).apply_label("999", "Jarvis/URGENT") is False


def test_archived_message_uid_resolution_and_removal():
    server = LabelFakeServer({"ALL": {7: ["Jarvis/NOISE", "Jarvis/URGENT"]}})
    assert _client(server).remove_label("7", "Jarvis/NOISE") is True
    assert server.stores["ALL"][7] == ["Jarvis/URGENT"]
    assert any("All Mail" in m for m in server.select_calls)        # fallback used
    assert server.select_calls[-1] == "INBOX"                       # session restored


# ---------------------------------------------------------------------------
# Caller surfacing: cleanup failure after a successful correction lands in
# update_history.json (the human-readable file), both recheck loops.
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        hist_path = tmp.name + ".history"
        with patch("database.DB_FILE", tmp.name), \
             patch("jobs.training_data.TRAINING_DATA_DIR", temp_data_dir), \
             patch("jobs.update.HISTORY_FILE", hist_path):
            database.init_db()
            with TestClient(app) as c:
                yield c


def _seed_log(gid, predicted="NOISE"):
    database.add_log(
        id=gid, sender="s@x.com", recipient="r@x.com", subject=f"S {gid}",
        predicted_category=predicted, confidence_score=0.9,
        body=f"B {gid}", cc="", mass_mail=False, attachment_types=[],
    )


def _read_history():
    import jobs.update as upd
    if not os.path.exists(upd.HISTORY_FILE):
        return []
    with open(upd.HISTORY_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def mock_classify():
    mock = MagicMock()
    with patch("jobs.correction.classify", mock), \
         patch("api.routes.admin.classify", mock):
        yield mock


def test_recheck_surfaces_cleanup_failure_after_moving_training_data(
    client, mock_classify
):
    """The user's live case as an integration test: correction resolves,
    training data moves, VERIFIED applies — and the failed old-label removal
    is no longer silent: update_history.json records it."""
    label_rows = {"g1": ["NOISE", "URGENT"]}
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT", "FOCUS", "REFERENCE"]
    _seed_log("g1")
    mock_imap = MagicMock()
    mock_imap.get_labels_for_emails.side_effect = lambda ids: {
        g: label_rows.get(g, []) for g in ids
    }
    mock_imap.remove_label.return_value = False   # the live failure
    mock_imap.apply_label.return_value = True

    with patch("imap_client.gmail_client", mock_imap):
        from jobs.correction import check_corrections_job
        check_corrections_job(limit=10, trigger="manual")

    # Training data MOVED, DB corrected, VERIFIED applied — as before
    with open(os.path.join(temp_data_dir, "URGENT.jsonl"), encoding="utf-8") as f:
        assert [json.loads(l)["subject"] for l in f if l.strip()] == ["S g1"]
    assert database.get_log_by_id("g1")["corrected_category"] == "URGENT"
    assert mock_imap.apply_label.call_count >= 1

    # And the drift is now OBSERVABLE in the human-readable file
    warnings = [e for e in _read_history()
                if e["status"] == "warning" and "Gmail cleanup incomplete" in e["message"]]
    assert warnings and "g1" in warnings[0]["message"]


def test_verified_confirm_surfaces_unsure_strip_failure(client, mock_classify):
    label_rows = {"g1": ["URGENT", "VERIFIED"]}
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT", "FOCUS", "REFERENCE"]
    _seed_log("g1", predicted="URGENT")
    mock_imap = MagicMock()
    mock_imap.get_labels_for_emails.side_effect = lambda ids: {
        g: label_rows.get(g, []) for g in ids
    }
    mock_imap.remove_label.return_value = False

    with patch("imap_client.gmail_client", mock_imap):
        from jobs.correction import check_corrections_job
        check_corrections_job(limit=10, trigger="manual")

    warnings = [e for e in _read_history()
                if e["status"] == "warning" and "Gmail cleanup incomplete" in e["message"]]
    assert warnings


def test_repair_endpoint_syncs_labels_from_db(client, mock_classify):
    """User-triggered remediation surface for rows fixed in the DB while
    Gmail kept the old label (the standing specialkost-style case). Uses
    the deployment's real label shape ('Jarvis/...' path-derived)."""
    mock_classify.get_available_categories.return_value = [
        "Jarvis/NOISE", "Jarvis/URGENT", "Jarvis/FOCUS", "Jarvis/REFERENCE"
    ]
    _seed_log("101", predicted="Jarvis/NOISE")
    database.update_log_correction("101", "Jarvis/URGENT")

    server = LabelFakeServer({"INBOX": {101: ["Jarvis/NOISE", "Jarvis/URGENT",
                                             "UNSURE_CLASSIFICATION"]}})
    fake = _client(server)
    with patch("imap_client.GmailClient", return_value=fake):
        response = client.post(
            "/logs/101/repair-gmail-labels", headers={"X-API-Key": "testkey"}
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["expected_category"] == "Jarvis/URGENT"
    action_map = {(a["action"], a["label"]): a["ok"] for a in payload["actions"]}
    assert ("remove", "Jarvis/NOISE") in action_map
    assert ("remove_flag", "UNSURE_CLASSIFICATION") in action_map
    assert action_map[("remove", "Jarvis/NOISE")] is True
    # Jarvis/URGENT already on the message; VERIFIED absent → applied
    assert ("apply", "VERIFIED") in action_map
    assert all(ok for ok in action_map.values())
    assert server.stores["INBOX"][101] == ["Jarvis/URGENT", "VERIFIED"]


def test_repair_endpoint_404_for_unknown_row(client):
    response = client.post("/logs/unknown-id/repair-gmail-labels",
                           headers={"X-API-Key": "testkey"})
    assert response.status_code == 404
