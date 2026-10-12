
"""Wave G — out-of-inbox label supplement (get_labels_for_emails).

Gmail ids the INBOX pass does not find (archived, or trashed but not
purged) are retried with the same bounded X-GM-MSGID OR-search in
[Gmail]/All Mail and, if still missing, [Gmail]/Trash. The session mailbox
must always end back on INBOX, batches stay <=50 ids, and every failure
degrades to today's silent-skip parity. Spam is deliberately not searched.
"""

import os
import re
import sys
import tempfile
from unittest.mock import MagicMock

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

os.environ["TRAINING_DATA_DIR"] = tempfile.mkdtemp()
os.environ["ADMIN_API_KEY"] = "testkey"
os.environ["TESTING"] = "true"

import main  # noqa: F401
from imap_client import GmailClient, X_GM_MSGID_PATTERN


GID_PATTERN = re.compile(r'X-GM-MSGID (\d+)')


class FakeGmailServer:
    """
    Scripted IMAP backend for a single GmailClient.

    Labels are known per mailbox: a message labelled in Gmail and then
    archived leaves INBOX but keeps its labels in All Mail; purged postData
    lives in Trash until the 30-day purge.
    """

    INBOX_BATCH = 50

    def __init__(self, inbox=None, all_mail=None, trash=None, list_fails=False):
        self.stores = {
            "INBOX": inbox or {},
            "ALL": all_mail or {},
            "TRASH": trash or {},
        }
        self.mailbox = "INBOX"
        self.select_calls = []
        self.search_calls = []
        self.uid_counter = iter(range(1, 100000))
        self.list_fails = list_fails

    # -- imaplib surface the client touches --------------------------------
    def noop(self):
        return ("OK", [b""])

    def list(self, *args, **kwargs):
        if self.list_fails:
            raise OSError("LIST exploded")
        return (
            "OK",
            [
                b'(\\HasNoChildren \\All) "/" "[Gmail]/All Mail"',
                b'(\\HasNoChildren \\Trash) "/" "[Gmail]/Trash"',
            ],
        )

    def select(self, mailbox, *args, **kwargs):
        self.mailbox = str(mailbox)
        self.select_calls.append(self.mailbox)
        return ("OK", [b"42"])

    def uid(self, command, *args):
        if command == 'SEARCH':
            criteria = args[1]
            self.search_calls.append((self.mailbox, criteria))
            ids = {int(m) for m in GID_PATTERN.findall(str(criteria))}
            store = self._current_store()
            hits = [gid for gid in ids if gid in store]
            uids = []
            for gid in hits:
                uid = next(self.uid_counter)
                uids.append((uid, gid))
            # stash for the FETCH that follows (like a server session would)
            self._uidmap = {str(uid): gid for uid, gid in uids}
            return ("OK", [b" ".join(str(uid).encode() for uid, _ in uids)])
        if command == 'FETCH':
            uid_str = args[0]
            out = []
            for uid_bytes in uid_str.split(b","):
                gid = self._uidmap.get(uid_bytes.decode())
                if gid is None:
                    continue
                labels = " ".join(self._current_store().get(int(gid), []))
                out.append((f"1 (X-GM-MSGID {gid} X-GM-LABELS ({labels}))".encode(), b""))
            return ("OK", out)
        raise AssertionError(f"unexpected uid command {command}")

    # -- helpers ------------------------------------------------------------
    def _current_store(self):
        if self.mailbox == "INBOX":
            return self.stores["INBOX"]
        if "Trash" in self.mailbox:
            return self.stores["TRASH"]
        return self.stores["ALL"]


def _client(server):
    """A GmailClient bound to the scripted server (bypasses credentials)."""
    client = GmailClient()
    client.user = "test@example.com"
    client.password = "not-a-real-password"
    client.connection = server
    return client


def _labels_for(store, labels):
    return {gid: list(labels) for gid in store}


def test_supplement_resolves_archived_message():
    server = FakeGmailServer(all_mail={123: ["FOCUS", "VERIFIED"]})
    assert _client(server).get_labels_for_emails(["123"]) == {"123": ["FOCUS", "VERIFIED"]}
    # The supplement searched All Mail and the session ended back on INBOX
    assert any("All Mail" in m for m in server.select_calls)
    assert server.select_calls[-1] == "INBOX"


def test_inbox_pass_untouched_when_everything_found():
    server = FakeGmailServer(inbox={123: ["NOISE"]})
    result = _client(server).get_labels_for_emails(["123"])
    assert result == {"123": ["NOISE"]}
    # No supplement at all — not even the final INBOX restore runs
    assert server.select_calls == []


def test_supplement_batches_bounded_at_50():
    gids = [str(i) for i in range(30000, 30120)]  # 120 ids
    # None of them exist anywhere — full pipeline still exercised
    server = FakeGmailServer()
    assert _client(server).get_labels_for_emails(gids) == {}
    inbox_searches = [criteria for (mbox, criteria) in server.search_calls if mbox == "INBOX"]
    supplement_searches = [criteria for (mbox, criteria) in server.search_calls if mbox != "INBOX"]
    assert len(inbox_searches) == 3          # ceil(120/50)
    assert len(supplement_searches) == 6     # 3 All Mail + 3 Trash (none found anywhere)
    # Batch bound: never more than 50 ids in any single SEARCH
    assert all(len(GID_PATTERN.findall(criteria)) <= 50 for (_, criteria) in server.search_calls)


def test_trash_fallback_for_still_missing():
    server = FakeGmailServer(
        all_mail={111: ["FOCUS"]},
        trash={222: ["URGENT"]},
    )
    result = _client(server).get_labels_for_emails(["111", "222"])
    assert result == {"111": ["FOCUS"], "222": ["URGENT"]}
    assert any("Trash" in m for m in server.select_calls)
    assert server.select_calls[-1] == "INBOX"


def test_list_failure_degrades_to_silent_skip():
    server = FakeGmailServer(list_fails=True)
    result = _client(server).get_labels_for_emails(["123"])
    assert result == {}          # same observable as the pre-supplement skip
    # The session mailbox was never switched
    assert "All Mail" not in {m for m in server.select_calls}


def test_gone_from_everywhere_is_absent_not_error():
    server = FakeGmailServer()
    assert _client(server).get_labels_for_emails(["999"]) == {}
    assert server.select_calls[-1] == "INBOX"


def test_zero_labels_entry_counts_as_seen():
    """A message stripped of all labels stops being retried in later mailboxes."""
    server = FakeGmailServer(all_mail={555: []})
    result = _client(server).get_labels_for_emails(["555"])
    assert result == {"555": []}
    # Trash pass skipped — 555 was already resolved
    assert not any("Trash" in m and m != "INBOX" for m in server.select_calls[:-1])
