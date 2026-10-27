import imaplib
import logging
import email
import os
import ssl
import re
from email.message import Message
from typing import List, Optional, Tuple, Dict
from dotenv import load_dotenv

from retry import with_retry

load_dotenv()

IMAP_SERVER = os.getenv("IMAP_SERVER") or "imap.gmail.com"
logger = logging.getLogger(__name__)

# Regex to match X-GM-LABELS content.
# Handles atoms (no quotes/parens) and quoted strings (with escaped quotes/backslashes).
# Group 1 captures the content inside the parentheses.
X_GM_LABELS_PATTERN = re.compile(r'X-GM-LABELS \(((?:[^()"]+|"(\\.|[^"\\])*")*)\)')
X_GM_MSGID_PATTERN = re.compile(r'X-GM-MSGID (\d+)')
SEQ_PATTERN = re.compile(r'^(\d+)')
# Regex to parse individual labels from the list (handling quotes)
LABEL_TOKEN_PATTERN = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"|([^"\s()]+)')

# Transient errors that warrant a retry (network drops, protocol aborts)
_IMAP_ERRORS = (imaplib.IMAP4.error, OSError)

# Size cap for the read sweep's composed `SEEN NOT X-GM-LABELS ...` SEARCH
# criteria (WI-103). With dozens of trained categories the line stays well
# under this; crossing it means an absurd label vocabulary and we fall back
# to a plain SEEN scan with client-side filtering instead of shipping a
# giant command the server may reject anyway.
MAX_NEGATED_SEARCH_CHARS = 4000

# Conventional Gmail All Mail names — the read sweep's literal fallbacks
# when LIST cannot discover an RFC 6154 \All special-use mailbox. Each is
# trusted only after a successful SELECT; a server where none resolves
# (non-Gmail, or unexpected naming) means the sweep warns (with the raw
# LIST sample captured for /admin/sweep-state) and runs INBOX-scoped for
# that cycle. Live Gmail quirk (WI-103 follow-up 4): accounts served under
# the googlemail.com domain expose '[Google Mail]/...' prefixes instead of
# '[Gmail]/...'.
ALL_MAIL_FALLBACK_NAME = "[Gmail]/All Mail"
ALL_MAIL_FALLBACK_NAMES = (
    "[Gmail]/All Mail",
    "[Google Mail]/All Mail",
)

# A LIST untagged line's mailbox name: the final quoted token
# ("(\flags) "delimiter" "name"). Selectively matched by the read sweep's
# All Mail discovery (_select_all_mail).
_LIST_QUOTED_NAME_PATTERN = re.compile(r'"([^"]*)"\s*$')


def _imap_quote_mailbox(name: str) -> str:
    """Idempotent IMAP mailbox-name quoting for SELECT.

    imaplib does NOT quote SELECT arguments: a mailbox name containing IMAP
    specials ([, ], the hierarchy character, spaces) must be sent as a
    quoted string, else the server answers 'BAD Could not parse command'.
    Live proof (WI-103 follow-up 7): Gmail's real 'SELECT [Gmail]/All Mail'
    — bare — is rejected exactly that way, while the quoted form works.

    Idempotent: one surrounding quote pair is stripped if present, then the
    name is re-quoted exactly once. Only All Mail limbs flow through here —
    bare-atom INBOX selects elsewhere in this file stay untouched."""
    if not name:
        return name
    stripped = name
    if len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"'):
        stripped = stripped[1:-1]
    return f'"{stripped}"'


def _imap_quote_label(label: str) -> str:
    """Return label properly quoted and escaped for use in IMAP commands.

    Escapes backslashes and double quotes so that a label value cannot
    inject additional IMAP tokens or commands.
    """
    escaped = label.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


class GmailClient:
    def __init__(self):
        # Prefer IMAP_USER, fallback to first email in MY_EMAIL
        self.user = os.getenv("IMAP_USER") or (os.getenv("MY_EMAIL") or "").split(",")[0].strip()
        self.password = os.getenv("IMAP_PASSWORD")
        self.connection = None
        # Resolved All Mail mailbox name for the read sweep (WI-103); None
        # until a LIST-\All or a verified conventional-name SELECT resolves it.
        self._all_mail_name: Optional[str] = None
        # How the last sweep candidate search ran (diagnostics for
        # GET /admin/sweep-state): "negated" = server-side SEEN NOT
        # X-GM-LABELS worked; "plain_seen" = fallback scan; None = no sweep
        # fetch ran in this process yet.
        self.last_sweep_search_mode: Optional[str] = None
        # All Mail discovery diagnostics (WI-103 follow-up 4): on discovery
        # failure they carry the raw LIST response sample and the exact
        # per-limb SELECT results so live-vs-mock deltas self-report via
        # GET /admin/sweep-state. Both cleared on successful resolution.
        self._all_mail_discovery_error: Optional[str] = None
        self._all_mail_list_sample: Optional[str] = None

        if not self.user or not self.password:
            # We don't raise error immediately to allow importing the module,
            # but connect will fail.
            pass

    def connect(self):
        if not self.user or not self.password:
             raise ValueError("IMAP_USER (or MY_EMAIL) and IMAP_PASSWORD must be set in .env")

        if self.connection:
            try:
                self.connection.noop()
                return
            except:
                self.connection = None

        context = ssl.create_default_context()
        self.connection = imaplib.IMAP4_SSL(IMAP_SERVER, ssl_context=context)
        self.connection.login(self.user, self.password)
        self.connection.select("INBOX")

    def disconnect(self):
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            try:
                self.connection.logout()
            except:
                pass
            self.connection = None

    def _reset_connection(self):
        """Drop the cached connection so the next call triggers a fresh connect."""
        self.connection = None

    def fetch_unprocessed_emails(self, known_labels: List[str], limit: Optional[int] = None) -> List[Tuple[str, Message]]:
        """
        Fetch UNSEEN emails that do not have any of the known_labels.
        Returns a list of (gmail_id, email_message_object), newest first.
        If limit is set, stops scanning once that many qualifying emails are found.

        Two-phase approach:
          Phase 1 – metadata-only scan (X-GM-LABELS + X-GM-MSGID, no bodies),
                    newest-first, collecting qualifying sequence IDs up to limit.
          Phase 2 – fetch BODY.PEEK[] only for those qualifying IDs.
        """
        def _do():
            self.connect()

            logger.info('Just about to search for unsees emails')
            # Search for UNSEEN emails
            typ, data = self.connection.search(None, 'UNSEEN')

            if typ != 'OK' or not data[0]:
                return []

            email_ids = data[0].split()[::-1]  # Reverse so newest emails (highest IDs) are processed first
            logger.info(f'Finished search for unseen emails, found {len(email_ids)} emails to classify')

            if not email_ids:
                return []

            # NOTE: IMAP FETCH returns results in ascending sequence-number order
            # regardless of the order requested. The shared metadata scan walks
            # `email_ids` (which *are* in newest-first order) to decide
            # qualification order.
            qualifying_seq_ids = self._scan_qualifying_metadata(email_ids, set(known_labels), limit=limit)

            if not qualifying_seq_ids:
                return []

            # Phase 2: fetch full bodies only for the qualifying emails.
            return self._fetch_bodies(qualifying_seq_ids)

        return with_retry(
            _do,
            retries=3,
            backoff=2.0,
            exceptions=_IMAP_ERRORS,
            on_retry=lambda exc, _: self._reset_connection(),
        )

    def _scan_qualifying_metadata(
        self,
        email_ids: List[bytes],
        known_labels_set: set,
        limit: Optional[int] = None,
        skip_ids: Optional[set] = None,
    ) -> List[Tuple[bytes, str]]:
        """
        Phase-1 metadata scan shared by the unread fetch and the read-mail
        sweep (WI-103): batched X-GM-LABELS + X-GM-MSGID fetches, no bodies.

        Walks `email_ids` newest-first and returns [(seq_id, gmail_id)] for
        emails that carry NONE of `known_labels_set` and are NOT in
        `skip_ids` (when given). Early-stops once `limit` qualifying emails
        are collected.

        skip_ids is the processed-marker guard (e.g. gmail ids that already
        have a DB log row): a skipped email neither qualifies nor consumes
        the `limit` budget — otherwise a wall of already-processed mail at
        the head of the scan would stall the sweep.
        """
        try:
            BATCH_SIZE = int(os.getenv("IMAP_BATCH_SIZE") or "50")
        except ValueError:
            BATCH_SIZE = 50

        qualifying: List[Tuple[bytes, str]] = []

        for i in range(0, len(email_ids), BATCH_SIZE):
            batch_ids = email_ids[i:i + BATCH_SIZE]
            ids_str = b','.join(batch_ids)

            typ, msg_data = self.connection.fetch(ids_str, '(X-GM-LABELS X-GM-MSGID)')
            if typ != 'OK':
                continue

            # Build a lookup: seq_id (str) -> (labels_str, gmail_id)
            metadata_by_seq: Dict[str, Tuple[str, str]] = {}
            for response_part in msg_data:
                raw_line = response_part[0] if isinstance(response_part, tuple) else response_part
                metadata = raw_line.decode('utf-8', errors='ignore')

                seq_match = SEQ_PATTERN.match(metadata)
                if not seq_match:
                    continue
                seq_id_str = seq_match.group(1)

                labels_str = ""
                lbl_match = X_GM_LABELS_PATTERN.search(metadata)
                if lbl_match:
                    labels_str = lbl_match.group(1)

                msgid_match = X_GM_MSGID_PATTERN.search(metadata)
                gmail_id = msgid_match.group(1) if msgid_match else ""

                metadata_by_seq[seq_id_str] = (labels_str, gmail_id)

            # Walk batch_ids in newest-first order to preserve ordering
            for bid in batch_ids:
                bid_str = bid.decode('utf-8', errors='ignore')
                if bid_str not in metadata_by_seq:
                    continue

                labels_str, gmail_id = metadata_by_seq[bid_str]

                # Skip already-processed ids (processed-marker) BEFORE the
                # label filter: they neither qualify nor consume the limit.
                if skip_ids and gmail_id and gmail_id in skip_ids:
                    continue

                # Skip if any known label is already applied
                skip = False
                for m in LABEL_TOKEN_PATTERN.finditer(labels_str):
                    label_found = m.group(1).replace('\\"', '"').replace('\\\\', '\\') if m.group(1) else m.group(2)
                    if label_found in known_labels_set:
                        skip = True
                        break

                if not skip:
                    qualifying.append((bid, gmail_id))
                    if limit is not None and len(qualifying) >= limit:
                        break  # got enough from this batch

            if limit is not None and len(qualifying) >= limit:
                break  # outer loop – got enough

        return qualifying

    def _fetch_bodies(self, qualifying_seq_ids: List[Tuple[bytes, str]]) -> List[Tuple[str, Message]]:
        """
        Phase 2 shared by both fetch paths: fetch BODY.PEEK[] (and
        X-GM-MSGID) only for the qualifying emails, then reassemble in
        qualifying order (IMAP returns ascending; we re-map via sequence
        ids).
        """
        try:
            BATCH_SIZE = int(os.getenv("IMAP_BATCH_SIZE") or "50")
        except ValueError:
            BATCH_SIZE = 50

        body_by_seq: Dict[str, Tuple[str, Message]] = {}

        for i in range(0, len(qualifying_seq_ids), BATCH_SIZE):
            batch_seq = qualifying_seq_ids[i:i + BATCH_SIZE]
            ids_str = b','.join(seq for seq, _gid in batch_seq)

            typ, msg_data = self.connection.fetch(ids_str, '(BODY.PEEK[] X-GM-MSGID)')
            if typ != 'OK':
                continue

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    metadata = response_part[0].decode('utf-8', errors='ignore')
                    seq_match = SEQ_PATTERN.match(metadata)
                    msgid_match = X_GM_MSGID_PATTERN.search(metadata)
                    if not seq_match or not msgid_match:
                        continue
                    seq_id_str = seq_match.group(1)
                    gmail_id = msgid_match.group(1)
                    raw_email = response_part[1]
                    msg = email.message_from_bytes(raw_email)
                    body_by_seq[seq_id_str] = (gmail_id, msg)

        # Reassemble in qualifying order (matching qualifying_seq_ids)
        results: List[Tuple[str, Message]] = []
        for seq_id in qualifying_seq_ids:
            key = seq_id[0].decode('utf-8', errors='ignore')
            if key in body_by_seq:
                results.append(body_by_seq[key])

        return results

    def has_unprocessed_unseen_emails(self, known_labels: List[str]) -> bool:
        """
        True iff at least one UNSEEN message in INBOX carries NONE of the
        known_labels — i.e. the unread queue still holds unprocessed mail.

        Metadata-only (no body download) and early-stopping. This is the
        hard-gate probe for the read-mail sweep (WI-103): the sweep may only
        start when this returns False. Any probe failure returns True —
        fail closed: never sweep read mail while the unread state is unknown.
        """
        def _do():
            self.connect()
            typ, data = self.connection.search(None, 'UNSEEN')
            if typ != 'OK' or not data[0]:
                return False
            email_ids = data[0].split()[::-1]  # newest first, like the fetch path
            qualifying = self._scan_qualifying_metadata(email_ids, set(known_labels), limit=1)
            return len(qualifying) > 0

        try:
            return with_retry(
                _do,
                retries=3,
                backoff=2.0,
                exceptions=_IMAP_ERRORS,
                on_retry=lambda exc, _: self._reset_connection(),
            )
        except Exception as e:
            logger.error(f"Unread-queue drain probe failed; treating as NOT drained: {e}")
            return True

    def _negated_seen_criteria(self, known_labels: List[str]) -> Optional[str]:
        """
        SEARCH criteria for read-mail sweep candidates: every READ email
        carrying NONE of the known labels, evaluated server-side.

        SEEN + one NOT X-GM-LABELS per known label (multiple keys are an
        implicit AND per RFC 3501). Returns None (caller uses the plain
        `SEEN` fallback) when there is nothing to negate or the criteria
        would be oversized. X-GM-LABELS is a documented SEARCH criterion
        (Gmail IMAP extensions); a server without NOT support returns a
        SUPERSET — the client-side label filter in _scan_qualifying_metadata
        remains the correctness backstop — or rejects the command, which the
        caller treats as the same fallback.
        """
        if not known_labels:
            return None
        not_chain = " ".join(
            f"NOT X-GM-LABELS {_imap_quote_label(label)}" for label in known_labels
        )
        if len(not_chain) > MAX_NEGATED_SEARCH_CHARS:
            logger.info(
                "Read sweep: %d known labels exceed the negated-SEARCH size cap; "
                "falling back to plain SEEN scan.", len(known_labels),
            )
            return None
        return f"SEEN {not_chain}"

    @staticmethod
    def _list_sample(entries, cap: int = 600) -> str:
        """Raw-world evidence: the LIST response sample (first lines plus
        any line mentioning 'all'), truncated for the sweep-state payload."""
        lines = []
        for entry in entries[:10]:
            if isinstance(entry, bytes):
                lines.append(entry.decode("utf-8", errors="ignore"))
            elif isinstance(entry, tuple):
                # imaplib turns some responses into tuples; flatten safely
                lines.append(" ".join(str(part) for part in entry))
            else:
                lines.append(str(entry))
        all_plus_first = [ln for ln in lines if "all" in ln.lower()] or lines
        sample = f"{len(entries)} entries: " + " || ".join(all_plus_first[:5])
        return sample[:cap]

    def _select_all_mail(self) -> bool:
        """
        Select the account's All Mail mailbox for the read sweep (WI-103:
        the backlog is archived history, which is invisible from INBOX).

        Resolution order (resolved name is cached on the client instance):
          1. the name resolved by a previous call;
          2. LIST + the RFC 6154 \\All special-use attribute (covers ANY
             localized folder NAME — the attribute is protocol-level);
          3. the conventional Gmail literals, IN ORDER, each verified by an
             actual SELECT before it may be cached:
             "[Gmail]/All Mail" → "[Google Mail]/All Mail" (accounts served
             under the googlemail.com domain use the Google Mail prefix);
          4. give up: warn (including the raw LIST sample) and return False
             — the caller stays on the already-selected INBOX, so behavior
             degrades explicitly, never silently. The full list sample and
             per-limb select results persist on the client as
             _all_mail_list_sample / _all_mail_discovery_error for
             GET /admin/sweep-state, and are cleared on next success.

        Only the Session's SELECTION changes here; every other IMAP helper
        in this file expects INBOX, so the sweep's caller must restore INBOX
        afterwards (it does, mirroring _labels_search_in_mailboxes).
        """
        limb_notes: list = []

        def _try_select(name: str, note: bool = False) -> bool:
            # Wire form is always the QUOTED mailbox string — live Gmail
            # BADs the bare special-char name (WI-103 follow-up 7).
            wire_name = _imap_quote_mailbox(name)
            try:
                typ, data = self.connection.select(wire_name)
                if typ == "OK":
                    if note:
                        limb_notes.append(f"SELECT {name!r} (sent {wire_name!r}) succeeded")
                    return True
                if note:
                    limb_notes.append(f"SELECT {name!r} (sent {wire_name!r}) -> typ={typ!r}")
                return False
            except _IMAP_ERRORS as e:
                if note:
                    limb_notes.append(
                        f"SELECT {name!r} (sent {wire_name!r}) raised "
                        f"{type(e).__name__}: {e}"
                    )
                return False

        if self._all_mail_name:
            # Normalize the cache: it always holds the LOGICAL (unquoted)
            # name; _try_select quotes for the wire (idempotently).
            self._all_mail_name = self._all_mail_name.strip('"')
            if _try_select(self._all_mail_name):
                self._all_mail_discovery_error = None
                self._all_mail_list_sample = None
                return True
            # Cached name stopped resolving (renamed/deleted mailbox):
            # forget it and re-resolve below.
            self._all_mail_name = None

        sample: Optional[str] = None
        try:
            typ, data = self.connection.list('""', '*')
            if typ == "OK":
                sample = self._list_sample(list(data) if data else [])
                self._all_mail_list_sample = sample
                if data:
                    for entry in data:
                        text = (entry.decode("utf-8", errors="ignore")
                                if isinstance(entry, bytes) else str(entry))
                        flags = text.split(")", 1)[0].lower()
                        # Name = the LAST quoted token of the LIST line
                        # ("(\flags) "delimiter" "name"). Don't split on
                        # whitespace — mailbox names contain spaces ("All
                        # Mail") — and don't reuse the split-on-'/' parsing
                        # from _supplement_mailbox_names, which leaves the
                        # delimiter's closing quote+space glued to the name
                        # (' "Gmail/…') on quoted-"/" servers.
                        name_match = _LIST_QUOTED_NAME_PATTERN.search(text)
                        name = name_match.group(1) if name_match else ""
                        if "\\all" in flags and name:
                            if _try_select(name):
                                limb_notes.append(f"LIST \\All attribute resolved {name!r}")
                                self._all_mail_name = name
                                self._all_mail_discovery_error = None
                                self._all_mail_list_sample = None
                                return True
                            limb_notes.append(
                                f"LIST found \\All on {name!r} but SELECT failed"
                            )
            else:
                limb_notes.append(f"LIST returned typ={typ!r}")
        except _IMAP_ERRORS as e:
            limb_notes.append(f"LIST raised {type(e).__name__}: {e}")
            if sample is None:
                self._all_mail_list_sample = None

        # Literal fallbacks, each SELECT-verified before it may be cached.
        # The Google Mail prefix covers accounts served under
        # googlemail.com — the live-Gmail failure behind WI-103 follow-up 4.
        for literal in ALL_MAIL_FALLBACK_NAMES:
            if _try_select(literal, note=True):
                self._all_mail_name = literal
                self._all_mail_discovery_error = None
                self._all_mail_list_sample = None
                return True

        self._all_mail_discovery_error = "; ".join(limb_notes)[:600]
        logger.warning(
            "Read sweep: no All Mail mailbox could be selected (LIST found no "
            "\\All mailbox; literal fallbacks %r unavailable); staying on INBOX. "
            "LIST sample: %s",
            list(ALL_MAIL_FALLBACK_NAMES), self._all_mail_list_sample or "<none>",
        )
        return False

    def fetch_read_unprocessed_emails(
        self,
        known_labels: List[str],
        limit: Optional[int] = None,
        skip_ids: Optional[set] = None,
    ) -> List[Tuple[str, Message]]:
        """
        Fetch READ (\\Seen) emails that do not carry any of the known_labels
        and are not in skip_ids. Returns [(gmail_id, msg)], newest first.

        SCOPE (WI-103 follow-up): the sweep runs in the account's All Mail
        mailbox — archived read history is the actual backlog and is
        invisible from INBOX. The UNSEEN phase and the
        has_unprocessed_unseen_emails() hard gate remain INBOX-scoped:
        unread mail in the INBOX governs priority exactly as the user's
        rule states. Every successful sweep restores the INBOX selection
        before returning (all other helpers expect INBOX); a connection
        whose restore fails is dropped so the next call reconnects fresh.

        This method contains NO ordering/interleaving logic of its own: the
        caller MUST sequence it after the UNSEEN phase and only once
        has_unprocessed_unseen_emails() is False (the hard gate).

        Candidate reduction: a server-side `SEEN NOT X-GM-LABELS ...` search
        is attempted first (cheap in steady state — an empty result means no
        scan at all). A rejected, oversized, or failing criteria falls back
        to a plain SEEN search whose results are filtered client-side in
        _scan_qualifying_metadata — correct in every case, merely costlier.

        skip_ids = processed-marker (e.g. gmail ids with a DB log row):
        zero-label read mail the system already processed once is excluded
        before any body download, so it is never endlessly re-fetched or
        re-classified cycle after cycle.
        """
        def _do():
            self.connect()
            try:
                # Scope: read history lives in All Mail (archived mail is
                # invisible from INBOX). On a server without All Mail this
                # warns and the cycle runs in the already-selected INBOX.
                if not self._select_all_mail():
                    logger.warning(
                        "Read sweep: All Mail unavailable — sweeping the "
                        "selected INBOX this cycle."
                    )
                mailbox = self._all_mail_name or "INBOX"

                fallback = True   # plain-SEEN scan unless the negated search succeeded
                email_ids: List[bytes] = []

                criteria = self._negated_seen_criteria(known_labels)
                if criteria is not None:
                    try:
                        typ, data = self.connection.search(None, criteria)
                        if typ == 'OK':
                            fallback = False
                            email_ids = (data[0].split()[::-1] if data[0] else [])
                            logger.info(
                                f"Read sweep: negated label search matched "
                                f"{len(email_ids)} candidates in {mailbox}"
                            )
                        else:
                            logger.warning(
                                "Read sweep: negated label search returned %s; "
                                "falling back to plain SEEN scan.", typ,
                            )
                    except _IMAP_ERRORS as e:
                        logger.warning(
                            f"Read sweep: negated label search rejected by server ({e}); "
                            "falling back to plain SEEN scan."
                        )

                if fallback:
                    typ, data = self.connection.search(None, 'SEEN')
                    if typ != 'OK' or not data[0]:
                        return []
                    email_ids = data[0].split()[::-1]

                self.last_sweep_search_mode = (
                    "plain_seen" if fallback else "negated"
                )

                qualifying = self._scan_qualifying_metadata(
                    email_ids, set(known_labels), limit=limit, skip_ids=skip_ids,
                )
                if not qualifying:
                    return []
                return self._fetch_bodies(qualifying)
            finally:
                # Every other IMAP helper in this file expects INBOX, and a
                # cached connection keeps its selection across calls
                # (connect() merely NOOPs). Restore exactly like
                # _labels_search_in_mailboxes: on failure drop the cached
                # connection so the next call reconnects into INBOX.
                try:
                    self.connection.select("INBOX")
                except Exception:
                    self.connection = None

        return with_retry(
            _do,
            retries=3,
            backoff=2.0,
            exceptions=_IMAP_ERRORS,
            on_retry=lambda exc, _: self._reset_connection(),
        )

    def _search_by_gmail_id(self, gmail_id: str) -> bytes:
        """
        Helper to find the UID of an email given its X-GM-MSGID, searching
        the INBOX selection.

        Returns the UID as bytes, or None if not found.
        Raises imaplib.IMAP4.error or OSError on connection failures.

        Label WRITE paths use _resolve_uid instead, which also searches
        out-of-inbox mailboxes. Deliberately NOT given to
        fetch_email_by_gmail_id (reclassify): making archived bodies
        fetchable would silently change reclassify's scope.
        """
        self.connect()
        typ, data = self.connection.uid('SEARCH', None, f'X-GM-MSGID {gmail_id}')
        if typ == 'OK' and data[0]:
            # Return the last one if multiple (shouldn't be multiple for one ID usually)
            return data[0].split()[-1]
        return None

    def _store_label(self, gmail_id: str, sign: str, label: str, verify: bool) -> None:
        """
        Shared STORE for +/-X-GM-LABELS with mailbox-aware uid resolution.
        UIDs are PER-MAILBOX: the STORE must happen in the SAME selection
        that resolved the uid — resolving in All Mail and storing in INBOX
        would target an unrelated message (or nothing). So: try INBOX
        (a fresh connect always selects INBOX); on a search miss walk the
        supplement mailboxes, storing inside each selection as we go;
        restore INBOX no matter what.

        Raises imaplib.IMAP4.error on any failure — INCLUDING search-miss
        in every mailbox — so with_retry retries transient part-failures
        and the caller's except path sees the final False. Removal
        verification (parameter verify) reads the labels back server-side:
        a completed STORE is not proof of effect (2026-10 live bug).
        """
        label_to_send = _imap_quote_label(label)

        def _store_in_current_selection():
            typ, data = self.connection.uid('SEARCH', None, f'X-GM-MSGID {gmail_id}')
            if not (typ == 'OK' and data[0]):
                return False  # not in THIS mailbox — caller walks to the next one
            uid = data[0].split()[-1]
            typ, store_data = self.connection.uid(
                'STORE', uid, f'{sign}X-GM-LABELS', f'({label_to_send})'
            )
            if typ != 'OK':
                raise imaplib.IMAP4.error(
                    f"STORE {sign}X-GM-LABELS {label!r} on {gmail_id} returned {typ}: {store_data}"
                )
            if verify:
                typ, vdata = self.connection.uid('FETCH', uid, '(X-GM-LABELS)')
                if typ != 'OK':
                    raise imaplib.IMAP4.error(
                        f"post-store verification fetch failed for {gmail_id}: {vdata}"
                    )
                remaining = self._labels_from_fetch_response(vdata)
                if label in remaining:
                    raise imaplib.IMAP4.error(
                        f"label {label!r} still present on {gmail_id} after removal STORE "
                        f"(remaining labels: {remaining})"
                    )
            return True

        self.connect()
        try:
            if not _store_in_current_selection():  # INBOX first
                for mailbox in self._supplement_mailbox_names():
                    typ, _ = self.connection.select(mailbox)
                    if typ != 'OK':
                        continue
                    if _store_in_current_selection():
                        return
                raise imaplib.IMAP4.error(
                    f"email {gmail_id} not found in any mailbox for {sign}X-GM-LABELS"
                )
        finally:
            try:
                self.connection.select("INBOX")
            except Exception:
                self.connection = None  # force a fresh (INBOX) connection next call

    def _labels_from_fetch_response(self, data) -> list:
        """Flatten a FETCH (X-GM-LABELS) response into a label list."""
        remaining = []
        for response_part in data:
            metadata = (response_part[0] if isinstance(response_part, tuple)
                        else response_part)
            if isinstance(metadata, bytes):
                metadata = metadata.decode('utf-8', errors='ignore')
            match = X_GM_LABELS_PATTERN.search(metadata)
            if match:
                for m in LABEL_TOKEN_PATTERN.finditer(match.group(1)):
                    remaining.append(
                        m.group(1).replace('\\"', '"').replace('\\\\', '\\')
                        if m.group(1) else m.group(2)
                    )
        return remaining

    def apply_label(self, gmail_id: str, label: str) -> bool:
        """
        Apply a label to the email using UID STORE +X-GM-LABELS.
        Accepts gmail_id (X-GM-MSGID). Uid resolution covers archived and
        trashed-but-not-purged mail (_store_label).

        Returns True iff the STORE completed; False on search-miss or
        repeated store failure — callers must surface False, not assume
        success (a swallowed failure here is how Gmail ended up stale while
        the DB said 'corrected').
        """
        def _do():
            self._store_label(gmail_id, "+", label, verify=False)

        try:
            with_retry(
                _do,
                retries=3,
                backoff=2.0,
                exceptions=_IMAP_ERRORS,
                on_retry=lambda exc, _: self._reset_connection(),
            )
            return True
        except Exception as e:
            logger.error(f"Error applying label {label!r} to {gmail_id}: {e}")
            return False

    def remove_label(self, gmail_id: str, label: str) -> bool:
        """
        Remove a label from the email using UID STORE -X-GM-LABELS.
        Accepts gmail_id (X-GM-MSGID). Uid resolution covers archived and
        trashed-but-not-purged mail (_store_label).

        Returns True iff the label is CONFIRMED absent afterwards. Any
        failure — search-miss in every mailbox, STORE error after retries,
        or a STORE that completed without actually removing the label
        (2026-10 live bug: DB said corrected, Gmail kept the old label) —
        returns False. Callers must surface that instead of proceeding as
        if Gmail now matches the database.
        """
        def _do():
            self._store_label(gmail_id, "-", label, verify=True)

        try:
            with_retry(
                _do,
                retries=3,
                backoff=2.0,
                exceptions=_IMAP_ERRORS,
                on_retry=lambda exc, _: self._reset_connection(),
            )
            return True
        except Exception as e:
            logger.error(f"Error removing label {label!r} from {gmail_id}: {e}")
            return False

    def fetch_email_by_gmail_id(self, gmail_id: str) -> Message:
        """
        Fetch the email content for a given X-GM-MSGID.
        """
        def _do():
            self.connect()
            uid = self._search_by_gmail_id(gmail_id)
            if not uid:
                return None

            # Fetch BODY.PEEK[] based on UID
            typ, msg_data = self.connection.uid('FETCH', uid, '(BODY.PEEK[])')
            if typ != 'OK':
                return None

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    return email.message_from_bytes(response_part[1])
            return None

        try:
            return with_retry(
                _do,
                retries=3,
                backoff=2.0,
                exceptions=_IMAP_ERRORS,
                on_retry=lambda exc, _: self._reset_connection(),
            )
        except Exception as e:
            logger.error(f"Error fetching email {gmail_id}: {e}")
            return None

    def get_labels_for_emails(self, gmail_ids: List[str]) -> Dict[str, List[str]]:
        """
        Fetch current labels for a list of Gmail IDs (X-GM-MSGID).
        Returns {gmail_id: [label1, label2, ...]}

        The primary pass searches the selected INBOX mailbox. Gmail ids not
        found there (archived, or trashed but not yet purged — the user's
        "label it, then remove it from the inbox" workflow) are retried in a
        bounded out-of-inbox supplement (_supplement_labels): the same
        X-GM-MSGID OR-search executed in [Gmail]/All Mail and, for what is
        still missing, [Gmail]/Trash. The session mailbox is always restored
        to INBOX afterwards. Spam is deliberately NOT searched — repairing a
        classification from inside Spam is not a supported flow.
        """
        def _do():
            self.connect()
            results = {}

            # Batch search for UIDs using OR logic
            # SEARCH (X-GM-MSGID 1 OR X-GM-MSGID 2 ...)
            # We need to construct this carefully to not exceed line length limits.
            # But usually IMAP libs handle large commands or we batch search too.

            uid_to_gmail_id = {}
            uids_to_fetch = []

            # Batch the SEARCH command too
            SEARCH_BATCH = 50
            for i in range(0, len(gmail_ids), SEARCH_BATCH):
                batch_gids = gmail_ids[i:i+SEARCH_BATCH]

                # Construct search criteria
                # Logic: (OR X-GM-MSGID <id> (OR X-GM-MSGID <id> ...))
                # Or simpler: OR OR OR (if library supports raw string)
                # Standard IMAP search for multiple items is often OR key1 OR key2 ...
                # But the OR operator takes two arguments.
                # So for N items, we need N-1 ORs nested or chained?
                # Actually, `SEARCH OR <key> <key>` only works for 2?
                # RFC 3501: OR <search-key> <search-key>
                # So for 3 items: OR <key1> OR <key2> <key3>
                # For N items: (N-1) "OR " prefixes followed by N keys.

                if not batch_gids:
                    continue

                if len(batch_gids) == 1:
                    criteria = f'X-GM-MSGID {batch_gids[0]}'
                else:
                    # Prefix (N-1) times "OR"
                    # Example for [1, 2, 3]: OR X-GM-MSGID 1 OR X-GM-MSGID 2 X-GM-MSGID 3
                    prefixes = "OR " * (len(batch_gids) - 1)
                    keys = " ".join([f'X-GM-MSGID {gid}' for gid in batch_gids])
                    criteria = f'{prefixes}{keys}'

                # We need to fetch UIDs and ideally X-GM-MSGID in the response to map them back?
                # SEARCH returns only IDs (UIDs). It doesn't tell us which criteria matched which ID.
                # So we have to fetch X-GM-MSGID for the found UIDs to rebuild the map.
                typ, data = self.connection.uid('SEARCH', None, criteria)

                if typ == 'OK' and data[0]:
                    found_uids = data[0].split()
                    if found_uids:
                        uids_to_fetch.extend(found_uids)

            if uids_to_fetch:
                # Fetch in batches (re-using uids found from search)
                # We fetch X-GM-MSGID again to map UID -> GmailID reliably
                BATCH_SIZE = 50
                for i in range(0, len(uids_to_fetch), BATCH_SIZE):
                    batch_uids = uids_to_fetch[i:i+BATCH_SIZE]
                    uid_str = b','.join(batch_uids)

                    typ, data = self.connection.uid('FETCH', uid_str, '(X-GM-MSGID X-GM-LABELS)')
                    if typ != 'OK':
                        continue

                    for response_part in data:
                        if isinstance(response_part, tuple):
                            metadata = response_part[0].decode('utf-8', errors='ignore')
                        else:
                            metadata = response_part.decode('utf-8', errors='ignore')

                        # Extract X-GM-MSGID to map back to input
                        msgid_match = X_GM_MSGID_PATTERN.search(metadata)
                        if not msgid_match:
                            continue
                        gid = msgid_match.group(1)

                        # Extract labels
                        labels = []
                        match = X_GM_LABELS_PATTERN.search(metadata)
                        if match:
                            labels_str = match.group(1)
                            # Parse labels taking quotes into account
                            for m in LABEL_TOKEN_PATTERN.finditer(labels_str):
                                if m.group(1):
                                    labels.append(m.group(1).replace('\\"', '"').replace('\\\\', '\\'))
                                else:
                                    labels.append(m.group(2))

                        results[gid] = labels

            # Out-of-inbox supplement: retry only the ids the INBOX pass did
            # not find (archived, or trashed but not purged). Bounded at the
            # same SEARCH_BATCH; missing results simply stay absent from the
            # returned dict — the recheck treats them the same as before
            # (correction.py skips unknown ids).
            missing = [g for g in gmail_ids if str(g) not in results]
            if missing:
                supplement = self._supplement_labels(missing)
                if supplement:
                    results.update(supplement)

            return results

        try:
            return with_retry(
                _do,
                retries=3,
                backoff=2.0,
                exceptions=_IMAP_ERRORS,
                on_retry=lambda exc, _: self._reset_connection(),
            )
        except Exception as e:
            logger.error(f"Error fetching labels batch: {e}")
            return {}

    def _supplement_labels(self, missing: List[str]) -> Dict[str, List[str]]:
        """
        Bounded out-of-inbox label read for gmail ids the INBOX pass missed.

        Re-runs the same X-GM-MSGID OR-search in [Gmail]/All Mail and, for
        whatever is still missing, [Gmail]/Trash; the session mailbox is
        restored to INBOX in all paths. Every failure degrades to today's
        behaviour: no entries for the missing ids (the recheck's silent-skip
        path). NOTE: we do NOT use `X-GM-RAW 'in:anywhere …'`, because
        X-GM-MSGID is not a Gmail web-search operator and the raw search
        would silently match nothing; a temporary mailbox selection with the
        standard search key reaches the same messages.

        Spam is skipped deliberately (see get_labels_for_emails docstring).
        """
        try:
            if not self.connection:
                return {}
            return self._labels_search_in_mailboxes(missing, self._supplement_mailbox_names())
        except Exception as e:
            logger.info(f"Out-of-inbox label supplement unavailable ({e}); "
                        f"missing ids stay unread")
            return {}

    def _supplement_mailbox_names(self) -> List[str]:
        """
        Resolve the special-use mailbox names to search for out-of-inbox
        mail, newest-traffic first. Uses LIST + RFC 6154 special-use flags
        (\\All, \\Trash) so localized Gmail UI languages still resolve;
        falls back to the conventional [Gmail]/ names.
        """
        names: List[str] = []
        all_mail_found = False
        trash_found = False
        try:
            typ, data = self.connection.list('""', '*')
            if typ == 'OK' and data:
                for entry in data:
                    text = entry.decode('utf-8', errors='ignore') if isinstance(entry, bytes) else str(entry)
                    flags = text.split(')', 1)[0].lower()
                    name = text.split(')', 1)[-1].split('/', 1)[-1].strip().strip('"')
                    if '\\all' in flags:
                        names.insert(0, name)   # All Mail first
                        all_mail_found = True
                    elif '\\trash' in flags:
                        trash_found = True
                        names.append(name)      # Trash after All Mail
        except Exception as e:
            logger.info(f"IMAP LIST for special mailboxes failed ({e}); using default names")

        # Fallback: conventional (English) Gmail mailbox names — used when
        # flag-based discovery found nothing (older/quirkier servers).
        if not all_mail_found and "[Gmail]/All Mail" not in names:
            names.insert(0, "[Gmail]/All Mail")
        if not trash_found and "[Gmail]/Trash" not in names:
            names.append("[Gmail]/Trash")
        return names

    def _labels_search_in_mailboxes(self, gmail_ids: List[str], mailbox_names: List[str]) -> Dict[str, List[str]]:
        """
        Search the given mailboxes for the gmail ids not yet resolved and
        return {gmail_id: labels}. Always restores INBOX as the selected
        mailbox before returning, and only ever issues batches of at most
        SEARCH_BATCH (50) ids — same cost shape as the primary pass.
        """
        results: Dict[str, List[str]] = {}
        remaining = list(gmail_ids)
        SEARCH_BATCH = 50

        for mailbox in mailbox_names:
            if not remaining:
                break
            try:
                typ, _ = self.connection.select(mailbox)
                if typ != 'OK':
                    continue
            except Exception as e:
                logger.info(f"Could not select {mailbox} for label supplement: {e}")
                continue

            still_missing: List[str] = []
            fetch_uids: set = set()
            for i in range(0, len(remaining), SEARCH_BATCH):
                batch_gids = remaining[i:i + SEARCH_BATCH]
                if len(batch_gids) == 1:
                    criteria = f'X-GM-MSGID {batch_gids[0]}'
                else:
                    prefixes = "OR " * (len(batch_gids) - 1)
                    keys = " ".join(f'X-GM-MSGID {gid}' for gid in batch_gids)
                    criteria = f'{prefixes}{keys}'

                typ, data = self.connection.uid('SEARCH', None, criteria)
                if typ == 'OK' and data[0]:
                    found_uids = data[0].split()
                    fetch_uids.update(found_uids)
                # Whatever this batch/bbox did not produce is retried in the
                # next mailbox (or silently dropped if none left)
                # (resolved ids are removed below after mapping)

            for i in range(0, len(fetch_uids), SEARCH_BATCH):
                batch_uids = sorted(fetch_uids)[i:i + SEARCH_BATCH]
                uid_str = b','.join(batch_uids)
                typ, data = self.connection.uid('FETCH', uid_str, '(X-GM-MSGID X-GM-LABELS)')
                if typ != 'OK':
                    continue
                for response_part in data:
                    if isinstance(response_part, tuple):
                        metadata = response_part[0].decode('utf-8', errors='ignore')
                    else:
                        metadata = response_part.decode('utf-8', errors='ignore')

                    msgid_match = X_GM_MSGID_PATTERN.search(metadata)
                    if not msgid_match:
                        continue
                    gid = msgid_match.group(1)

                    labels = []
                    match = X_GM_LABELS_PATTERN.search(metadata)
                    if match:
                        labels_str = match.group(1)
                        for m in LABEL_TOKEN_PATTERN.finditer(labels_str):
                            if m.group(1):
                                labels.append(m.group(1).replace('\\"', '"').replace('\\\\', '\\'))
                            else:
                                labels.append(m.group(2))

                    # Even a zero-label entry counts as "seen" — an email
                    # with all labels stripped should stop being retried in
                    # later mailboxes.
                    results[gid] = labels

            remaining = [g for g in remaining if str(g) not in results]

        # Restore the session mailbox: all other IMAP helpers expect INBOX
        # (it is selected in connect(), imap_client.py connect()).
        try:
            self.connection.select("INBOX")
        except Exception:
            self.connection = None  # force a fresh (INBOX) connection next call

        return results

    def scan_labeled_emails(self, known_labels: List[str]) -> Dict[str, Tuple[List[str], Message]]:
        """
        Scan INBOX for all emails that have at least one of the known_labels.
        Returns {gmail_id: (labels_list, email_message_object)}.

        This is used to discover emails that have labels applied in Gmail
        but are missing from the local database (e.g. after a DB reset).
        """
        def _do():
            self.connect()
            results: Dict[str, Tuple[List[str], Message]] = {}

            all_uids = set()

            # Search for each known label
            for label in known_labels:
                # X-GM-LABELS requires the label to be quoted and escaped
                criteria = f'X-GM-LABELS {_imap_quote_label(label)}'
                typ, data = self.connection.uid('SEARCH', None, criteria)
                if typ == 'OK' and data[0]:
                    found_uids = data[0].split()
                    all_uids.update(found_uids)

            if not all_uids:
                return results

            logger.info(f"Found {len(all_uids)} unique emails with known labels in IMAP.")

            # Batch fetch: get BODY, X-GM-MSGID, and X-GM-LABELS for all found UIDs
            uids_list = list(all_uids)
            BATCH_SIZE = 50

            for i in range(0, len(uids_list), BATCH_SIZE):
                batch_uids = uids_list[i:i + BATCH_SIZE]
                uid_str = b','.join(batch_uids)

                typ, data = self.connection.uid(
                    'FETCH', uid_str, '(BODY.PEEK[] X-GM-MSGID X-GM-LABELS)'
                )
                if typ != 'OK':
                    continue

                for response_part in data:
                    if not isinstance(response_part, tuple):
                        continue

                    metadata = response_part[0].decode('utf-8', errors='ignore')

                    # Extract X-GM-MSGID
                    msgid_match = X_GM_MSGID_PATTERN.search(metadata)
                    if not msgid_match:
                        continue
                    gid = msgid_match.group(1)

                    # Extract labels
                    labels = []
                    match = X_GM_LABELS_PATTERN.search(metadata)
                    if match:
                        labels_str = match.group(1)
                        for m in LABEL_TOKEN_PATTERN.finditer(labels_str):
                            if m.group(1):
                                labels.append(
                                    m.group(1).replace('\\"', '"').replace('\\\\', '\\')
                                )
                            else:
                                labels.append(m.group(2))

                    # Parse email body
                    raw_email = response_part[1]
                    msg = email.message_from_bytes(raw_email)

                    results[gid] = (labels, msg)

            return results

        try:
            return with_retry(
                _do,
                retries=3,
                backoff=2.0,
                exceptions=_IMAP_ERRORS,
                on_retry=lambda exc, _: self._reset_connection(),
            )
        except Exception as e:
            logger.warning(f"Error scanning labeled emails after retries: {e}")
            return {}


# Module-level singleton — shared across all background jobs so that a single
# persistent SSL connection is reused instead of opening and closing one on
# every job invocation.  connect() is called lazily inside each method and
# already handles reconnection (via NOOP) when the server has dropped the link.
gmail_client = GmailClient()
