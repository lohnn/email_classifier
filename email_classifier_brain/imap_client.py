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

class GmailClient:
    def __init__(self):
        # Prefer IMAP_USER, fallback to first email in MY_EMAIL
        self.user = os.getenv("IMAP_USER") or (os.getenv("MY_EMAIL") or "").split(",")[0].strip()
        self.password = os.getenv("IMAP_PASSWORD")
        self.connection = None

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

            try:
                BATCH_SIZE = int(os.getenv("IMAP_BATCH_SIZE") or "50")
            except ValueError:
                BATCH_SIZE = 50

            # ------------------------------------------------------------------
            # Phase 1: metadata-only scan (no body download) newest-first.
            # Collects qualifying sequence IDs and their gmail IDs up to limit.
            #
            # NOTE: IMAP FETCH returns results in ascending sequence-number order
            # regardless of the order requested.  To honour newest-first we build
            # a lookup dict from each batch response, then walk `batch_ids` (which
            # *are* in newest-first order) to decide qualification order.
            # ------------------------------------------------------------------
            qualifying_seq_ids: List[bytes] = []
            known_labels_set = set(known_labels)

            for i in range(0, len(email_ids), BATCH_SIZE):
                batch_ids = email_ids[i:i + BATCH_SIZE]
                ids_str = b','.join(batch_ids)

                typ, msg_data = self.connection.fetch(ids_str, '(X-GM-LABELS X-GM-MSGID)')
                if typ != 'OK':
                    continue

                # Build a lookup: seq_id (str) -> labels_str
                metadata_by_seq: Dict[str, str] = {}
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

                    metadata_by_seq[seq_id_str] = labels_str

                # Walk batch_ids in newest-first order to preserve ordering
                for bid in batch_ids:
                    bid_str = bid.decode('utf-8', errors='ignore')
                    if bid_str not in metadata_by_seq:
                        continue

                    labels_str = metadata_by_seq[bid_str]

                    # Skip if any known label is already applied
                    skip = False
                    for m in LABEL_TOKEN_PATTERN.finditer(labels_str):
                        label_found = m.group(1).replace('\\"', '"').replace('\\\\', '\\') if m.group(1) else m.group(2)
                        if label_found in known_labels_set:
                            skip = True
                            break

                    if not skip:
                        qualifying_seq_ids.append(bid)
                        if limit is not None and len(qualifying_seq_ids) >= limit:
                            break  # got enough from this batch

                if limit is not None and len(qualifying_seq_ids) >= limit:
                    break  # outer loop – got enough

            if not qualifying_seq_ids:
                return []

            # ------------------------------------------------------------------
            # Phase 2: fetch full bodies only for the qualifying emails.
            #
            # Again, IMAP returns results in ascending order, so we collect into
            # a dict and then reassemble in qualifying_seq_ids order.
            # ------------------------------------------------------------------
            body_by_seq: Dict[str, Tuple[str, Message]] = {}

            for i in range(0, len(qualifying_seq_ids), BATCH_SIZE):
                batch_seq = qualifying_seq_ids[i:i + BATCH_SIZE]
                ids_str = b','.join(batch_seq)

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

            # Reassemble in newest-first order (matching qualifying_seq_ids)
            results: List[Tuple[str, Message]] = []
            for seq_id in qualifying_seq_ids:
                key = seq_id.decode('utf-8', errors='ignore')
                if key in body_by_seq:
                    results.append(body_by_seq[key])

            return results

        return with_retry(
            _do,
            retries=3,
            backoff=2.0,
            exceptions=_IMAP_ERRORS,
            on_retry=lambda exc, _: self._reset_connection(),
        )

    def _search_by_gmail_id(self, gmail_id: str) -> bytes:
        """
        Helper to find the UID of an email given its X-GM-MSGID.
        Returns the UID as bytes, or None if not found.
        Raises imaplib.IMAP4.error or OSError on connection failures.
        """
        self.connect()
        typ, data = self.connection.uid('SEARCH', None, f'X-GM-MSGID {gmail_id}')
        if typ == 'OK' and data[0]:
            # Return the last one if multiple (shouldn't be multiple for one ID usually)
            return data[0].split()[-1]
        return None

    def apply_label(self, gmail_id: str, label: str):
        """
        Apply a label to the email using UID STORE +X-GM-LABELS.
        Accepts gmail_id (X-GM-MSGID).
        """
        # Quote label if it has spaces
        label_to_send = f'"{label}"' if " " in label else label

        def _do():
            self.connect()
            uid = self._search_by_gmail_id(gmail_id)
            if not uid:
                print(f"Could not find email with Gmail ID {gmail_id} to apply label.")
                return
            typ, data = self.connection.uid('STORE', uid, '+X-GM-LABELS', f'({label_to_send})')
            if typ != 'OK':
                raise imaplib.IMAP4.error(f"Failed to apply label {label} to {gmail_id}: {data}")

        try:
            with_retry(
                _do,
                retries=3,
                backoff=2.0,
                exceptions=_IMAP_ERRORS,
                on_retry=lambda exc, _: self._reset_connection(),
            )
        except Exception as e:
            print(f"Error applying label {label} to {gmail_id}: {e}")

    def remove_label(self, gmail_id: str, label: str):
        """
        Remove a label from the email using UID STORE -X-GM-LABELS.
        Accepts gmail_id (X-GM-MSGID).
        """
        # Quote label if it has spaces
        label_to_send = f'"{label}"' if " " in label else label

        def _do():
            self.connect()
            uid = self._search_by_gmail_id(gmail_id)
            if not uid:
                print(f"Could not find email with Gmail ID {gmail_id} to remove label.")
                return
            typ, data = self.connection.uid('STORE', uid, '-X-GM-LABELS', f'({label_to_send})')
            if typ != 'OK':
                raise imaplib.IMAP4.error(f"Failed to remove label {label} from {gmail_id}: {data}")

        try:
            with_retry(
                _do,
                retries=3,
                backoff=2.0,
                exceptions=_IMAP_ERRORS,
                on_retry=lambda exc, _: self._reset_connection(),
            )
        except Exception as e:
            print(f"Error removing label {label} from {gmail_id}: {e}")

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
            print(f"Error fetching email {gmail_id}: {e}")
            return None

    def get_labels_for_emails(self, gmail_ids: List[str]) -> Dict[str, List[str]]:
        """
        Fetch current labels for a list of Gmail IDs (X-GM-MSGID).
        Returns {gmail_id: [label1, label2, ...]}
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

            if not uids_to_fetch:
                return results

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
            print(f"Error fetching labels batch: {e}")
            return {}

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
                # X-GM-LABELS requires the label to be quoted
                criteria = f'X-GM-LABELS "{label}"'
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
