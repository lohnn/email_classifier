import imaplib
import email
import os
import ssl
import re
from email.message import Message
from typing import List, Tuple, Dict
from dotenv import load_dotenv

load_dotenv()

IMAP_SERVER = os.getenv("IMAP_SERVER") or "imap.gmail.com"

# Regex to match X-GM-LABELS content.
# Handles atoms (no quotes/parens) and quoted strings (with escaped quotes/backslashes).
# Group 1 captures the content inside the parentheses.
X_GM_LABELS_PATTERN = re.compile(r'X-GM-LABELS \(((?:[^()"]+|"(\\.|[^"\\])*")*)\)')
X_GM_MSGID_PATTERN = re.compile(r'X-GM-MSGID (\d+)')
SEQ_PATTERN = re.compile(r'^(\d+)')

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

    def fetch_unprocessed_emails(self, known_labels: List[str]) -> List[Tuple[bytes, Message]]:
        """
        Fetch UNSEEN emails that do not have any of the known_labels.
        Returns a list of (gmail_id, email_message_object).
        """
        self.connect()

        # Search for UNSEEN emails
        typ, data = self.connection.search(None, 'UNSEEN')

        if typ != 'OK' or not data[0]:
            return []

        email_ids = data[0].split()
        results = []

        if not email_ids:
            return results

        try:
            BATCH_SIZE = int(os.getenv("IMAP_BATCH_SIZE", "50"))
        except ValueError:
            BATCH_SIZE = 50

        for i in range(0, len(email_ids), BATCH_SIZE):
            batch_ids = email_ids[i:i + BATCH_SIZE]

            # Construct a comma-separated list of IDs for batch fetching
            ids_str = b','.join(batch_ids)

            # Fetch BODY.PEEK[] (full content), X-GM-LABELS, and X-GM-MSGID
            # PEEK prevents marking as \Seen implicitly by the fetch of body.
            typ, msg_data = self.connection.fetch(ids_str, '(BODY.PEEK[] X-GM-LABELS X-GM-MSGID)')

            if typ != 'OK':
                continue

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    # response_part[0] is bytes, header line
                    # Example: b'1 (X-GM-LABELS (\\Inbox \\Important) X-GM-MSGID 123456789 BODY.PEEK[] {1234}'
                    metadata = response_part[0].decode('utf-8', errors='ignore')

                    # Extract sequence number (ID) - not used for return but for internal logic if needed
                    seq_match = SEQ_PATTERN.match(metadata)
                    if not seq_match:
                        continue
                    
                    # Extract X-GM-MSGID
                    msgid_match = X_GM_MSGID_PATTERN.search(metadata)
                    if not msgid_match:
                        # Should technically not happen if Gmail, but safeguard
                        continue
                    gmail_id = msgid_match.group(1)

                    # Extract content inside X-GM-LABELS (...)
                    labels_str = ""
                    match = X_GM_LABELS_PATTERN.search(metadata)
                    if match:
                        labels_str = match.group(1)

                    # The second element is the body content
                    raw_email = response_part[1]

                    skip = False
                    for label in known_labels:
                        # Robust check for label presence
                        # Gmail labels in the list are separated by spaces.
                        # Labels with spaces are quoted.

                        escaped_label = re.escape(label)
                        # We check if the label exists as a standalone token or quoted token
                        # This regex checks for:
                        # Start of string or space or open paren
                        # Optional quote
                        # The label
                        # Optional quote
                        # End of string or space or close paren

                        # Note: This is an approximation. A true parser would be better but this covers most cases.
                        pattern = fr'(?:^|\s|\()"??{escaped_label}"??(?:$|\s|\))'

                        if re.search(pattern, labels_str):
                            skip = True
                            break

                    if skip:
                        continue

                    msg = email.message_from_bytes(raw_email)
                    results.append((gmail_id, msg))

        return results

    def _search_by_gmail_id(self, gmail_id: str) -> bytes:
        """
        Helper to find the UID of an email given its X-GM-MSGID.
        Returns the UID as bytes, or None if not found.
        """
        self.connect()
        try:
            # Search for the UID corresponding to the X-GM-MSGID
            typ, data = self.connection.uid('SEARCH', None, f'X-GM-MSGID {gmail_id}')
            if typ == 'OK' and data[0]:
                # Return the last one if multiple (shouldn't be multiple for one ID usually)
                return data[0].split()[-1]
        except imaplib.IMAP4.error as e:
            print(f"Error searching for Gmail ID {gmail_id}: {e}")
        return None

    def apply_label(self, gmail_id: str, label: str):
        """
        Apply a label to the email using UID STORE +X-GM-LABELS.
        Accepts gmail_id (X-GM-MSGID).
        """
        self.connect()

        uid = self._search_by_gmail_id(gmail_id)
        if not uid:
            print(f"Could not find email with Gmail ID {gmail_id} to apply label.")
            return

        # Quote label if it has spaces
        label_to_send = f'"{label}"' if " " in label else label

        try:
            typ, data = self.connection.uid('STORE', uid, '+X-GM-LABELS', f'({label_to_send})')
            if typ != 'OK':
                print(f"Failed to apply label {label} to {gmail_id}: {data}")
        except Exception as e:
            print(f"Error applying label {label} to {gmail_id}: {e}")

    def remove_label(self, gmail_id: str, label: str):
        """
        Remove a label from the email using UID STORE -X-GM-LABELS.
        Accepts gmail_id (X-GM-MSGID).
        """
        self.connect()

        uid = self._search_by_gmail_id(gmail_id)
        if not uid:
            print(f"Could not find email with Gmail ID {gmail_id} to remove label.")
            return

        # Quote label if it has spaces
        label_to_send = f'"{label}"' if " " in label else label

        try:
            typ, data = self.connection.uid('STORE', uid, '-X-GM-LABELS', f'({label_to_send})')
            if typ != 'OK':
                print(f"Failed to remove label {label} from {gmail_id}: {data}")
        except Exception as e:
            print(f"Error removing label {label} from {gmail_id}: {e}")

    def fetch_email_by_gmail_id(self, gmail_id: str) -> Message:
        """
        Fetch the email content for a given X-GM-MSGID.
        """
        self.connect()
        uid = self._search_by_gmail_id(gmail_id)
        if not uid:
            return None

        try:
            # Fetch BODY.PEEK[] based on UID
            typ, msg_data = self.connection.uid('FETCH', uid, '(BODY.PEEK[])')
            if typ != 'OK':
                return None
            
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    return email.message_from_bytes(response_part[1])
        except Exception as e:
            print(f"Error fetching email {gmail_id}: {e}")
            return None

    def get_labels_for_emails(self, gmail_ids: List[str]) -> Dict[str, List[str]]:
        """
        Fetch current labels for a list of Gmail IDs (X-GM-MSGID).
        Returns {gmail_id: [label1, label2, ...]}
        """
        self.connect()
        results = {}

        # We need to find UIDs first.
        # This loop might be slow for many IDs but it's robust.
        uids_to_fetch = []
        uid_to_gmail_id = {}

        for gid in gmail_ids:
            uid = self._search_by_gmail_id(gid)
            if uid:
                uids_to_fetch.append(uid)
                # Keep mapping. uid is bytes from imaplib
                uid_to_gmail_id[uid] = gid
            else:
                # Email might be deleted or not found
                pass

        if not uids_to_fetch:
            return results

        # Fetch in batches
        BATCH_SIZE = 50
        for i in range(0, len(uids_to_fetch), BATCH_SIZE):
            batch_uids = uids_to_fetch[i:i+BATCH_SIZE]
            uid_str = b','.join(batch_uids)

            try:
                typ, data = self.connection.uid('FETCH', uid_str, '(X-GM-LABELS)')
                if typ != 'OK':
                    continue

                for response_part in data:
                    if isinstance(response_part, tuple):
                        metadata = response_part[0].decode('utf-8', errors='ignore')
                    else:
                        metadata = response_part.decode('utf-8', errors='ignore')

                    # Extract UID from metadata "123 (UID 456 X-GM-LABELS ...)"
                    uid_match = re.search(r'UID (\d+)', metadata)
                    if not uid_match:
                        continue

                    found_uid_str = uid_match.group(1)
                    found_uid_bytes = found_uid_str.encode('utf-8')

                    # Find corresponding Gmail ID
                    gid = uid_to_gmail_id.get(found_uid_bytes)
                    if not gid:
                        # Try finding by string if bytes key failed
                        gid = uid_to_gmail_id.get(found_uid_str)

                    if not gid:
                        continue

                    # Extract labels
                    labels = []
                    match = X_GM_LABELS_PATTERN.search(metadata)
                    if match:
                        labels_str = match.group(1)
                        # Parse labels taking quotes into account
                        token_pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"|([^"\s()]+)')
                        for m in token_pattern.finditer(labels_str):
                            if m.group(1):
                                labels.append(m.group(1).replace('\\"', '"').replace('\\\\', '\\'))
                            else:
                                labels.append(m.group(2))

                    results[gid] = labels
            except Exception as e:
                print(f"Error fetching labels batch: {e}")

        return results
