import imaplib
import email
import os
import ssl
import re
from email.message import Message
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

IMAP_SERVER = os.getenv("IMAP_SERVER") or "imap.gmail.com"

# Regex to match X-GM-LABELS content.
# Handles atoms (no quotes/parens) and quoted strings (with escaped quotes/backslashes).
# Group 1 captures the content inside the parentheses.
X_GM_LABELS_PATTERN = re.compile(r'X-GM-LABELS \(((?:[^()"]+|"(\\.|[^"\\])*")*)\)')
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
        Returns a list of (uid, email_message_object).
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

            # Fetch BODY.PEEK[] (full content) and X-GM-LABELS for all IDs in batch
            # PEEK prevents marking as \Seen implicitly by the fetch of body.
            typ, msg_data = self.connection.fetch(ids_str, '(BODY.PEEK[] X-GM-LABELS)')

            if typ != 'OK':
                continue

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    # response_part[0] is bytes, header line
                    # Example: b'1 (X-GM-LABELS (\\Inbox \\Important) BODY.PEEK[] {1234}'
                    metadata = response_part[0].decode('utf-8', errors='ignore')

                    # Extract sequence number (ID) from the beginning of metadata
                    # Format is: SEQ (ITEMS...)
                    seq_match = SEQ_PATTERN.match(metadata)
                    if not seq_match:
                        continue
                    e_id = seq_match.group(1).encode('utf-8')

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
                    results.append((e_id, msg))

        return results

    def apply_label(self, email_id: bytes, label: str):
        """
        Apply a label to the email using STORE +X-GM-LABELS.
        """
        self.connect()

        # Quote label if it has spaces
        label_to_send = f'"{label}"' if " " in label else label

        try:
            typ, data = self.connection.store(email_id, '+X-GM-LABELS', f'({label_to_send})')
            if typ != 'OK':
                print(f"Failed to apply label {label} to {email_id}: {data}")
        except Exception as e:
            print(f"Error applying label {label} to {email_id}: {e}")
