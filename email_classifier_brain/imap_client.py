import imaplib
import email
import os
import ssl
import re
from email.message import Message
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

IMAP_SERVER = "imap.gmail.com"

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

        for e_id in email_ids:
            # Fetch BODY.PEEK[] (full content) and X-GM-LABELS
            # PEEK prevents marking as \Seen implicitly by the fetch of body.
            typ, msg_data = self.connection.fetch(e_id, '(BODY.PEEK[] X-GM-LABELS)')
            if typ != 'OK':
                continue

            raw_email = None
            labels_str = ""

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    # response_part[0] is bytes, header line
                    # Example: b'1 (X-GM-LABELS (\\Inbox \\Important) BODY.PEEK[] {1234}'
                    metadata = response_part[0].decode('utf-8', errors='ignore')

                    # Extract content inside X-GM-LABELS (...)
                    match = re.search(r'X-GM-LABELS \((.*?)\)', metadata)
                    if match:
                        labels_str = match.group(1)

                    # The second element is the body content
                    raw_email = response_part[1]

            if raw_email:
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
