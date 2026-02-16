
import pytest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure we can import modules from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import imap_client

@pytest.fixture
def mock_imap_conn():
    with patch('imaplib.IMAP4_SSL') as MockIMAP:
        mock_conn = MockIMAP.return_value
        yield mock_conn

@pytest.fixture
def client(mock_imap_conn):
    # Setup env vars
    with patch.dict(os.environ, {"IMAP_USER": "test@example.com", "IMAP_PASSWORD": "password"}):
        client = imap_client.GmailClient()
        # Ensure connection uses our mock
        client.connection = mock_imap_conn
        return client

def test_fetch_unprocessed_emails_multiple(client, mock_imap_conn):
    # Setup search response: 3 emails
    mock_imap_conn.search.return_value = ('OK', [b'1 2 3'])

    # Setup fetch response
    # We need to simulate the structure returned by imaplib for 3 emails.
    # It's a list of parts.
    # Email 1
    header1 = b'1 (X-GM-LABELS (\\Inbox) BODY.PEEK[] {10}'
    body1 = b'Subject: One\r\n\r\nBody1'
    # Email 2 (with skipped label - user label typically has no backslash)
    header2 = b'2 (X-GM-LABELS (\\Inbox Skipped) BODY.PEEK[] {10}'
    body2 = b'Subject: Two\r\n\r\nBody2'
    # Email 3
    header3 = b'3 (X-GM-LABELS (\\Inbox) BODY.PEEK[] {10}'
    body3 = b'Subject: Three\r\n\r\nBody3'

    # The list contains tuples for message parts and bytes for closing parens ')'
    fetch_data = [
        (header1, body1), b')',
        (header2, body2), b')',
        (header3, body3), b')'
    ]
    mock_imap_conn.fetch.return_value = ('OK', fetch_data)

    # Call the method
    results = client.fetch_unprocessed_emails(known_labels=["Skipped"])

    # Verify fetch called with comma separated IDs
    mock_imap_conn.fetch.assert_called_with(b'1,2,3', '(BODY.PEEK[] X-GM-LABELS)')

    # Verify results
    # Should have 2 emails (email 2 skipped)
    assert len(results) == 2

    # Check first email
    eid1, msg1 = results[0]
    assert eid1 == b'1'
    assert msg1['Subject'] == 'One'

    # Check second email (which was ID 3)
    eid2, msg2 = results[1]
    assert eid2 == b'3'
    assert msg2['Subject'] == 'Three'

def test_fetch_unprocessed_emails_parentheses(client, mock_imap_conn):
    # Test for labels containing parentheses, which caused issues with simple regex
    mock_imap_conn.search.return_value = ('OK', [b'1'])

    # Label "My (Label)" which broke the old regex
    header = b'1 (X-GM-LABELS ("My (Label)") BODY.PEEK[] {10}'
    body = b'Subject: Parens\r\n\r\nBody'

    mock_imap_conn.fetch.return_value = ('OK', [(header, body), b')'])

    # We want to skip emails with "My (Label)"
    results = client.fetch_unprocessed_emails(known_labels=["My (Label)"])

    # Should be skipped
    assert len(results) == 0

def test_fetch_unprocessed_emails_empty(client, mock_imap_conn):
    # Setup search response: empty
    mock_imap_conn.search.return_value = ('OK', [b''])

    results = client.fetch_unprocessed_emails(known_labels=[])

    assert len(results) == 0
    # Fetch should not be called
    mock_imap_conn.fetch.assert_not_called()

def test_fetch_unprocessed_emails_single(client, mock_imap_conn):
    mock_imap_conn.search.return_value = ('OK', [b'10'])

    header = b'10 (X-GM-LABELS (\\Inbox) BODY.PEEK[] {10}'
    body = b'Subject: Single\r\n\r\nBody'
    mock_imap_conn.fetch.return_value = ('OK', [(header, body), b')'])

    results = client.fetch_unprocessed_emails(known_labels=[])

    assert len(results) == 1
    assert results[0][0] == b'10'
    mock_imap_conn.fetch.assert_called_with(b'10', '(BODY.PEEK[] X-GM-LABELS)')

def test_fetch_unprocessed_emails_batching(client, mock_imap_conn):
    # Test batching behavior when more than 50 emails are present
    # Create 60 IDs
    ids = [str(i).encode() for i in range(1, 61)]
    ids_str = b' '.join(ids)
    mock_imap_conn.search.return_value = ('OK', [ids_str])

    # Mock fetch side effect to handle calls
    # We expect 2 calls: one for 1-50, one for 51-60

    def fetch_side_effect(ids_bytes, query):
        requested = ids_bytes.split(b',')
        resp = []
        for rid in requested:
            header = f'{rid.decode()} (X-GM-LABELS (\\Inbox) BODY.PEEK[] {{10}}'.encode()
            body = b'Subject: Batch\r\n\r\nBody'
            resp.append((header, body))
            resp.append(b')')
        return ('OK', resp)

    mock_imap_conn.fetch.side_effect = fetch_side_effect

    results = client.fetch_unprocessed_emails(known_labels=[])

    assert len(results) == 60
    assert mock_imap_conn.fetch.call_count == 2

    # Verify the arguments of the calls
    call1_args = mock_imap_conn.fetch.call_args_list[0][0][0]
    call2_args = mock_imap_conn.fetch.call_args_list[1][0][0]

    # First batch should have 50 items
    assert len(call1_args.split(b',')) == 50
    # Second batch should have 10 items
    assert len(call2_args.split(b',')) == 10
