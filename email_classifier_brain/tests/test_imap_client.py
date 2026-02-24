
import pytest
from unittest.mock import MagicMock, patch, call
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


def _make_metadata_response(seq_id, gmail_id, labels_str="\\Inbox"):
    """Helper to build a Phase 1 metadata response line."""
    return f'{seq_id} (X-GM-LABELS ({labels_str}) X-GM-MSGID {gmail_id})'.encode()


def _make_body_response(seq_id, gmail_id, subject="Test", body="Body"):
    """Helper to build a Phase 2 body response tuple."""
    header = f'{seq_id} (BODY[] {{100}} X-GM-MSGID {gmail_id})'.encode()
    raw_email = f'Subject: {subject}\r\n\r\n{body}'.encode()
    return (header, raw_email)


def test_fetch_unprocessed_emails_multiple(client, mock_imap_conn):
    """Test that emails with known labels are filtered out."""
    mock_imap_conn.search.return_value = ('OK', [b'1 2 3'])

    # Phase 1: metadata fetch — IMAP returns in ascending order (1, 2, 3)
    # even though we request reversed (3, 2, 1)
    phase1_data = [
        _make_metadata_response(1, '1001', '\\Inbox'),
        _make_metadata_response(2, '1002', '\\Inbox Skipped'),
        _make_metadata_response(3, '1003', '\\Inbox'),
    ]

    # Phase 2: body fetch for qualifying IDs (3, 1 — newest first)
    phase2_data = [
        _make_body_response(1, '1001', 'One', 'Body1'), b')',
        _make_body_response(3, '1003', 'Three', 'Body3'), b')',
    ]

    mock_imap_conn.fetch.side_effect = [
        ('OK', phase1_data),  # Phase 1
        ('OK', phase2_data),  # Phase 2
    ]

    results = client.fetch_unprocessed_emails(known_labels=["Skipped"])

    assert len(results) == 2

    # Should be newest-first: 3 (Three), then 1 (One)
    eid1, msg1 = results[0]
    assert eid1 == '1003'
    assert msg1['Subject'] == 'Three'

    eid2, msg2 = results[1]
    assert eid2 == '1001'
    assert msg2['Subject'] == 'One'


def test_fetch_unprocessed_emails_parentheses(client, mock_imap_conn):
    """Test labels containing parentheses are correctly matched."""
    mock_imap_conn.search.return_value = ('OK', [b'1'])

    phase1_data = [
        f'1 (X-GM-LABELS ("My (Label)") X-GM-MSGID 1001)'.encode(),
    ]

    mock_imap_conn.fetch.side_effect = [('OK', phase1_data)]

    results = client.fetch_unprocessed_emails(known_labels=["My (Label)"])

    # Should be skipped — no Phase 2 call
    assert len(results) == 0
    assert mock_imap_conn.fetch.call_count == 1  # only Phase 1


def test_fetch_unprocessed_emails_empty(client, mock_imap_conn):
    mock_imap_conn.search.return_value = ('OK', [b''])

    results = client.fetch_unprocessed_emails(known_labels=[])

    assert len(results) == 0
    mock_imap_conn.fetch.assert_not_called()


def test_fetch_unprocessed_emails_single(client, mock_imap_conn):
    mock_imap_conn.search.return_value = ('OK', [b'10'])

    phase1_data = [_make_metadata_response(10, '2001', '\\Inbox')]
    phase2_data = [_make_body_response(10, '2001', 'Single', 'Body'), b')']

    mock_imap_conn.fetch.side_effect = [
        ('OK', phase1_data),
        ('OK', phase2_data),
    ]

    results = client.fetch_unprocessed_emails(known_labels=[])

    assert len(results) == 1
    assert results[0][0] == '2001'


def test_fetch_unprocessed_emails_newest_first_with_limit(client, mock_imap_conn):
    """Core regression test: with a limit, the NEWEST qualifying emails
    must be returned, not the oldest."""
    # 5 unseen emails
    mock_imap_conn.search.return_value = ('OK', [b'1 2 3 4 5'])

    # IMAP returns metadata in ascending order regardless of request
    phase1_data = [
        _make_metadata_response(1, '1001', '\\Inbox'),
        _make_metadata_response(2, '1002', '\\Inbox'),
        _make_metadata_response(3, '1003', '\\Inbox'),
        _make_metadata_response(4, '1004', '\\Inbox'),
        _make_metadata_response(5, '1005', '\\Inbox'),
    ]

    # Phase 2: only the 2 newest (5, 4)
    phase2_data = [
        _make_body_response(4, '1004', 'Four', 'Body4'), b')',
        _make_body_response(5, '1005', 'Five', 'Body5'), b')',
    ]

    mock_imap_conn.fetch.side_effect = [
        ('OK', phase1_data),
        ('OK', phase2_data),
    ]

    results = client.fetch_unprocessed_emails(known_labels=[], limit=2)

    assert len(results) == 2
    # Newest first: 5 then 4
    assert results[0][0] == '1005'
    assert results[0][1]['Subject'] == 'Five'
    assert results[1][0] == '1004'
    assert results[1][1]['Subject'] == 'Four'


def test_fetch_unprocessed_emails_batching(client, mock_imap_conn):
    """Test batching behavior when more than BATCH_SIZE emails are present."""
    # Create 60 IDs
    ids = [str(i).encode() for i in range(1, 61)]
    ids_str = b' '.join(ids)
    mock_imap_conn.search.return_value = ('OK', [ids_str])

    def fetch_side_effect(ids_bytes, query):
        requested = ids_bytes.split(b',')
        resp = []
        if 'X-GM-LABELS' in query:
            # Phase 1: metadata
            for rid in requested:
                rid_int = int(rid)
                resp.append(_make_metadata_response(rid_int, rid_int + 1000, '\\Inbox'))
        else:
            # Phase 2: bodies
            for rid in requested:
                rid_int = int(rid)
                resp.append(_make_body_response(rid_int, rid_int + 1000, f'Subj{rid_int}', 'Body'))
                resp.append(b')')
        return ('OK', resp)

    mock_imap_conn.fetch.side_effect = fetch_side_effect

    results = client.fetch_unprocessed_emails(known_labels=[])

    assert len(results) == 60
    # Phase 1: 2 batches (50 + 10), Phase 2: 2 batches (50 + 10)
    assert mock_imap_conn.fetch.call_count == 4

    # First result should be the highest ID (newest first)
    assert results[0][0] == '1060'
    assert results[-1][0] == '1001'
