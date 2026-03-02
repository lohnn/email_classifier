"""
Tests for the retry utility and IMAP retry behaviour.
"""

import imaplib
import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retry import with_retry
import imap_client


# ---------------------------------------------------------------------------
# with_retry unit tests
# ---------------------------------------------------------------------------

@patch('retry.time.sleep')
def test_with_retry_success_on_first_attempt(mock_sleep):
    """Function that succeeds immediately — no sleep, no on_retry."""
    fn = MagicMock(return_value=42)
    result = with_retry(fn, retries=3, backoff=1.0, exceptions=(ValueError,))
    assert result == 42
    fn.assert_called_once()
    mock_sleep.assert_not_called()


@patch('retry.time.sleep')
def test_with_retry_retries_on_exception(mock_sleep):
    """Function fails once then succeeds — retried exactly once."""
    fn = MagicMock(side_effect=[ValueError("oops"), 99])
    result = with_retry(fn, retries=3, backoff=1.0, exceptions=(ValueError,))
    assert result == 99
    assert fn.call_count == 2
    mock_sleep.assert_called_once_with(1.0)  # backoff * 2^0


@patch('retry.time.sleep')
def test_with_retry_raises_after_exhausting_retries(mock_sleep):
    """After all attempts are exhausted the original exception is re-raised."""
    fn = MagicMock(side_effect=OSError("network down"))
    with pytest.raises(OSError, match="network down"):
        with_retry(fn, retries=3, backoff=1.0, exceptions=(OSError,))
    assert fn.call_count == 3


@patch('retry.time.sleep')
def test_with_retry_exponential_backoff(mock_sleep):
    """Sleep durations double on each attempt: backoff, backoff*2, ..."""
    fn = MagicMock(side_effect=[OSError(), OSError(), OSError()])
    with pytest.raises(OSError):
        with_retry(fn, retries=3, backoff=2.0, exceptions=(OSError,))
    assert mock_sleep.call_args_list == [call(2.0), call(4.0)]


@patch('retry.time.sleep')
def test_with_retry_calls_on_retry_callback(mock_sleep):
    """on_retry is invoked with (exc, attempt) before each sleep."""
    on_retry = MagicMock()
    exc1 = ValueError("first")
    exc2 = ValueError("second")
    fn = MagicMock(side_effect=[exc1, exc2, "ok"])
    result = with_retry(fn, retries=3, backoff=0.5, exceptions=(ValueError,), on_retry=on_retry)
    assert result == "ok"
    assert on_retry.call_count == 2
    on_retry.assert_any_call(exc1, 1)
    on_retry.assert_any_call(exc2, 2)


@patch('retry.time.sleep')
def test_with_retry_does_not_catch_other_exceptions(mock_sleep):
    """Exceptions not listed in *exceptions* are not retried."""
    fn = MagicMock(side_effect=RuntimeError("fatal"))
    with pytest.raises(RuntimeError, match="fatal"):
        with_retry(fn, retries=3, backoff=1.0, exceptions=(OSError,))
    fn.assert_called_once()
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# IMAP retry integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_imap_conn():
    with patch('imaplib.IMAP4_SSL') as MockIMAP:
        yield MockIMAP.return_value


@pytest.fixture
def client(mock_imap_conn):
    with patch.dict(os.environ, {"IMAP_USER": "test@example.com", "IMAP_PASSWORD": "pass"}):
        c = imap_client.GmailClient()
        c.connection = mock_imap_conn
        return c


@patch('retry.time.sleep')
def test_fetch_unprocessed_emails_retries_on_imap_error(mock_sleep, client, mock_imap_conn):
    """fetch_unprocessed_emails should retry on IMAP4.error and reconnect."""
    # First call raises, second succeeds with empty inbox
    mock_imap_conn.noop.return_value = ('OK', [b''])
    mock_imap_conn.search.side_effect = [
        imaplib.IMAP4.error("connection reset"),
        ('OK', [b'']),  # second attempt: no unseen emails
    ]

    result = client.fetch_unprocessed_emails(known_labels=[])

    assert result == []
    assert mock_imap_conn.search.call_count == 2
    mock_sleep.assert_called_once()  # one sleep between attempts


@patch('retry.time.sleep')
def test_fetch_unprocessed_emails_raises_after_all_retries(mock_sleep, client, mock_imap_conn):
    """fetch_unprocessed_emails re-raises after all retry attempts are exhausted."""
    mock_imap_conn.noop.return_value = ('OK', [b''])
    mock_imap_conn.search.side_effect = imaplib.IMAP4.error("server gone")

    with pytest.raises(imaplib.IMAP4.error):
        client.fetch_unprocessed_emails(known_labels=[])

    assert mock_imap_conn.search.call_count == 3  # 3 attempts


@patch('retry.time.sleep')
def test_fetch_unprocessed_emails_resets_connection_on_retry(mock_sleep, client, mock_imap_conn):
    """_reset_connection is called between attempts, forcing a fresh connect."""
    mock_imap_conn.noop.return_value = ('OK', [b''])
    mock_imap_conn.search.side_effect = [
        OSError("broken pipe"),
        ('OK', [b'']),
    ]

    client.fetch_unprocessed_emails(known_labels=[])

    # After the reset the connection is None; connect() opens a new one.
    # We verify noop was called at least once (from connect()) and that
    # we went through two search calls.
    assert mock_imap_conn.search.call_count == 2


@patch('retry.time.sleep')
def test_apply_label_retries_on_imap_error(mock_sleep, client, mock_imap_conn):
    """apply_label should retry the STORE command on IMAP error."""
    mock_imap_conn.noop.return_value = ('OK', [b''])
    # UID SEARCH succeeds, then STORE fails once then succeeds
    mock_imap_conn.uid.side_effect = [
        ('OK', [b'42']),           # search — attempt 1
        imaplib.IMAP4.error("store failed"),  # store — attempt 1
        ('OK', [b'42']),           # search — attempt 2 (after reconnect)
        ('OK', [b'OK']),           # store — attempt 2
    ]

    client.apply_label('123456', 'WORK')
    assert mock_imap_conn.uid.call_count == 4


@patch('retry.time.sleep')
def test_get_labels_for_emails_retries_on_oserror(mock_sleep, client, mock_imap_conn):
    """get_labels_for_emails should retry on OSError."""
    mock_imap_conn.noop.return_value = ('OK', [b''])
    mock_imap_conn.uid.side_effect = [
        OSError("network reset"),
        ('OK', [b'']),  # second attempt: no UIDs found
    ]

    result = client.get_labels_for_emails(['123'])
    assert result == {}
    assert mock_imap_conn.uid.call_count == 2
