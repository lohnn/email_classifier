import sqlite3
import datetime
import os
import json
from typing import Optional, List, Dict, Any
import config

# Ensure the database file is in the same directory as this script or appropriately located.
# Using relative path assuming execution from email_classifier_brain/ or similar.
DB_FILE = config.DB_PATH

def get_db_connection() -> sqlite3.Connection:
    """Create a database connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initialize the database tables."""
    conn = get_db_connection()
    c = conn.cursor()

    # check if 'logs' table exists and has 'id' as INTEGER (old schema)
    # If so, drop it.
    try:
        c.execute("PRAGMA table_info(logs)")
        columns = c.fetchall()
        if columns:
            # Check type of 'id' column
            id_col = next((col for col in columns if col['name'] == 'id'), None)
            if id_col and id_col['type'].upper() == 'INTEGER':
                print("Detected old schema (id is INTEGER). Dropping table 'logs' for migration to Gmail ID.")
                c.execute("DROP TABLE logs")
                columns = [] # Reset columns to trigger creation/check
    except Exception as e:
        print(f"Error checking schema: {e}")

    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            sender TEXT,
            recipient TEXT,
            cc TEXT,
            subject TEXT,
            body TEXT,
            mass_mail BOOLEAN,
            attachment_types TEXT,
            predicted_category TEXT,
            confidence_score REAL,
            corrected_category TEXT,
            is_read BOOLEAN DEFAULT 0,
            last_recheck TEXT,
            ambiguous_labels TEXT
        )
    ''')

    # Check for new columns and migrate if necessary
    try:
        c.execute("PRAGMA table_info(logs)")
        existing_cols = [col['name'] for col in c.fetchall()]

        if 'last_recheck' not in existing_cols:
            print("Migrating DB: Adding last_recheck column")
            c.execute("ALTER TABLE logs ADD COLUMN last_recheck TEXT")

        if 'ambiguous_labels' not in existing_cols:
            print("Migrating DB: Adding ambiguous_labels column")
            c.execute("ALTER TABLE logs ADD COLUMN ambiguous_labels TEXT")

    except Exception as e:
        print(f"Error migrating schema: {e}")

    conn.commit()
    conn.close()

def add_log(
    id: str,
    sender: str,
    recipient: str,
    subject: str,
    predicted_category: str,
    confidence_score: float,
    timestamp: Optional[datetime.datetime] = None,
    body: str = "",
    cc: str = "",
    mass_mail: bool = False,
    attachment_types: Optional[List[str]] = None
) -> None:
    """Add a new classification log entry."""
    conn = get_db_connection()
    c = conn.cursor()
    # Use provided timestamp or current time
    ts_str = timestamp.isoformat() if timestamp else datetime.datetime.now().isoformat()

    att_types_str = json.dumps(attachment_types or [])

    # We do NOT update last_recheck or ambiguous_labels on add_log (except maybe on insert default null)
    # If updating an existing log, we keep its recheck status unless explicitly reset logic is desired.
    # For now, preserve existing values on update is tricky with this ON CONFLICT logic unless we exclude them.
    # The current query overwrites everything else but leaves other cols alone if we don't mention them?
    # No, DO UPDATE SET must specify cols.
    # If we don't specify last_recheck, it keeps the old value (SQLite behavior for excluded columns not in SET).

    c.execute('''
        INSERT INTO logs (
            id, timestamp, sender, recipient, cc, subject, body,
            mass_mail, attachment_types, predicted_category, confidence_score, is_read
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(id) DO UPDATE SET
            timestamp=excluded.timestamp,
            sender=excluded.sender,
            recipient=excluded.recipient,
            cc=excluded.cc,
            subject=excluded.subject,
            body=excluded.body,
            mass_mail=excluded.mass_mail,
            attachment_types=excluded.attachment_types,
            predicted_category=excluded.predicted_category,
            confidence_score=excluded.confidence_score
    ''', (
        id, ts_str, sender, recipient, cc, subject, body,
        int(mass_mail), att_types_str, predicted_category, confidence_score
    ))
    conn.commit()

    conn.close()

def get_log_by_id(log_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a specific log entry by its ID."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM logs WHERE id = ?", (log_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def update_log_correction(log_id: str, corrected_category: str) -> None:
    """Update a log entry with the corrected category."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        UPDATE logs
        SET corrected_category = ?
        WHERE id = ?
    ''', (corrected_category, log_id))
    conn.commit()
    conn.close()

def get_stats(start_time: Optional[datetime.datetime] = None, end_time: Optional[datetime.datetime] = None) -> Dict[str, int]:
    """
    Get classification statistics (counts per category).
    Optionally filtered by time range.
    """
    conn = get_db_connection()
    c = conn.cursor()

    query = "SELECT predicted_category, COUNT(*) as count FROM logs"
    params = []

    conditions = []
    if start_time:
        conditions.append("timestamp >= ?")
        params.append(start_time.isoformat())
    if end_time:
        conditions.append("timestamp <= ?")
        params.append(end_time.isoformat())

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " GROUP BY predicted_category"

    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    return {row['predicted_category']: row['count'] for row in rows}

def get_unread_notifications() -> List[Dict[str, Any]]:
    """Get all unread logs."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM logs WHERE is_read = 0 ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def ack_notifications(log_ids: Optional[List[str]] = None) -> None:
    """
    Mark notifications as read.
    If log_ids is provided, mark only those.
    If log_ids is None (or empty), mark ALL unread notifications.
    """
    conn = get_db_connection()
    c = conn.cursor()
    if log_ids:
        placeholders = ','.join('?' for _ in log_ids)
        c.execute(f"UPDATE logs SET is_read = 1 WHERE id IN ({placeholders})", log_ids)
    else:
        c.execute("UPDATE logs SET is_read = 1 WHERE is_read = 0")
    conn.commit()
    conn.close()

def pop_unread_notifications() -> List[Dict[str, Any]]:
    """Get all unread notifications and mark them as read immediately."""
    # Reuse existing functions to avoid duplication
    unread = get_unread_notifications()
    if unread:
        ids = [str(row['id']) for row in unread]
        ack_notifications(ids)
    return unread

def get_read_notifications(start_time: datetime.datetime, end_time: datetime.datetime) -> List[Dict[str, Any]]:
    """Get read notifications within a time range."""
    conn = get_db_connection()
    c = conn.cursor()

    c.execute('''
        SELECT * FROM logs
        WHERE is_read = 1
        AND timestamp >= ?
        AND timestamp <= ?
        ORDER BY timestamp DESC
    ''', (start_time.isoformat(), end_time.isoformat()))

    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_logs_for_reclassification() -> List[Dict[str, Any]]:
    """Get all logs that haven't been manually corrected, for re-evaluation."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM logs WHERE corrected_category IS NULL")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_candidate_logs_for_recheck(limit: int = 200) -> List[Dict[str, Any]]:
    """
    Get logs eligible for re-check based on the gliding scale logic.
    Prioritizes newer emails.
    """
    conn = get_db_connection()
    c = conn.cursor()

    now = datetime.datetime.now()

    # Thresholds for 'timestamp'
    t_1d = (now - datetime.timedelta(days=1)).isoformat()
    t_7d = (now - datetime.timedelta(days=7)).isoformat()
    t_30d = (now - datetime.timedelta(days=30)).isoformat()

    # Thresholds for 'last_recheck'
    r_12h = (now - datetime.timedelta(hours=12)).isoformat()
    r_24h = (now - datetime.timedelta(hours=24)).isoformat()
    r_7d = (now - datetime.timedelta(days=7)).isoformat()
    r_30d = (now - datetime.timedelta(days=30)).isoformat()

    # Logic:
    # 1. < 1 day old: recheck if last_recheck < 12h ago (or null)
    # 2. 1-7 days old: recheck if last_recheck < 24h ago (or null)
    # 3. 7-30 days old: recheck if last_recheck < 7d ago (or null)
    # 4. > 30 days old: recheck if last_recheck < 30d ago (or null)

    query = '''
        SELECT * FROM logs
        WHERE
            -- Case 1: < 1 day old
            (timestamp > ? AND (last_recheck IS NULL OR last_recheck < ?))
            OR
            -- Case 2: 1-7 days old
            (timestamp <= ? AND timestamp > ? AND (last_recheck IS NULL OR last_recheck < ?))
            OR
            -- Case 3: 7-30 days old
            (timestamp <= ? AND timestamp > ? AND (last_recheck IS NULL OR last_recheck < ?))
            OR
            -- Case 4: > 30 days old
            (timestamp <= ? AND (last_recheck IS NULL OR last_recheck < ?))
        ORDER BY timestamp DESC
        LIMIT ?
    '''

    params = (
        t_1d, r_12h,           # Case 1
        t_1d, t_7d, r_24h,     # Case 2
        t_7d, t_30d, r_7d,     # Case 3
        t_30d, r_30d,          # Case 4
        limit
    )

    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    return [dict(row) for row in rows]

def update_recheck_status(log_id: str, ambiguous_labels: Optional[List[str]] = None) -> None:
    """
    Update the last_recheck timestamp and ambiguous_labels for a log.
    If ambiguous_labels is None or empty, it sets the column to NULL.
    """
    conn = get_db_connection()
    c = conn.cursor()

    now_str = datetime.datetime.now().isoformat()

    amb_str = json.dumps(ambiguous_labels) if ambiguous_labels else None

    c.execute('''
        UPDATE logs
        SET last_recheck = ?,
            ambiguous_labels = ?
        WHERE id = ?
    ''', (now_str, amb_str, log_id))

    conn.commit()
    conn.close()

def get_ambiguous_logs() -> List[Dict[str, Any]]:
    """Get logs that have been flagged as ambiguous (multiple trained labels found)."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM logs WHERE ambiguous_labels IS NOT NULL ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]
