import sqlite3
import datetime
import os
import json
from typing import Optional, List, Dict, Any

# Ensure the database file is in the same directory as this script or appropriately located.
# Using relative path assuming execution from email_classifier_brain/ or similar.
DB_FILE = os.path.join(os.path.dirname(__file__), "email_history.db")

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
            is_read BOOLEAN DEFAULT 0
        )
    ''')
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

    try:
        c.execute('''
            INSERT INTO logs (
                id, timestamp, sender, recipient, cc, subject, body,
                mass_mail, attachment_types, predicted_category, confidence_score, is_read
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        ''', (
            id, ts_str, sender, recipient, cc, subject, body,
            int(mass_mail), att_types_str, predicted_category, confidence_score
        ))
        conn.commit()
    except sqlite3.IntegrityError:
        # If ID exists, we could update it, or just ignore. 
        # For now, let's update the prediction if it's a re-run? 
        # But add_log roughly implies new entry. Let's just print/pass for now or update?
        # User wants "re-classify", so we probably update.
        # Let's do an UPSERT via REPLACE or explicitly UPDATE.
        # But this function is "add_log".
        # Let's keep it as INSERT and fail (or ignore) if exists, enforcing uniqueness.
        # The reclassify logic handles updates via a different function usually, 
        # but let's allow "add_log" to overwrite if we re-process an email.
        print(f"Log with ID {id} already exists. Updating...")
        c.execute('''
            UPDATE logs SET
                timestamp=?, sender=?, recipient=?, cc=?, subject=?, body=?,
                mass_mail=?, attachment_types=?, predicted_category=?, confidence_score=?
            WHERE id=?
        ''', (
            ts_str, sender, recipient, cc, subject, body,
            int(mass_mail), att_types_str, predicted_category, confidence_score,
            id
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
