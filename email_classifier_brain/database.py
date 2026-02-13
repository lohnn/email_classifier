import sqlite3
import datetime
import os
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
    """Initialize the database tables if they do not exist."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sender TEXT,
            subject TEXT,
            predicted_category TEXT,
            confidence_score REAL,
            is_read BOOLEAN DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def add_log(sender: str, subject: str, predicted_category: str, confidence_score: float) -> None:
    """Add a new classification log entry."""
    conn = get_db_connection()
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute('''
        INSERT INTO logs (timestamp, sender, subject, predicted_category, confidence_score, is_read)
        VALUES (?, ?, ?, ?, ?, 0)
    ''', (timestamp, sender, subject, predicted_category, confidence_score))
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

def ack_notifications(log_ids: Optional[List[int]] = None) -> None:
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
