
"""Fix D — get_candidate_logs_for_recheck ordering.

Among due rows: ambiguous first, then most-recent label-touch
(last_label_change, falling back to timestamp), then
least-recently-checked. Old rows must never be starved by the cap.
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

os.environ["TRAINING_DATA_DIR"] = tempfile.mkdtemp()
os.environ["ADMIN_API_KEY"] = "testkey"
os.environ["TESTING"] = "true"

import main  # noqa: F401
import database


def _insert_log(gid, timestamp, last_recheck=None, ambiguous=None, last_label_change=None):
    now = datetime.now()
    t_35d = (now - timedelta(days=35)).isoformat()
    conn = database.get_db_connection()
    c = conn.cursor()
    c.execute(
        '''INSERT INTO logs (id, timestamp, subject, body, predicted_category,
                             confidence_score, last_recheck, ambiguous_labels,
                             last_label_change)
           VALUES (?, ?, ?, ?, 'NOISE', 0.9, ?, ?, ?)''',
        (gid, timestamp, f"S {gid}", f"B {gid}", last_recheck or t_35d, ambiguous, last_label_change),
    )
    conn.commit()
    conn.close()


def test_ordering_ambiguous_then_label_touch_then_round_robin():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        original = database.DB_FILE
        database.DB_FILE = tmp.name
        try:
            database.init_db()
            now = datetime.now()
            _insert_log("amb", (now - timedelta(days=40)).isoformat(),
                        ambiguous='["NOISE","URGENT"]')
            _insert_log("lab", (now - timedelta(days=40)).isoformat(),
                        last_label_change=(now - timedelta(days=2)).isoformat())
            _insert_log("new", (now - timedelta(days=5)).isoformat(), last_recheck=None)
            _insert_log("plain", (now - timedelta(days=40)).isoformat())

            rows = database.get_candidate_logs_for_recheck(limit=10)
            ids = [r["id"] for r in rows]

            # All four must be due (no cap starvation)
            assert set(ids) == {"amb", "lab", "new", "plain"}
            # Priority order
            assert ids == ["amb", "lab", "new", "plain"]
        finally:
            database.DB_FILE = original
