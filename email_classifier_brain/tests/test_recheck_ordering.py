
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


# ---------------------------------------------------------------------------
# Fresh-mail fast path (hardcoded, user spec): rows younger than 2h are
# re-eligible EVERY pass; after 2h the 12h suppression takes over, counting
# from the last sub-2h stamp. Other cohorts byte-for-byte unchanged.
# ---------------------------------------------------------------------------

def _insert_fresh(gid, age_h, last_recheck_age_h=None):
    now = datetime.now()
    _insert_log(
        gid,
        (now - timedelta(hours=age_h)).isoformat(),
        last_recheck=(now - timedelta(hours=last_recheck_age_h)).isoformat()
                     if last_recheck_age_h is not None else None,
    )


def test_fresh_row_under_2h_eligible_even_when_recently_stamped():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        original = database.DB_FILE
        database.DB_FILE = tmp.name
        try:
            database.init_db()
            _insert_fresh("a1", age_h=1, last_recheck_age_h=0.5)  # stamped 30min ago
            ids = [r["id"] for r in database.get_candidate_logs_for_recheck(limit=10)]
            assert "a1" in ids        # fast path fires despite fresh stamp
        finally:
            database.DB_FILE = original


def test_three_hour_old_row_recently_stamped_not_eligible():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        original = database.DB_FILE
        database.DB_FILE = tmp.name
        try:
            database.init_db()
            _insert_fresh("a2", age_h=3, last_recheck_age_h=2)   # stamped past the 2h crossing
            ids = [r["id"] for r in database.get_candidate_logs_for_recheck(limit=10)]
            assert "a2" not in ids   # 12h suppression counts from the last sub-2h stamp
        finally:
            database.DB_FILE = original


def test_never_checked_fresh_row_eligible():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        original = database.DB_FILE
        database.DB_FILE = tmp.name
        try:
            database.init_db()
            _insert_fresh("a3", age_h=0.2, last_recheck_age_h=None)
            ids = [r["id"] for r in database.get_candidate_logs_for_recheck(limit=10)]
            assert "a3" in ids
        finally:
            database.DB_FILE = original


def test_specialkost_replay_old_correction_still_suppressed():
    """User's literal case, modulo absolute times: classified 7.3h before the
    run, last stamped 6h before it, corrected 1.2h before it (invisible to
    the DB anyway) — all 3 intermediate passes must skip, without error."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        original = database.DB_FILE
        database.DB_FILE = tmp.name
        try:
            database.init_db()
            now = datetime.now()
            _insert_log("sp",
                        (now - timedelta(hours=7.3)).isoformat(),   # 12:31Z vs 19:50Z
                        last_recheck=(now - timedelta(hours=6)).isoformat())  # 13:50Z stamp
            ids = [r["id"] for r in database.get_candidate_logs_for_recheck(limit=10)]
            assert ids == []     # not due pre-fix; suppression intact post-fix
        finally:
            database.DB_FILE = original


def test_older_cohort_windows_unchanged():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        original = database.DB_FILE
        database.DB_FILE = tmp.name
        try:
            database.init_db()
            now = datetime.now()
            # 1-7d cohort: due when last_recheck >= 24h old, not when recent
            _insert_log("c_due", (now - timedelta(days=5)).isoformat(),
                        last_recheck=(now - timedelta(hours=25)).isoformat())
            _insert_log("c_fresh_stamp", (now - timedelta(days=5)).isoformat(),
                        last_recheck=(now - timedelta(hours=2)).isoformat())
            ids = [r["id"] for r in database.get_candidate_logs_for_recheck(limit=10)]
            assert "c_due" in ids
            assert "c_fresh_stamp" not in ids
        finally:
            database.DB_FILE = original
