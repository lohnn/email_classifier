
"""Fix C — unknown Gmail labels become observable candidate-new-category events.

The recheck job must record non-VERIFIED, non-UNSURE labels that the model
never trained on (candidate_labels on the log), the API must expose them,
and accept/discard endpoints must resolve them. Known labels and system
labels must never produce candidate events.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

temp_data_dir = tempfile.mkdtemp()
os.environ["TRAINING_DATA_DIR"] = temp_data_dir
os.environ["ADMIN_API_KEY"] = "testkey"
os.environ["TESTING"] = "true"

import main  # noqa: F401
from main import app
import database


@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("database.DB_FILE", tmp.name), \
             patch("jobs.training_data.TRAINING_DATA_DIR", temp_data_dir):
            database.init_db()
            with TestClient(app) as c:
                yield c


@pytest.fixture
def mock_classify():
    mock = MagicMock()
    with patch("api.routes.admin.classify", mock), \
         patch("jobs.correction.classify", mock):
        yield mock


@pytest.fixture
def mock_imap():
    mock_client = MagicMock()
    with patch("imap_client.gmail_client", mock_client), \
         patch("imap_client.GmailClient", return_value=mock_client):
        yield mock_client


def _seed_log(gid, predicted="NOISE"):
    database.add_log(
        id=gid,
        sender="s@x.com",
        recipient="me@x.com",
        subject=f"Seed {gid}",
        predicted_category=predicted,
        confidence_score=0.9,
        body=f"Body {gid}",
        cc="",
        mass_mail=False,
        attachment_types=[],
    )


def _labels_map(mapping):
    def fake(gmail_ids):
        return {g: mapping[g] for g in gmail_ids if g in mapping}
    return fake


def test_unknown_label_recorded_as_candidate(client, mock_classify, mock_imap):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    mock_imap.get_labels_for_emails.side_effect = _labels_map({"g1": ["PURCHASES"]})

    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")

    row = database.get_log_by_id("g1")
    assert json.loads(row["candidate_labels"]) == ["PURCHASES"]
    assert row["ambiguous_labels"] is None


def test_verifed_and_unsure_labels_never_candidates(client, mock_classify, mock_imap):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    mock_imap.get_labels_for_emails.side_effect = _labels_map(
        {"g1": ["VERIFIED", "UNSURE_CLASSIFICATION"]}
    )

    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")

    row = database.get_log_by_id("g1")
    assert row["candidate_labels"] is None


def test_known_label_flow_unchanged_no_candidate(client, mock_classify, mock_imap):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    mock_imap.get_labels_for_emails.side_effect = _labels_map({"g1": ["URGENT"]})

    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")

    row = database.get_log_by_id("g1")
    assert row["candidate_labels"] is None
    assert row["corrected_category"] == "URGENT"  # correction recorded as before
    assert os.path.exists(os.path.join(temp_data_dir, "URGENT.jsonl"))


def test_correction_move_removes_previous_training_copy(client, mock_classify, mock_imap):
    """End-to-end: a per-file NOISE copy must vanish when the recheck moves the email to URGENT."""
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    entry = {"subject": "Seed g1", "body": "Body g1", "from": "s@x.com"}
    with open(os.path.join(temp_data_dir, "NOISE.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    mock_imap.get_labels_for_emails.side_effect = _labels_map({"g1": ["URGENT"]})

    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")

    with open(os.path.join(temp_data_dir, "NOISE.jsonl"), encoding="utf-8") as f:
        assert not [l for l in f if l.strip()]  # old copy purged


def test_candidate_listing_and_accept_endpoint(client, mock_classify, mock_imap):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    database.update_recheck_status("g1", ambiguous_labels=None, candidate_labels=["PURCHASES"])

    response = client.get("/logs/candidate-categories", headers={"X-API-Key": "testkey"})
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1 and items[0]["id"] == "g1"

    response = client.post(
        "/logs/candidate-categories/g1/accept",
        json={"category": "PURCHASES"},
        headers={"X-API-Key": "testkey"},
    )
    assert response.status_code == 200
    assert response.json()["category"] == "PURCHASES"

    # Teaching file created, DB corrected, candidate cleared
    assert os.path.exists(os.path.join(temp_data_dir, "PURCHASES.jsonl"))
    assert database.get_log_by_id("g1")["corrected_category"] == "PURCHASES"
    assert database.get_log_by_id("g1")["candidate_labels"] is None
    assert mock_imap.apply_label.call_count >= 2  # category + VERIFIED

    # Listing no longer contains it
    response = client.get("/logs/candidate-categories", headers={"X-API-Key": "testkey"})
    assert response.json() == []


def test_accept_rejects_path_traversal_and_known_categories(client, mock_classify):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT"]
    _seed_log("g1")
    database.update_recheck_status("g1", candidate_labels=["../evil"])

    response = client.post(
        "/logs/candidate-categories/g1/accept",
        json={"category": "../evil"},
        headers={"X-API-Key": "testkey"},
    )
    assert response.status_code == 400

    response = client.post(
        "/logs/candidate-categories/g1/accept",
        json={"category": "NOISE"},
        headers={"X-API-Key": "testkey"},
    )
    assert response.status_code == 400


def test_discard_endpoint_clears_flag(client):
    _seed_log("g1")
    database.update_recheck_status("g1", candidate_labels=["PURCHASES"])

    response = client.post(
        "/logs/candidate-categories/g1/discard", headers={"X-API-Key": "testkey"}
    )
    assert response.status_code == 200
    assert database.get_log_by_id("g1")["candidate_labels"] is None

    response = client.post(
        "/logs/candidate-categories/g1/discard", headers={"X-API-Key": "testkey"}
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Wave G — the user's literal workflow: "add the correct label in Gmail,
# then remove/archive the message", plus UNSURE-cleanup symmetry.
# The out-of-inbox label *reading* itself is covered in test_imap_supplement;
# here the recheck is fed the labels that reader produces.
# ---------------------------------------------------------------------------

def test_user_workflow_add_label_archive_then_recheck_moves_training_data(
    client, mock_classify, mock_imap
):
    """User adds FOCUS to a NOISE-classified (and UNSURE-flagged) email and
    archives it. The recheck must: resolve the correction, MOVE the training
    example out of NOISE.jsonl into FOCUS.jsonl, clean up the classifier's
    UNSURE flag and the old category label, apply VERIFIED, and never touch
    the user's own labels or archive state."""
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT", "FOCUS", "REFERENCE"]
    _seed_log("g1", predicted="NOISE")
    example = {"subject": "Seed g1", "body": "Body g1", "from": "s@x.com"}
    with open(os.path.join(temp_data_dir, "NOISE.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(example) + "\n")

    # Post-archive labels as the (supplemented) label reader returns them
    mock_imap.get_labels_for_emails.side_effect = _labels_map(
        {"g1": ["NOISE", "FOCUS", "UNSURE_CLASSIFICATION"]}
    )

    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")

    # Training data MOVED, not appended
    reads = {}
    for label in ("NOISE", "FOCUS", "URGENT", "REFERENCE"):
        path = os.path.join(temp_data_dir, f"{label}.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                reads[label] = [json.loads(l) for l in f if l.strip()]
    assert reads.get("NOISE") == []                     # old line GONE
    assert [e["subject"] for e in reads.get("FOCUS", [])] == ["Seed g1"]

    row = database.get_log_by_id("g1")
    assert row["corrected_category"] == "FOCUS"         # resolved, NOT ambiguous
    assert row["ambiguous_labels"] is None
    assert row["candidate_labels"] is None              # UNSURE + VERIFIED never candidates

    # Gmail tail: old category removed, UNSURE flag removed, VERIFIED applied;
    # the user's FOCUS label is theirs — the brain must not re-apply it.
    removed = [c.args for c in mock_imap.remove_label.call_args_list]
    applied = [c.args for c in mock_imap.apply_label.call_args_list]
    assert ("g1", "NOISE") in removed
    assert ("g1", "UNSURE_CLASSIFICATION") in removed
    assert all("FOCUS" not in args for args in applied)          # no re-applying the user's label
    assert ("g1", "VERIFIED") in applied


def test_prediction_plus_user_label_plus_unsure_resolves_not_ambiguous(
    client, mock_classify, mock_imap
):
    """[predicted-P, user-added-R, UNSURE_CLASSIFICATION] must resolve to R:
    the known-categories filter drops UNSURE (and VERIFIED) BEFORE the
    exactly-2-trained-labels resolution — the whole user workflow hinges on
    this not punting into ambiguous_labels."""
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT", "FOCUS", "REFERENCE"]
    _seed_log("g1", predicted="URGENT")
    mock_imap.get_labels_for_emails.side_effect = _labels_map(
        {"g1": ["URGENT", "FOCUS", "UNSURE_CLASSIFICATION"]}
    )

    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")

    row = database.get_log_by_id("g1")
    assert row["corrected_category"] == "FOCUS"
    assert row["ambiguous_labels"] is None


def test_cleanup_removes_unsure_but_never_user_labels(client, mock_classify, mock_imap):
    """A correction that resolves while the user's own unknown tag sits on the
    email: the UNSURE flag is cleared, the user's label is untouched (it is
    recorded as a candidate new category instead)."""
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT", "FOCUS", "REFERENCE"]
    _seed_log("g1", predicted="NOISE")
    mock_imap.get_labels_for_emails.side_effect = _labels_map(
        {"g1": ["FOCUS", "MY_OWN_TAG", "UNSURE_CLASSIFICATION"]}
    )

    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")

    row = database.get_log_by_id("g1")
    assert row["corrected_category"] == "FOCUS"
    assert json.loads(row["candidate_labels"]) == ["MY_OWN_TAG"]

    removed = [c.args for c in mock_imap.remove_label.call_args_list]
    assert ("g1", "UNSURE_CLASSIFICATION") in removed
    assert all("MY_OWN_TAG" not in args for args in removed)


def test_admin_correct_clears_unsure_flag(client, mock_classify, mock_imap):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT", "FOCUS", "REFERENCE"]
    _seed_log("g1", predicted="NOISE")

    response = client.post(
        "/logs/g1/correction",
        json={"corrected_category": "FOCUS"},
        headers={"X-API-Key": "testkey"},
    )
    assert response.status_code == 200

    removed = [c.args for c in mock_imap.remove_label.call_args_list]
    assert ("g1", "NOISE") in removed
    assert ("g1", "UNSURE_CLASSIFICATION") in removed
    applied = [c.args for c in mock_imap.apply_label.call_args_list]
    assert ("g1", "FOCUS") in applied
    assert ("g1", "VERIFIED") in applied


# ---------------------------------------------------------------------------
# Verified-confirm gesture: "the predicted label was RIGHT" = add VERIFIED.
# The confirmed example trains under the predicted category, the row freezes
# against reclassify, and the leftover UNSURE flag is stripped.
# ---------------------------------------------------------------------------

def _confirm_with_labels(client, mock_classify, mock_imap, labels):
    mock_classify.get_available_categories.return_value = ["NOISE", "URGENT", "FOCUS", "REFERENCE"]
    _seed_log("g1", predicted="URGENT")
    if labels is not None:
        mock_imap.get_labels_for_emails.side_effect = _labels_map({"g1": labels})
    from jobs.correction import check_corrections_job
    check_corrections_job(limit=10, trigger="manual")


def test_verified_confirm_writes_example_and_strips_unsure(client, mock_classify, mock_imap):
    _confirm_with_labels(client, mock_classify, mock_imap,
                         ["URGENT", "VERIFIED", "UNSURE_CLASSIFICATION"])

    # Confirmed example trains under the predicted category
    with open(os.path.join(temp_data_dir, "URGENT.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert [r["subject"] for r in rows] == ["Seed g1"]

    row = database.get_log_by_id("g1")
    assert row["corrected_category"] == "URGENT"

    removed = [c.args for c in mock_imap.remove_label.call_args_list]
    assert ("g1", "UNSURE_CLASSIFICATION") in removed
    # No other label surgery in the verified path: nothing is re-applied
    assert mock_imap.apply_label.call_count == 0


def test_verified_confirm_without_unsure_makes_only_the_noop_removal(client, mock_classify, mock_imap):
    _confirm_with_labels(client, mock_classify, mock_imap, ["URGENT", "VERIFIED"])

    with open(os.path.join(temp_data_dir, "URGENT.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert [r["subject"] for r in rows] == ["Seed g1"]
    assert database.get_log_by_id("g1")["corrected_category"] == "URGENT"
    # remove_label no-ops on an absent label — exactly one removal attempt,
    # and it targets only the classifier's own flag
    removed = [c.args for c in mock_imap.remove_label.call_args_list]
    assert removed == [("g1", "UNSURE_CLASSIFICATION")]


def test_confirmed_row_excluded_from_reclassify(client, mock_classify, mock_imap):
    _seed_log("g2", predicted="NOISE")  # stays unconfirmed
    _confirm_with_labels(client, mock_classify, mock_imap,
                         ["URGENT", "VERIFIED", "UNSURE_CLASSIFICATION"])

    eligible_ids = [r["id"] for r in database.get_logs_for_reclassification(limit=100)]
    assert "g1" not in eligible_ids       # confirmed row is frozen
    assert "g2" in eligible_ids           # uncorrected rows remain eligible
