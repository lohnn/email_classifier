
"""Fix A — corrections are MOVE not append in the training data.

add_to_training_data(previous_category=...) must purge the (subject, body)
copy from the previous label's jsonl after placing it under the new label,
keep per-file dedup, preserve the DB un-stringifying behaviour for
attachment_types/mass_mail, and tolerate missing/corrupt old files.
"""

import json
import os
import sys
import tempfile

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

temp_data_dir = tempfile.mkdtemp()
os.environ["TRAINING_DATA_DIR"] = temp_data_dir
os.environ["ADMIN_API_KEY"] = "testkey"
os.environ["TESTING"] = "true"

import main  # noqa: F401  (consistent with the other test modules)
from jobs.training_data import add_to_training_data

import pytest


@pytest.fixture(autouse=True)
def clean_data_dir():
    # fresh temp training dir per test (main imported the module with the
    # module-level temp dir; point training_data at a per-test dir)
    import jobs.training_data as td
    per_test = tempfile.mkdtemp()
    original = td.TRAINING_DATA_DIR
    td.TRAINING_DATA_DIR = per_test
    yield per_test
    td.TRAINING_DATA_DIR = original


def _log_entry(subject="Invoice attached", body="Please find the invoice", sender="a@b.com", att='["PDF"]', mass=1):
    # Shaped like a database row: attachment_types JSON-string, mass_mail int
    return {
        "id": "g1",
        "subject": subject,
        "body": body,
        "sender": sender,
        "recipient": "me@me.com",
        "cc": "",
        "mass_mail": mass,
        "attachment_types": att,
    }


def _write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_move_removes_old_line(clean_data_dir):
    entry = {"subject": "Invoice attached", "body": "Body", "from": "a@b.com"}
    _write_jsonl(os.path.join(clean_data_dir, "NOISE.jsonl"), [entry])

    add_to_training_data(_log_entry(body="Body"), "URGENT", previous_category="NOISE")

    assert len(_read_jsonl(os.path.join(clean_data_dir, "URGENT.jsonl"))) == 1
    assert _read_jsonl(os.path.join(clean_data_dir, "NOISE.jsonl")) == []


def test_move_writes_unstringified_fields(clean_data_dir):
    add_to_training_data(_log_entry(), "URGENT", previous_category="NOISE")
    stored = _read_jsonl(os.path.join(clean_data_dir, "URGENT.jsonl"))[0]
    assert stored["attachment_types"] == ["PDF"]  # DB JSON-string decoded
    assert isinstance(stored["mass_mail"], bool) and stored["mass_mail"] is True  # DB int 1


def test_same_category_correction_keeps_file(clean_data_dir):
    entry = {"subject": "Invoice attached", "body": "Body", "from": "a@b.com"}
    _write_jsonl(os.path.join(clean_data_dir, "URGENT.jsonl"), [entry])

    add_to_training_data(_log_entry(body="Body"), "URGENT", previous_category="URGENT")

    lines = _read_jsonl(os.path.join(clean_data_dir, "URGENT.jsonl"))
    assert len(lines) == 1  # no purge of itself, no duplicate


def test_move_removes_all_copies_from_old(clean_data_dir):
    entry = {"subject": "Invoice attached", "body": "Body", "from": "a@b.com"}
    _write_jsonl(os.path.join(clean_data_dir, "NOISE.jsonl"), [entry, entry])
    add_to_training_data(_log_entry(body="Body"), "URGENT", previous_category="NOISE")
    assert _read_jsonl(os.path.join(clean_data_dir, "NOISE.jsonl")) == []


def test_move_tolerates_missing_old_file(clean_data_dir):
    add_to_training_data(_log_entry(), "URGENT", previous_category="GHOST")
    assert len(_read_jsonl(os.path.join(clean_data_dir, "URGENT.jsonl"))) == 1


def test_move_with_dedup_in_new_file_still_purges_old(clean_data_dir):
    existing = {"subject": "Invoice attached", "body": "Body", "from": "a@b.com"}
    _write_jsonl(os.path.join(clean_data_dir, "URGENT.jsonl"), [existing])
    _write_jsonl(os.path.join(clean_data_dir, "NOISE.jsonl"), [existing])

    add_to_training_data(_log_entry(body="Body"), "URGENT", previous_category="NOISE")

    # No duplicate added to URGENT, and the NOISE copy is still gone
    assert len(_read_jsonl(os.path.join(clean_data_dir, "URGENT.jsonl"))) == 1
    assert _read_jsonl(os.path.join(clean_data_dir, "NOISE.jsonl")) == []


def test_move_keeps_other_rows_in_old_file(clean_data_dir):
    other = {"subject": "Other email", "body": "Other body", "from": "c@d.com"}
    target = {"subject": "Invoice attached", "body": "Body", "from": "a@b.com"}
    _write_jsonl(os.path.join(clean_data_dir, "NOISE.jsonl"), [target, other])

    add_to_training_data(_log_entry(body="Body"), "URGENT", previous_category="NOISE")

    remaining = _read_jsonl(os.path.join(clean_data_dir, "NOISE.jsonl"))
    assert [e["subject"] for e in remaining] == ["Other email"]
