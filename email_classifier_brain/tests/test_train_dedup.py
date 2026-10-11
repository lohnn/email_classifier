
"""Fix B — train.py resolves cross-file duplicate (subject, body) examples.

One email appearing in two label jsonl files (or twice in one file) must
train under exactly ONE label: the latest-written copy, with a loud printed
warning. No duplicates must survive into the dataset.
"""

import json
import os
import sys
import tempfile
import time

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

os.environ["TRAINING_DATA_DIR"] = tempfile.mkdtemp()
os.environ["ADMIN_API_KEY"] = "testkey"
os.environ["TESTING"] = "true"

import main  # noqa: F401
from train import EmailSample, load_training_data


def _write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _entry(subject, body, label_key):
    return {
        "subject": subject,
        "body": body,
        "from": f"{label_key}@x.com",
        "to": "me@x.com",
        "cc": "",
        "mass_mail": False,
        "attachment_types": [],
    }


def test_cross_file_duplicate_resolves_to_latest(tmp_path, capsys):
    a = tmp_path / "A.jsonl"
    b = tmp_path / "B.jsonl"
    _write_jsonl(a, [_entry("Same mail", "Same body", "a")])
    _write_jsonl(b, [_entry("Same mail", "Same body", "b")])

    # Make B the latest write
    past = time.time() - 10000
    os.utime(a, (past, past))

    samples = load_training_data(str(tmp_path))

    keys = [(s.subject, s.body) for s in samples]
    assert keys.count(("Same mail", "Same body")) == 1  # exactly ONE survives
    assert all(s.label == "B" for s in samples if s.subject == "Same mail")

    out = capsys.readouterr().out
    assert "duplicate" in out.lower()
    assert "B" in out and "A" in out  # both labels named in the warning


def test_in_file_duplicate_resolves_to_one(tmp_path, capsys):
    f = tmp_path / "A.jsonl"
    _write_jsonl(f, [_entry("Same mail", "Same body", "a"),
                     _entry("Same mail", "Same body", "a")])
    samples = load_training_data(str(tmp_path))
    assert len(samples) == 1
    assert "duplicate" in capsys.readouterr().out.lower()


def test_distinct_examples_not_deduplicated(tmp_path, capsys):
    f = tmp_path / "A.jsonl"
    _write_jsonl(f, [_entry("One", "Body one", "a"),
                     _entry("Two", "Body two", "a")])
    samples = load_training_data(str(tmp_path))
    assert len(samples) == 2
    out = capsys.readouterr().out
    assert "duplicate" not in out.lower()


def test_resolve_duplicates_helper_keeps_higher_rank(tmp_path):
    from train import _resolve_duplicates
    fa = tmp_path / "A.jsonl"
    fb = tmp_path / "B.jsonl"
    fa.write_text("{}\n")
    fb.write_text("{}\n")
    mtimes = {str(fa): 1.0, str(fb): 2.0}
    old = EmailSample(subject="S", body="B", label="A", source_path=str(fa), source_line=1)
    new = EmailSample(subject="S", body="B", label="B", source_path=str(fb), source_line=1)

    kept, resolutions = _resolve_duplicates([old, new], mtimes)

    assert [s.label for s in kept] == ["B"]
    assert resolutions[0]["discarded"].label == "A"
    assert resolutions[0]["kept"].label == "B"
