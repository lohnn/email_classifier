#!/usr/bin/env python3
"""
clean_training_data.py — One-Time Training Data Cleanup
========================================================

Reads all .jsonl files in the training data directory, applies
clean_subject() and clean_body() to each entry, and writes them
back in place.

Usage:
    python clean_training_data.py
"""

import json
import os
import sys

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from config import TRAINING_DATA_DIR, clean_subject, clean_body


def clean_jsonl_file(filepath: str) -> tuple[int, int]:
    """
    Clean a single JSONL file in place.

    Returns (total_entries, cleaned_entries).
    """
    entries = []
    cleaned = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping invalid JSON line: {e}")
                entries.append(line)  # Keep original
                continue

            original_subject = entry.get("subject", "")
            original_body = entry.get("body", "")

            new_subject = clean_subject(original_subject)
            new_body = clean_body(original_body)

            changed = False
            if new_subject != original_subject:
                entry["subject"] = new_subject
                changed = True
            if new_body != original_body:
                entry["body"] = new_body
                changed = True

            if changed:
                cleaned += 1

            entries.append(json.dumps(entry, ensure_ascii=False))

    # Write back
    with open(filepath, "w", encoding="utf-8") as f:
        for entry_line in entries:
            f.write(entry_line + "\n")

    return len(entries), cleaned


def main():
    data_dir = TRAINING_DATA_DIR
    if not os.path.isdir(data_dir):
        print(f"Training data directory not found: {data_dir}")
        sys.exit(1)

    print(f"Cleaning training data in: {data_dir}")
    total_files = 0
    total_entries = 0
    total_cleaned = 0

    for dirpath, _, filenames in sorted(os.walk(data_dir)):
        for filename in sorted(filenames):
            if not filename.endswith(".jsonl"):
                continue

            filepath = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(filepath, data_dir)
            entries, cleaned = clean_jsonl_file(filepath)

            status = f"  {cleaned} cleaned" if cleaned > 0 else "  (already clean)"
            print(f"  {rel_path}: {entries} entries{status}")

            total_files += 1
            total_entries += entries
            total_cleaned += cleaned

    print(f"\nDone. {total_files} files, {total_entries} entries, {total_cleaned} cleaned.")


if __name__ == "__main__":
    main()
