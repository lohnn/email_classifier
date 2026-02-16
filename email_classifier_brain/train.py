"""
train.py — SetFit Email Classification Training Script
=======================================================

Fine-tunes `intfloat/multilingual-e5-small` using the SetFit framework for
few-shot email classification with rich metadata features.

Training data is loaded from JSON files in `TrainingData/`:
    TrainingData/
    ├── NOISE.json                     ← flat label "NOISE"
    ├── WORK/
    │   ├── URGENT.json                ← nested label "WORK/URGENT"
    │   └── FOCUS.json                 ← nested label "WORK/FOCUS"
    └── PERSONAL/
        └── FINANCE.json               ← nested label "PERSONAL/FINANCE"

Each JSON file is an array of objects with these fields:
    {
      "subject": "...",
      "body": "...",
      "from": "sender@example.com",
      "to": "recipient@example.com",
      "cc": "",
      "mass_mail": false,
      "attachment_types": ["PDF", "DOCX"]
    }

Categories are auto-discovered from the .json file paths (relative to
TrainingData/). Subdirectories create hierarchical labels separated by '/'.
To add a new category, create a new JSON file or subdirectory.

Usage:
    python train.py

After training, the model and label mapping are saved to `model/`.

Model Storage
-------------
The trained model (~490 MB) is stored in Google Drive and synced
via rclone. After training, run `retrain.sh` to upload:

    ./retrain.sh   # pull data → train → upload → push

A MODEL_INFO.json provenance file is saved alongside the model,
linking it back to the exact training data commit.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone

import torch

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from config import (
    BASE_MODEL,
    MODEL_OUTPUT_DIR,
    TRAINING_DATA_DIR,
    format_model_input,
)


# ---------------------------------------------------------------------------
# 1. Data Structures
# ---------------------------------------------------------------------------

@dataclass
class EmailSample:
    """A single training email with metadata."""

    subject: str
    body: str
    label: str
    sender: str = ""
    to: str = ""
    cc: str = ""
    mass_mail: bool = False
    attachment_types: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Data Loader
# ---------------------------------------------------------------------------

def load_training_data(data_dir: str) -> list[EmailSample]:
    """
    Load training data from JSON files, with support for nested labels.

    Recursively walks {data_dir} for .json files. The label is derived
    from the path relative to {data_dir}, with the .json extension
    stripped. Subdirectories become label hierarchy separated by '/'.

    Examples:
        TrainingData/URGENT.json          → label "URGENT"
        TrainingData/WORK/FOCUS.json      → label "WORK/FOCUS"
        TrainingData/WORK/REVIEW/CODE.json → label "WORK/REVIEW/CODE"

    Args:
        data_dir: Path to the root training data directory.

    Returns:
        List of EmailSample instances.
    """
    samples: list[EmailSample] = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Training data directory not found: '{data_dir}'. "
            f"Create it with one .json file per category."
        )

    for dirpath, _, filenames in sorted(os.walk(data_dir)):
        for filename in sorted(filenames):
            if not filename.endswith(".json"):
                continue

            # Build hierarchical label from relative path
            rel_path = os.path.relpath(os.path.join(dirpath, filename), data_dir)
            label_name = os.path.splitext(rel_path)[0].replace(os.sep, '/')

            filepath = os.path.join(dirpath, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                entries = json.load(f)

            for entry in entries:
                samples.append(EmailSample(
                    subject=entry.get("subject", ""),
                    body=entry.get("body", ""),
                    label=label_name,
                    sender=entry.get("from", ""),
                    to=entry.get("to", ""),
                    cc=entry.get("cc", ""),
                    mass_mail=entry.get("mass_mail", False),
                    attachment_types=entry.get("attachment_types", []),
                ))

    if not samples:
        raise ValueError(
            f"No training data found in '{data_dir}'. "
            f"Add .json files with email examples."
        )

    return samples


# ---------------------------------------------------------------------------
# 3. Build Dataset
# ---------------------------------------------------------------------------

def build_dataset(
    samples: list[EmailSample],
) -> tuple[Dataset, dict[int, str]]:
    """
    Build a Hugging Face Dataset from EmailSample instances.

    Returns:
        dataset: HF Dataset with 'text' (prefixed) and 'label' (int) columns.
        label_mapping: dict mapping integer indices to label strings.
    """
    # Format each sample using the shared input formatter
    texts = [
        format_model_input(
            subject=s.subject,
            body=s.body,
            sender=s.sender,
            to=s.to,
            cc=s.cc,
            mass_mail=s.mass_mail,
            attachment_types=s.attachment_types,
        )
        for s in samples
    ]
    raw_labels = [s.label for s in samples]

    # Auto-discover unique labels (sorted for deterministic ordering)
    unique_labels = sorted(set(raw_labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    int_labels = [label_to_index[lbl] for lbl in raw_labels]

    dataset = Dataset.from_dict({"text": texts, "label": int_labels})

    print(f"Discovered {len(unique_labels)} categories: {unique_labels}")
    print(f"Dataset size: {len(dataset)} samples")

    # Print a sample input for verification
    if texts:
        print(f"\nExample model input:\n  {texts[0]}\n")

    return dataset, index_to_label


# ---------------------------------------------------------------------------
# 4. Model Provenance
# ---------------------------------------------------------------------------

def _git_info(repo_dir: str) -> dict[str, str]:
    """
    Capture git remote URL and HEAD commit from a directory.

    Returns a dict with 'repo' and 'commit' keys. Values are empty
    strings if the directory is not a git repo.
    """
    info: dict[str, str] = {"repo": "", "commit": ""}
    try:
        info["repo"] = subprocess.check_output(
            ["git", "-C", repo_dir, "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        info["commit"] = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def _write_model_info(
    label_mapping: dict[int, str],
    sample_count: int,
) -> None:
    """
    Write MODEL_INFO.json alongside the saved model.

    This file links the model snapshot back to the training data
    repo and commit, so you can always trace which data produced
    a given model — even when the model lives outside Git
    (e.g. in Google Drive).
    """
    git = _git_info(TRAINING_DATA_DIR)
    info = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_data_repo": git["repo"],
        "training_data_commit": git["commit"],
        "base_model": BASE_MODEL,
        "framework": "setfit",
        "categories": sorted(label_mapping.values()),
        "sample_count": sample_count,
    }
    info_path = os.path.join(MODEL_OUTPUT_DIR, "MODEL_INFO.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Model provenance saved to '{info_path}'")


# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------

def train() -> None:
    """Fine-tune the SetFit model and save it along with the label mapping."""
    samples = load_training_data(TRAINING_DATA_DIR)
    dataset, label_mapping = build_dataset(samples)

    # Auto-detect Apple Silicon GPU (MPS) for faster training
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training on device: {device}")

    # Load the base model
    model = SetFitModel.from_pretrained(BASE_MODEL, device=device)

    # Configure training arguments
    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        num_iterations=20,  # Number of text pairs for contrastive learning
        save_strategy="no",  # We save the final model ourselves below
    )

    # Create trainer and train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # Save the model
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Model saved to '{MODEL_OUTPUT_DIR}/'")

    # Save the label mapping alongside the model
    label_mapping_path = os.path.join(MODEL_OUTPUT_DIR, "label_mapping.json")
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    print(f"Label mapping saved to '{label_mapping_path}'")
    print(f"Labels: {label_mapping}")

    # Save model provenance info (links model snapshot to training data)
    _write_model_info(label_mapping, len(samples))


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
