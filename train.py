"""
train.py — SetFit Email Classification Training Script
=======================================================

Fine-tunes `intfloat/multilingual-e5-small` using the SetFit framework for
few-shot email classification with rich metadata features.

Training data is loaded from JSON files in `TrainingData/`:
    TrainingData/
    ├── URGENT.json
    ├── FOCUS.json
    ├── REFERENCE.json
    └── NOISE.json

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

Categories are auto-discovered from the .json filenames.
To add a new category, create a new JSON file (e.g. BILLING.json).

Usage:
    python train.py

After training, the model and label mapping are saved to `model/`.

Git LFS Instructions
--------------------
The trained model contains large binary files (.bin / .safetensors).
Use Git LFS to version-control the `model/` directory:

    1. Install Git LFS (system package):
           sudo apt-get install git-lfs   # Debian / Raspberry Pi OS
           brew install git-lfs            # macOS

    2. Initialize Git LFS in the repo:
           git lfs install

    3. Track model files:
           git lfs track "model/**"

    4. Commit and push:
           git add .gitattributes model/
           git commit -m "Add trained email classification model"
           git push
"""

import json
import os
from dataclasses import dataclass, field

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
    Load training data from JSON files.

    Each file {data_dir}/{LABEL_NAME}.json contains an array of objects.
    The label is the filename without the .json extension.

    Args:
        data_dir: Path to the directory containing .json files.

    Returns:
        List of EmailSample instances.
    """
    samples: list[EmailSample] = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Training data directory not found: '{data_dir}'. "
            f"Create it with one .json file per category."
        )

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".json"):
            continue

        label_name = os.path.splitext(filename)[0]  # "URGENT.json" → "URGENT"
        filepath = os.path.join(data_dir, filename)

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
# 4. Train
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


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
