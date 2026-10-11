"""
evaluate.py — Cross-Validation Evaluation Harness
==================================================

Answers the question train.py cannot: "did my training-data change make the
model better or worse?" Runs stratified k-fold cross-validation (k=5) over
the current TrainingData/ snapshot and reports quality metrics — without
touching the production model in `model/`.

How it works:
    1. Loads training data exactly like train.py (`load_training_data`) and
       formats inputs through the same `config.format_model_input()` path —
       evaluation sees what training and inference see.
    2. Splits the data into k stratified folds. For each fold, a fresh
       SetFit model is trained on k-1 folds and predicted on the held-out
       fold. Every example is validated exactly once.
    3. Pools ALL out-of-fold predictions and computes metrics once
       (accuracy, per-class precision/recall/F1, confusion matrix, and the
       unsure rate via `classify.is_unsure_classification`).
    4. Writes the report to MODEL_EVAL.json next to MODEL_INFO.json in the
       model dir.

MODEL_EVAL.json semantics
-------------------------
MODEL_EVAL.json is the quality score of the *current training-data
snapshot*, not of any specific saved model file. Cross-validation trains
k throwaway models and discards them; the numbers describe what a model
trained on this data is expected to achieve. The file records the training
data's git repo/commit so it can be compared across retrains.

Usage:
    python evaluate.py            # k=5 (default)
    python evaluate.py --folds 3  # fewer folds (faster, noisier)

Requirements: each category needs at least k examples (a stratified fold
must hold out at least one example per class). With tiny datasets the
metrics are high-variance — treat them as a signal, not a guarantee.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from config import (
    BASE_MODEL,
    BODY_TOKEN_BUDGET,
    MAX_BODY_CHARS,
    MODEL_OUTPUT_DIR,
    TRAINING_DATA_DIR,
    UNSURE_CONFIDENCE_THRESHOLD,
    UNSURE_DELTA_THRESHOLD,
    format_model_input,
)

if TYPE_CHECKING:
    # For type checkers only — train.py imports torch/setfit at module level,
    # so it is imported lazily inside functions to keep this module
    # importable without the ML stack (e.g. in unit tests).
    from train import EmailSample


# ---------------------------------------------------------------------------
# 1. Model Input Formatting
# ---------------------------------------------------------------------------

def _format_samples(samples: list["EmailSample"]) -> tuple[list[str], list[str]]:
    """
    Format samples into (texts, labels) using the shared input formatter.

    Uses config.format_model_input() — the same code path as train.py and
    classify.py — so evaluation never drifts from the production format.
    """
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
    labels = [s.label for s in samples]
    return texts, labels


# ---------------------------------------------------------------------------
# 2. Cross-Validation
# ---------------------------------------------------------------------------

def _train_and_predict_fold(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    device: str,
) -> list:
    """
    Train a fresh SetFit model on one fold's training split and return
    per-class probability rows for the validation split.

    Heavy imports (torch, datasets, setfit) live inside this function so the
    module stays importable without the ML stack installed (e.g. for unit
    tests of the report-building logic).
    """
    import torch  # noqa: F401 — imported for side effect of clear error if missing
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})

    model = SetFitModel.from_pretrained(BASE_MODEL, device=device)
    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        num_iterations=20,  # Same as train.py — keep fold models comparable
        save_strategy="no",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()

    probs = model.predict_proba(val_texts)
    # predict_proba returns a tensor or ndarray — normalize to plain lists
    return [row.tolist() for row in probs]


def run_cross_validation(
    samples: list["EmailSample"],
    k: int = 5,
) -> tuple[list[str], list[str], list[list[float]], list[str], dict]:
    """
    Run stratified k-fold cross-validation over the samples.

    Every example is held out exactly once; predictions from all folds are
    pooled and returned in original sample order.

    Args:
        samples: The full training set.
        k: Number of folds. Each class must have at least k examples.

    Returns:
        texts:      Formatted model inputs, one per sample.
        y_true:     True label strings, one per sample.
        y_probs:    Per-class probability rows (pooled out-of-fold), one per sample.
        categories: Sorted list of all category label strings (column order
                    of y_probs rows).
        truncation: {"total", "truncated", "rate"} — body-truncation
                    prevalence measured while formatting the samples
                    (observable signal for the E5 context-window cap).
    """
    import torch
    from sklearn.model_selection import StratifiedKFold

    import config
    config.truncation_stats.reset()
    texts, labels = _format_samples(samples)
    trunc_snap = config.truncation_stats.snapshot()
    truncation = {
        "total": trunc_snap["model_inputs_total"],
        "truncated": trunc_snap["model_inputs_truncated"],
        "rate": (
            round(trunc_snap["model_inputs_truncated"] / trunc_snap["model_inputs_total"], 4)
            if trunc_snap["model_inputs_total"] else 0.0
        ),
    }

    categories = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(categories)}
    y_true_indices = [label_to_index[lbl] for lbl in labels]

    # StratifiedKFold requires at least k members in every class
    counts = {c: labels.count(c) for c in categories}
    too_small = {c: n for c, n in counts.items() if n < k}
    if too_small:
        details = ", ".join(f"{c} ({n} examples)" for c, n in sorted(too_small.items()))
        raise ValueError(
            f"Cannot run {k}-fold cross-validation: these categories have "
            f"fewer than {k} examples: {details}. "
            f"Add more training data or re-run with --folds {min(counts.values())}."
        )

    # Device selection mirrors train.py
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Evaluating on device: {device}")

    # Pooled out-of-fold probabilities, aligned with sample order
    y_probs: list[list[float] | None] = [None] * len(samples)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, y_true_indices), start=1):
        print(f"\n--- Fold {fold_idx}/{k} "
              f"(train: {len(train_idx)}, validate: {len(val_idx)}) ---")

        train_texts = [texts[i] for i in train_idx]
        train_labels = [y_true_indices[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]

        fold_probs = _train_and_predict_fold(
            train_texts, train_labels, val_texts, device,
        )
        for i, probs in zip(val_idx, fold_probs):
            y_probs[i] = probs

        print(f"Fold {fold_idx}/{k} complete.")

    # Every sample was validated in exactly one fold — nothing may be missing
    assert all(p is not None for p in y_probs), "unfilled out-of-fold predictions"
    return texts, labels, y_probs, categories, truncation  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 3. Metrics & Report
# ---------------------------------------------------------------------------

def build_report(
    y_true: list[str],
    y_pred: list[str],
    y_probs: list[list[float]],
    categories: list[str],
    k: int,
    is_unsure,
    training_data_git: dict[str, str] | None = None,
    truncation: dict | None = None,
) -> dict:
    """
    Build the evaluation report dict from pooled out-of-fold predictions.

    Pure function over its arguments — sklearn is the only import, and it is
    deferred to call time — so the function is unit-testable without the
    SetFit/torch stack. `is_unsure` is injected (normally
    classify.is_unsure_classification) for the same reason.

    Args:
        y_true:     True label strings (pooled across folds).
        y_pred:     Predicted label strings (pooled across folds).
        y_probs:    Per-class probability rows aligned with y_true/y_pred.
        categories: Sorted category list (confusion-matrix axis order).
        k:          Number of folds used.
        is_unsure:  Callable probs -> bool flagging uncertain predictions.
        training_data_git: Optional {"repo", "commit"} dict linking the report
                    to the training-data snapshot (from train._git_info).
        truncation: Optional {"total", "truncated", "rate"} dict describing
                    how many formatted inputs had their body capped to fit
                    the E5 context window (from run_cross_validation).

    Returns:
        The report dict (also serialized to MODEL_EVAL.json).
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    accuracy = accuracy_score(y_true, y_pred)

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=categories, zero_division=0,
    )
    per_class = {
        label: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for label, p, r, f, s in zip(categories, precisions, recalls, f1s, supports)
    }

    matrix = confusion_matrix(y_true, y_pred, labels=categories).tolist()

    unsure_flags = [bool(is_unsure(probs)) for probs in y_probs]
    unsure_rate = round(sum(unsure_flags) / len(unsure_flags), 4) if unsure_flags else 0.0

    git = training_data_git or {"repo": "", "commit": ""}
    return {
        # This report scores the training-DATA snapshot via cross-validation,
        # not a specific saved model — the k fold models are discarded.
        "evaluates": "training_data_snapshot",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "method": "stratified_k_fold_cross_validation",
        "k": k,
        "sample_count": len(y_true),
        "training_data_repo": git["repo"],
        "training_data_commit": git["commit"],
        "base_model": BASE_MODEL,
        "categories": categories,
        "accuracy": round(float(accuracy), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": categories,
            # rows[i][j] = count of true label categories[i] predicted as categories[j]
            "rows": matrix,
        },
        "unsure_rate": unsure_rate,
        "unsure_thresholds": {
            "confidence": UNSURE_CONFIDENCE_THRESHOLD,
            "delta": UNSURE_DELTA_THRESHOLD,
        },
        # Body-truncation prevalence: how many inputs hit the E5
        # context-window cap during formatting. A high rate means many
        # predictions are made on partial bodies.
        "body_truncation": truncation or {"total": 0, "truncated": 0, "rate": 0.0},
    }


# ---------------------------------------------------------------------------
# 4. Pretty Printing
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    """Print per-class metrics and the confusion matrix as readable tables."""
    categories = report["categories"]
    per_class = report["per_class"]

    print("\n" + "=" * 72)
    print(f"CROSS-VALIDATION RESULTS  (k={report['k']}, "
          f"{report['sample_count']} samples, {len(categories)} categories)")
    print("=" * 72)
    print(f"Overall accuracy : {report['accuracy']:.4f}")
    print(f"Unsure rate      : {report['unsure_rate']:.4f} "
          f"(confidence < {report['unsure_thresholds']['confidence']} or "
          f"top-2 delta < {report['unsure_thresholds']['delta']})")
    bt = report["body_truncation"]
    print(f"Body truncation  : {bt['truncated']}/{bt['total']} inputs "
          f"({bt['rate']:.1%}) exceeded the E5 body cap "
          f"({MAX_BODY_CHARS} chars ≈ {BODY_TOKEN_BUDGET} tokens)")

    # Per-class metrics table
    header = f"{'Category':<28} {'Precision':>9} {'Recall':>8} {'F1':>8} {'Support':>8}"
    print("\nPer-class metrics:")
    print(header)
    print("-" * len(header))
    for label in categories:
        m = per_class[label]
        print(f"{label:<28} {m['precision']:>9.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['support']:>8d}")

    # Confusion matrix — rows are true labels, columns are predictions.
    # Abbreviate long hierarchical labels to their last path component so
    # the table stays readable (full names are in MODEL_EVAL.json).
    def _short(label: str) -> str:
        tail = label.rsplit("/", 1)[-1]
        return tail[:10]

    col_labels = [_short(c) for c in categories]
    col_w = max(10, max(len(c) for c in col_labels) + 2)
    first_w = max(10, max(len(_short(c)) for c in categories) + 2)

    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(" " * first_w + "".join(f"{c:>{col_w}}" for c in col_labels))
    for label, row in zip(categories, report["confusion_matrix"]["rows"]):
        cells = "".join(f"{v:>{col_w}}" for v in row)
        print(f"{_short(label):<{first_w}}{cells}")
    print()


# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------

def evaluate(k: int = 5) -> None:
    """Run cross-validation and write MODEL_EVAL.json to the model dir."""
    # Imported here rather than at module level: classify.py loads the
    # production model at import time unless TESTING=true, and train.py
    # imports the ML stack at module level. Lazy imports keep this module
    # importable in lightweight test contexts.
    from classify import is_unsure_classification
    from train import _git_info, load_training_data

    samples = load_training_data(TRAINING_DATA_DIR)
    print(f"Loaded {len(samples)} training samples from '{TRAINING_DATA_DIR}'")

    _, y_true, y_probs, categories, truncation = run_cross_validation(samples, k=k)

    # Predicted label = argmax of each pooled out-of-fold probability row
    y_pred = [categories[max(range(len(row)), key=lambda j: row[j])] for row in y_probs]

    report = build_report(
        y_true, y_pred, y_probs, categories, k,
        is_unsure_classification,
        training_data_git=_git_info(TRAINING_DATA_DIR),
        truncation=truncation,
    )
    print_report(report)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    eval_path = os.path.join(MODEL_OUTPUT_DIR, "MODEL_EVAL.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Evaluation report saved to '{eval_path}'")


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the email classifier with stratified k-fold "
                    "cross-validation over the current TrainingData/ snapshot.",
    )
    parser.add_argument(
        "--folds", type=int, default=5, metavar="K",
        help="Number of CV folds (default: 5). Each category needs at least "
             "K examples.",
    )
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    evaluate(k=args.folds)
