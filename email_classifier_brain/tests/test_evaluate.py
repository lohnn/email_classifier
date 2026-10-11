"""
Tests for evaluate.py report-building logic.

These tests stub predictions and call build_report() directly — no SetFit
training is involved. sklearn is required for the metric tests (it is a
transitive dependency of setfit, pinned in requirements.txt); the tests are
skipped if it is unavailable.
"""

import os
import sys

import pytest

# Ensure the brain directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set TESTING=true so that classify.py (imported transitively) skips loading
# the heavy ML model
os.environ["TESTING"] = "true"

import evaluate

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")


def _always_sure(_probs) -> bool:
    return False


def _always_unsure(_probs) -> bool:
    return True


class TestBuildReport:
    """Unit tests for evaluate.build_report() with stubbed predictions."""

    # 8 samples across 2 categories; 6 correct, 2 misclassified.
    Y_TRUE = ["A", "A", "A", "A", "B", "B", "B", "B"]
    Y_PRED = ["A", "A", "A", "B", "B", "B", "B", "A"]
    Y_PROBS = [
        [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6],
        [0.2, 0.8], [0.1, 0.9], [0.3, 0.7], [0.6, 0.4],
    ]
    CATEGORIES = ["A", "B"]

    def _report(self, is_unsure=_always_sure):
        return evaluate.build_report(
            y_true=self.Y_TRUE,
            y_pred=self.Y_PRED,
            y_probs=self.Y_PROBS,
            categories=self.CATEGORIES,
            k=5,
            is_unsure=is_unsure,
        )

    def test_accuracy(self):
        report = self._report()
        # 6 of 8 correct
        assert report["accuracy"] == pytest.approx(0.75)

    def test_per_class_metrics(self):
        report = self._report()
        a = report["per_class"]["A"]
        b = report["per_class"]["B"]
        # A: TP=3, FP=1 (B predicted as A), FN=1 → P=3/4, R=3/4
        assert a["precision"] == pytest.approx(0.75)
        assert a["recall"] == pytest.approx(0.75)
        assert a["f1"] == pytest.approx(0.75)
        assert a["support"] == 4
        # B: TP=3, FP=1 (A predicted as B), FN=1 → P=3/4, R=3/4
        assert b["precision"] == pytest.approx(0.75)
        assert b["recall"] == pytest.approx(0.75)
        assert b["support"] == 4

    def test_confusion_matrix(self):
        report = self._report()
        cm = report["confusion_matrix"]
        assert cm["labels"] == ["A", "B"]
        # Row 0 (true A): 3 predicted A, 1 predicted B
        # Row 1 (true B): 1 predicted A, 3 predicted B
        assert cm["rows"] == [[3, 1], [1, 3]]

    def test_unsure_rate_none_unsure(self):
        report = self._report(is_unsure=_always_sure)
        assert report["unsure_rate"] == 0.0

    def test_unsure_rate_all_unsure(self):
        report = self._report(is_unsure=_always_unsure)
        assert report["unsure_rate"] == 1.0

    def test_unsure_rate_fractional(self):
        # Flag exactly the first two predictions as unsure → 2/8 = 0.25
        calls = {"n": 0}

        def first_two_unsure(_probs):
            calls["n"] += 1
            return calls["n"] <= 2

        report = self._report(is_unsure=first_two_unsure)
        assert report["unsure_rate"] == pytest.approx(0.25)

    def test_report_shell_fields(self):
        report = self._report()
        assert report["evaluates"] == "training_data_snapshot"
        assert report["method"] == "stratified_k_fold_cross_validation"
        assert report["k"] == 5
        assert report["sample_count"] == 8
        assert report["categories"] == ["A", "B"]
        assert "evaluated_at" in report
        assert "training_data_repo" in report
        assert "training_data_commit" in report
        # Thresholds are recorded so the unsure rate is interpretable
        thresholds = report["unsure_thresholds"]
        assert set(thresholds) == {"confidence", "delta"}
        assert 0.0 < thresholds["confidence"] <= 1.0
        assert 0.0 <= thresholds["delta"] <= 1.0

    def test_body_truncation_defaults_to_zero(self):
        """Without truncation data the report shows a zeroed prevalence block."""
        report = self._report()
        assert report["body_truncation"] == {"total": 0, "truncated": 0, "rate": 0.0}

    def test_body_truncation_recorded_when_provided(self):
        report = evaluate.build_report(
            y_true=self.Y_TRUE,
            y_pred=self.Y_PRED,
            y_probs=self.Y_PROBS,
            categories=self.CATEGORIES,
            k=5,
            is_unsure=_always_sure,
            truncation={"total": 8, "truncated": 3, "rate": 0.375},
        )
        assert report["body_truncation"]["truncated"] == 3
        assert report["body_truncation"]["total"] == 8
        assert report["body_truncation"]["rate"] == pytest.approx(0.375)

    def test_report_is_json_serializable(self):
        import json
        report = self._report()
        json.dumps(report)  # must not raise
