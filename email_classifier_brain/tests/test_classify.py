"""
Tests for classify.py helper functions that don't require the ML model.
"""

import os
import sys

import pytest

# Ensure the brain directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set TESTING=true so that classify.py skips loading the heavy ML model
os.environ["TESTING"] = "true"

import classify


class TestIsUnsureClassification:
    """Unit tests for classify.is_unsure_classification()."""

    def test_high_confidence_single_winner_is_sure(self):
        """A clearly dominant class with high probability is not unsure."""
        # top=0.90, second=0.07 → score > threshold, delta > 0.10
        probs = [0.90, 0.07, 0.02, 0.01]
        assert classify.is_unsure_classification(probs) is False

    def test_low_confidence_below_threshold_is_unsure(self):
        """A top probability below UNSURE_CONFIDENCE_THRESHOLD triggers unsure."""
        # 0.60 < default threshold of 0.65
        probs = [0.60, 0.25, 0.10, 0.05]
        assert classify.is_unsure_classification(probs) is True

    def test_close_top_two_is_unsure(self):
        """When top-2 probabilities are within UNSURE_DELTA_THRESHOLD, flag as unsure."""
        # top=0.47, second=0.44 → delta=0.03 < 0.10 threshold
        probs = [0.47, 0.44, 0.05, 0.04]
        assert classify.is_unsure_classification(probs) is True

    def test_exact_threshold_boundary_confident(self):
        """A score exactly at the confidence threshold is not unsure (boundary is exclusive)."""
        # Use 0.65 exactly; 0.65 is NOT < 0.65
        probs = [0.65, 0.25, 0.10]
        assert classify.is_unsure_classification(probs) is False

    def test_delta_exactly_at_threshold_confident(self):
        """A delta exactly at UNSURE_DELTA_THRESHOLD is not unsure."""
        # delta = 0.10 exactly; 0.10 is NOT < 0.10
        probs = [0.75, 0.65, 0.10]
        # top=0.75, second=0.65 → delta=0.10; 0.75 >= 0.65 → not low conf
        assert classify.is_unsure_classification(probs) is False

    def test_single_class_above_threshold_is_sure(self):
        """With only one class (degenerate case), only the confidence check applies."""
        probs = [1.0]
        assert classify.is_unsure_classification(probs) is False

    def test_single_class_below_threshold_is_unsure(self):
        """A single class below confidence threshold is still unsure."""
        probs = [0.50]
        assert classify.is_unsure_classification(probs) is True

    def test_numpy_array_compatible(self):
        """is_unsure_classification handles numpy-array-like inputs (from predict_proba)."""
        try:
            import numpy as np
            probs = np.array([0.90, 0.07, 0.02, 0.01])
            assert classify.is_unsure_classification(probs) is False

            probs_unsure = np.array([0.48, 0.45, 0.04, 0.03])
            assert classify.is_unsure_classification(probs_unsure) is True
        except ImportError:
            pytest.skip("numpy not available")
