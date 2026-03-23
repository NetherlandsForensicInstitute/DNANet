"""Tests for pixel-level segmentation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from dnanet.evaluation.metrics.pixel import (
    average_binary_iou,
    pixel_f1_score,
    pixel_precision,
    pixel_recall,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_prediction():
    """GT and prediction match perfectly."""
    gt = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
    pred = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
    return [gt], [pred]


@pytest.fixture
def partial_prediction():
    """Some FP and FN present."""
    gt = np.array([[1, 1, 1, 0, 0]], dtype=float)
    pred = np.array([[0.8, 0.8, 0.2, 0.9, 0.1]], dtype=float)
    # At threshold=0.5: pred_pos = [T, T, F, T, F]
    # TP=2, FP=1, FN=1
    return [gt], [pred]


@pytest.fixture
def all_negative():
    """GT is all zeros."""
    gt = np.zeros((2, 10), dtype=float)
    pred = np.zeros((2, 10), dtype=float)
    return [gt], [pred]


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

class TestPixelPrecision:
    def test_perfect(self, perfect_prediction):
        gts, preds = perfect_prediction
        assert pixel_precision(gts, preds) == pytest.approx(1.0)

    def test_partial(self, partial_prediction):
        gts, preds = partial_prediction
        # TP=2, FP=1 => precision = 2/3
        assert pixel_precision(gts, preds) == pytest.approx(2 / 3)

    def test_no_positives_predicted(self):
        gt = np.array([[1, 1, 0]], dtype=float)
        pred = np.array([[0, 0, 0]], dtype=float)
        assert pixel_precision([gt], [pred]) == 0.0

    def test_multiple_samples(self):
        gt1 = np.array([[1, 0]], dtype=float)
        pred1 = np.array([[1, 0]], dtype=float)
        gt2 = np.array([[0, 1]], dtype=float)
        pred2 = np.array([[1, 1]], dtype=float)
        # TP=2, FP=1 => 2/3
        assert pixel_precision([gt1, gt2], [pred1, pred2]) == pytest.approx(2 / 3)

    def test_custom_threshold(self):
        gt = np.array([[1, 0]], dtype=float)
        pred = np.array([[0.3, 0.2]], dtype=float)
        # threshold=0.25: pred_pos=[T, F] -> TP=1, FP=0
        assert pixel_precision([gt], [pred], threshold=0.25) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

class TestPixelRecall:
    def test_perfect(self, perfect_prediction):
        gts, preds = perfect_prediction
        assert pixel_recall(gts, preds) == pytest.approx(1.0)

    def test_partial(self, partial_prediction):
        gts, preds = partial_prediction
        # TP=2, FN=1 => recall = 2/3
        assert pixel_recall(gts, preds) == pytest.approx(2 / 3)

    def test_no_positives_in_gt(self, all_negative):
        gts, preds = all_negative
        assert pixel_recall(gts, preds) == 0.0

    def test_all_missed(self):
        gt = np.array([[1, 1]], dtype=float)
        pred = np.array([[0, 0]], dtype=float)
        assert pixel_recall([gt], [pred]) == 0.0


# ---------------------------------------------------------------------------
# F1 Score
# ---------------------------------------------------------------------------

class TestPixelF1:
    def test_perfect(self, perfect_prediction):
        gts, preds = perfect_prediction
        assert pixel_f1_score(gts, preds) == pytest.approx(1.0)

    def test_partial(self, partial_prediction):
        gts, preds = partial_prediction
        # P=2/3, R=2/3 => F1=2/3
        assert pixel_f1_score(gts, preds) == pytest.approx(2 / 3)

    def test_both_zero(self, all_negative):
        gts, preds = all_negative
        assert pixel_f1_score(gts, preds) == 0.0


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

class TestAverageBinaryIoU:
    def test_perfect(self, perfect_prediction):
        gts, preds = perfect_prediction
        assert average_binary_iou(gts, preds) == pytest.approx(1.0)

    def test_no_overlap(self):
        gt = np.array([[1, 1, 0, 0]], dtype=float)
        pred = np.array([[0, 0, 1, 1]], dtype=float)
        # intersection=0, union=4 => IoU=0
        assert average_binary_iou([gt], [pred]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        gt = np.array([[1, 1, 1, 0]], dtype=float)
        pred = np.array([[0, 1, 1, 1]], dtype=float)
        # intersection=2, union=2+3-2=4-1=... gt=3, pred=3, inter=2, union=4
        assert average_binary_iou([gt], [pred]) == pytest.approx(2 / 4)

    def test_empty_lists(self):
        assert average_binary_iou([], []) == 0.0

    def test_all_zeros(self, all_negative):
        gts, preds = all_negative
        # union=0 for each sample, contributes 0
        assert average_binary_iou(gts, preds) == pytest.approx(0.0)

    def test_average_across_samples(self):
        gt1 = np.array([[1, 1, 0, 0]], dtype=float)
        pred1 = np.array([[1, 1, 0, 0]], dtype=float)  # IoU=1
        gt2 = np.array([[1, 1, 0, 0]], dtype=float)
        pred2 = np.array([[0, 0, 1, 1]], dtype=float)  # IoU=0
        assert average_binary_iou([gt1, gt2], [pred1, pred2]) == pytest.approx(0.5)
