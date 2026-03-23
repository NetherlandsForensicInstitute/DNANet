"""Tests for the evaluation task."""

from __future__ import annotations

import numpy as np
import pytest

from dnanet.tasks.evaluate import (
    _METRIC_REGISTRY,
    _compute_pixel_metrics,
    _save_results,
)


# ---------------------------------------------------------------------------
# _compute_pixel_metrics
# ---------------------------------------------------------------------------

class TestComputePixelMetrics:
    def test_perfect_predictions(self):
        gt = [np.array([[1, 1, 0, 0]], dtype=float)]
        pred = [np.array([[1, 1, 0, 0]], dtype=float)]
        results = _compute_pixel_metrics(gt, pred, ["pixel_precision", "pixel_recall"])
        assert results["pixel_precision"] == pytest.approx(1.0)
        assert results["pixel_recall"] == pytest.approx(1.0)

    def test_partial_predictions(self):
        gt = [np.array([[1, 1, 0]], dtype=float)]
        pred = [np.array([[1, 0, 1]], dtype=float)]
        # TP=1, FP=1, FN=1
        results = _compute_pixel_metrics(gt, pred, ["pixel_precision", "pixel_recall", "pixel_f1_score"])
        assert results["pixel_precision"] == pytest.approx(0.5)
        assert results["pixel_recall"] == pytest.approx(0.5)
        assert results["pixel_f1_score"] == pytest.approx(0.5)

    def test_iou_metric(self):
        gt = [np.array([[1, 1, 0, 0]], dtype=float)]
        pred = [np.array([[1, 1, 0, 0]], dtype=float)]
        results = _compute_pixel_metrics(gt, pred, ["average_binary_iou"])
        assert results["average_binary_iou"] == pytest.approx(1.0)

    def test_unknown_metric_skipped(self):
        gt = [np.zeros((1, 4))]
        pred = [np.zeros((1, 4))]
        results = _compute_pixel_metrics(gt, pred, ["nonexistent_metric"])
        assert len(results) == 0

    def test_empty_metric_list(self):
        gt = [np.zeros((1, 4))]
        pred = [np.zeros((1, 4))]
        results = _compute_pixel_metrics(gt, pred, [])
        assert results == {}


# ---------------------------------------------------------------------------
# _save_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_saves_json(self, tmp_path):
        results = {"pixel_precision": 0.95, "pixel_recall": 0.90}
        path = _save_results(results, str(tmp_path))
        assert path.exists()
        import json
        loaded = json.loads(path.read_text())
        assert loaded["pixel_precision"] == 0.95

    def test_custom_filename(self, tmp_path):
        results = {"f1": 0.85}
        path = _save_results(results, str(tmp_path), filename="results.json")
        assert path.name == "results.json"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestMetricRegistry:
    def test_contains_all_pixel_metrics(self):
        expected = ["pixel_precision", "pixel_recall", "pixel_f1_score", "average_binary_iou"]
        for name in expected:
            assert name in _METRIC_REGISTRY
