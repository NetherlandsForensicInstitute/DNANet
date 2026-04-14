"""Tests for allele-level segmentation metrics."""

from __future__ import annotations

import pytest

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.evaluation.metrics.allele import (
    AlleleRecall,
    AlleleF1Score,
    AllelePrecision,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _marker(name: str, dye: int, alleles: list[tuple[str, int]]) -> Marker:
    """Shorthand to create a Marker with named alleles and heights."""
    return Marker(
        name=name,
        dye_row=dye,
        alleles=frozenset(Allele(name=n, height=h) for n, h in alleles),
    )


def _compute(metric, gts, preds) -> float:
    metric.update(gts, preds)
    return float(metric.compute())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_match():
    """Prediction matches ground truth exactly."""
    gt = [_marker("D5S818", 0, [("13", 500), ("15", 300)])]
    pred = [_marker("D5S818", 0, [("13", 480), ("15", 310)])]
    return [gt], [pred]


@pytest.fixture
def partial_match():
    """One correct, one extra, one missed."""
    gt = [
        _marker("D5S818", 0, [("13", 500), ("15", 300)]),
    ]
    pred = [
        _marker("D5S818", 0, [("13", 500), ("14", 200)]),
    ]
    # Predicted: D5S818_13, D5S818_14
    # GT:        D5S818_13, D5S818_15
    # TP=1, FP=1, FN=1
    return [gt], [pred]


@pytest.fixture
def no_predictions():
    """No alleles predicted."""
    gt = [_marker("D5S818", 0, [("13", 500)])]
    pred = []
    return [gt], [pred]


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

class TestAllelePrecision:
    def test_perfect(self, perfect_match):
        gts, preds = perfect_match
        assert _compute(AllelePrecision(), gts, preds) == pytest.approx(1.0)

    def test_partial(self, partial_match):
        gts, preds = partial_match
        # TP=1, FP=1 => precision=0.5
        assert _compute(AllelePrecision(), gts, preds) == pytest.approx(0.5)

    def test_no_predictions(self, no_predictions):
        gts, preds = no_predictions
        assert _compute(AllelePrecision(), gts, preds) == 0.0

    def test_multiple_samples(self):
        gt1 = [_marker("D5S818", 0, [("13", 500)])]
        pred1 = [_marker("D5S818", 0, [("13", 480)])]  # TP=1
        gt2 = [_marker("vWA", 1, [("16", 300)])]
        pred2 = [_marker("vWA", 1, [("17", 400)])]  # FP=1
        # Total: TP=1, FP=1 => precision=0.5
        metric = AllelePrecision()
        metric.update([gt1], [pred1])
        metric.update([gt2], [pred2])
        assert float(metric.compute()) == pytest.approx(0.5)

    def test_reset_clears_state(self, partial_match):
        gts, preds = partial_match
        metric = AllelePrecision()
        metric.update(gts, preds)
        metric.reset()
        metric.update([[]], [[]])
        assert float(metric.compute()) == 0.0

    def test_rejects_mismatched_sample_counts(self):
        metric = AllelePrecision()
        gt = [_marker("D5S818", 0, [("13", 500)])]
        pred = [_marker("D5S818", 0, [("13", 480)])]
        with pytest.raises(ValueError, match="same number of samples"):
            metric.update([gt], [pred, pred])


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

class TestAlleleRecall:
    def test_perfect(self, perfect_match):
        gts, preds = perfect_match
        assert _compute(AlleleRecall(), gts, preds) == pytest.approx(1.0)

    def test_partial(self, partial_match):
        gts, preds = partial_match
        # TP=1, FN=1 => recall=0.5
        assert _compute(AlleleRecall(), gts, preds) == pytest.approx(0.5)

    def test_no_gt_alleles(self):
        gts = [[]]
        preds = [[_marker("D5S818", 0, [("13", 500)])]]
        assert _compute(AlleleRecall(), gts, preds) == 0.0


# ---------------------------------------------------------------------------
# F1 Score
# ---------------------------------------------------------------------------

class TestAlleleF1:
    def test_perfect(self, perfect_match):
        gts, preds = perfect_match
        assert _compute(AlleleF1Score(), gts, preds) == pytest.approx(1.0)

    def test_partial(self, partial_match):
        gts, preds = partial_match
        # P=0.5, R=0.5 => F1=0.5
        assert _compute(AlleleF1Score(), gts, preds) == pytest.approx(0.5)

    def test_both_empty(self):
        assert _compute(AlleleF1Score(), [[]], [[]]) == 0.0


# ---------------------------------------------------------------------------
# Locus filtering
# ---------------------------------------------------------------------------

class TestLocusFiltering:
    def test_filter_to_specific_locus(self):
        gt = [
            _marker("D5S818", 0, [("13", 500)]),
            _marker("vWA", 1, [("16", 300)]),
        ]
        pred = [
            _marker("D5S818", 0, [("13", 480)]),
            _marker("vWA", 1, [("17", 400)]),  # wrong allele
        ]
        # Only D5S818: TP=1, FP=0 => precision=1.0
        assert _compute(AllelePrecision(locus="D5S818"), [gt], [pred]) == pytest.approx(1.0)
        # Only vWA: TP=0, FP=1 => precision=0.0
        assert _compute(AllelePrecision(locus="vWA"), [gt], [pred]) == pytest.approx(0.0)

