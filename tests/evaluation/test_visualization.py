"""Tests for visualization functions.

These tests verify figure creation without displaying plots.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from dnanet.evaluation.visualization import (
    DYE_COLORS,
    plot_profile,
    plot_profile_marker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def signal_5dye() -> np.ndarray:
    """Synthetic 5-dye EPG signal."""
    return np.random.default_rng(42).random((5, 200)).astype(np.float32)


@pytest.fixture
def annotation_5dye() -> np.ndarray:
    """Binary annotation mask matching signal_5dye."""
    ann = np.zeros((5, 200), dtype=float)
    ann[0, 30:50] = 1
    ann[1, 80:100] = 1
    return ann


@pytest.fixture
def prediction_5dye() -> np.ndarray:
    """Probability prediction mask matching signal_5dye."""
    pred = np.random.default_rng(123).random((5, 200)).astype(np.float32) * 0.3
    pred[0, 30:50] = 0.9
    pred[1, 80:100] = 0.8
    return pred


# ---------------------------------------------------------------------------
# plot_profile
# ---------------------------------------------------------------------------

class TestPlotProfile:
    def test_signal_only(self, signal_5dye):
        fig = plot_profile(signal_5dye)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 5

    def test_with_annotation(self, signal_5dye, annotation_5dye):
        fig = plot_profile(signal_5dye, annotation=annotation_5dye)
        assert isinstance(fig, Figure)

    def test_with_prediction_mask(self, signal_5dye, prediction_5dye):
        fig = plot_profile(
            signal_5dye,
            prediction=prediction_5dye,
            prediction_as_mask=True,
        )
        assert isinstance(fig, Figure)

    def test_with_prediction_line(self, signal_5dye, prediction_5dye):
        fig = plot_profile(
            signal_5dye,
            prediction=prediction_5dye,
            prediction_as_mask=False,
        )
        assert isinstance(fig, Figure)

    def test_with_title(self, signal_5dye):
        fig = plot_profile(signal_5dye, title="Test Profile")
        assert fig._suptitle is not None

    def test_custom_dye_colors(self, signal_5dye):
        fig = plot_profile(
            signal_5dye,
            dye_colors=["r", "g", "b", "k", "m"],
        )
        assert isinstance(fig, Figure)

    def test_all_annotations_and_predictions(
        self, signal_5dye, annotation_5dye, prediction_5dye,
    ):
        fig = plot_profile(
            signal_5dye,
            annotation=annotation_5dye,
            prediction=prediction_5dye,
            title="Full Profile",
        )
        assert isinstance(fig, Figure)

    def test_single_dye(self):
        signal = np.random.default_rng(1).random((1, 100))
        fig = plot_profile(signal)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_profile_marker
# ---------------------------------------------------------------------------

class TestPlotProfileMarker:
    def test_basic_marker_plot(self, signal_5dye):
        scaler = np.linspace(50, 350, 200)
        fig = plot_profile_marker(
            signal_5dye,
            scaler,
            marker_bp_range=(100.0, 150.0),
            dye_row=0,
            title="Test Marker",
        )
        assert isinstance(fig, Figure)

    def test_with_annotation(self, signal_5dye, annotation_5dye):
        scaler = np.linspace(50, 350, 200)
        fig = plot_profile_marker(
            signal_5dye,
            scaler,
            marker_bp_range=(100.0, 150.0),
            dye_row=0,
            annotation=annotation_5dye,
        )
        assert isinstance(fig, Figure)

    def test_with_prediction(self, signal_5dye, prediction_5dye):
        scaler = np.linspace(50, 350, 200)
        fig = plot_profile_marker(
            signal_5dye,
            scaler,
            marker_bp_range=(100.0, 150.0),
            dye_row=0,
            prediction=prediction_5dye,
        )
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_dye_colors_standard():
    assert len(DYE_COLORS) == 6
    assert DYE_COLORS[0] == "blue"
    assert "orange" in DYE_COLORS
