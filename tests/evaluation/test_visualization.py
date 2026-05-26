"""Tests for visualization functions.

These tests verify figure creation without displaying plots.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from dnanet.core.constants import LabelCategory
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
    """Multiclass annotation map matching signal_5dye."""
    ann = np.zeros((5, 200), dtype=np.int32)
    ann[0, 30:50] = list(LabelCategory).index(LabelCategory.ALLELE)
    ann[1, 80:100] = list(LabelCategory).index(LabelCategory.STUTTER)
    return ann


@pytest.fixture
def prediction_5dye() -> np.ndarray:
    """Multiclass prediction map matching signal_5dye."""
    pred = np.zeros((5, 200), dtype=np.int32)
    pred[0, 30:50] = list(LabelCategory).index(LabelCategory.PULL_UP)
    pred[1, 80:100] = list(LabelCategory).index(LabelCategory.BLEED_THROUGH)
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
        assert len(fig.axes) == 10

    def test_with_prediction(self, signal_5dye, prediction_5dye):
        fig = plot_profile(signal_5dye, prediction=prediction_5dye)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 10

    def test_with_title(self, signal_5dye):
        fig = plot_profile(signal_5dye, title='Test Profile')
        assert fig._suptitle is not None

    def test_custom_dye_colors(self, signal_5dye):
        fig = plot_profile(
            signal_5dye,
            dye_colors=['r', 'g', 'b', 'k', 'm'],
        )
        assert isinstance(fig, Figure)

    def test_all_annotations_and_predictions(
        self,
        signal_5dye,
        annotation_5dye,
        prediction_5dye,
    ):
        fig = plot_profile(
            signal_5dye,
            annotation=annotation_5dye,
            prediction=prediction_5dye,
            title='Full Profile',
        )
        assert isinstance(fig, Figure)
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [
            'Allele',
            'Stutter',
            'Pull Up',
            'Bleed Through',
        ]

    def test_single_dye(self):
        signal = np.random.default_rng(1).random((1, 100))
        fig = plot_profile(signal)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_annotation_and_prediction_use_separate_tracks(self):
        signal = np.random.default_rng(7).random((1, 100))
        annotation = np.zeros((1, 100), dtype=np.int32)
        prediction = np.zeros((1, 100), dtype=np.int32)
        annotation[0, 20:23] = list(LabelCategory).index(LabelCategory.ALLELE)
        prediction[0, 40:43] = list(LabelCategory).index(LabelCategory.STUTTER)

        fig = plot_profile(signal, annotation=annotation, prediction=prediction)

        assert len(fig.axes[1].patches) == 1
        assert len(fig.axes[2].patches) == 1

    def test_single_scanpoint_annotation_is_rendered_as_square(self):
        signal = np.random.default_rng(11).random((1, 100))
        annotation = np.zeros((1, 100), dtype=np.int32)
        annotation[0, 10] = list(LabelCategory).index(LabelCategory.ALLELE)

        fig = plot_profile(signal, annotation=annotation)

        assert len(fig.axes[1].patches) == 0
        assert len(fig.axes[1].collections) == 1
        assert tuple(fig.axes[1].collections[0].get_edgecolors()[0]) == to_rgba('black')

    def test_multi_scanpoint_annotation_is_rendered_as_bar(self):
        signal = np.random.default_rng(13).random((1, 100))
        annotation = np.zeros((1, 100), dtype=np.int32)
        annotation[0, 10:14] = list(LabelCategory).index(LabelCategory.ALLELE)

        fig = plot_profile(signal, annotation=annotation)

        assert len(fig.axes[1].patches) == 1
        assert len(fig.axes[1].collections) == 0

    def test_multiclass_uses_label_category_colors(self):
        signal = np.random.default_rng(17).random((1, 100))
        annotation = np.zeros((1, 100), dtype=np.int32)
        prediction = np.zeros((1, 100), dtype=np.int32)
        annotation[0, 10:14] = list(LabelCategory).index(LabelCategory.ALLELE)
        prediction[0, 40] = list(LabelCategory).index(LabelCategory.STUTTER)

        fig = plot_profile(signal, annotation=annotation, prediction=prediction)

        ann_patch = fig.axes[1].patches[0]
        pred_square = fig.axes[2].collections[0]

        assert ann_patch.get_facecolor() == to_rgba(LabelCategory.ALLELE.color)
        assert tuple(pred_square.get_facecolors()[0]) == to_rgba(LabelCategory.STUTTER.color)


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
            title='Test Marker',
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
    assert DYE_COLORS[0] == 'blue'
    assert 'orange' in DYE_COLORS
