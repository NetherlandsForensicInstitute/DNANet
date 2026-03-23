"""Tests for allele calling from segmentation predictions."""

from __future__ import annotations

import numpy as np
import pytest

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.panel import Panel
from dnanet.evaluation.allele_caller import AlleleCaller, NearestBasePairCaller


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_panel() -> Panel:
    """Panel with 2 markers on dye 0, 1 on dye 1."""
    return Panel(markers=[
        Marker(
            name="MarkerA",
            dye_row=0,
            alleles=(
                Allele(name="10", base_pair=100.0, left_bin=0.5, right_bin=0.5),
                Allele(name="11", base_pair=110.0, left_bin=0.5, right_bin=0.5),
                Allele(name="12", base_pair=120.0, left_bin=0.5, right_bin=0.5),
            ),
        ),
        Marker(
            name="MarkerB",
            dye_row=0,
            alleles=(
                Allele(name="20", base_pair=200.0, left_bin=0.5, right_bin=0.5),
                Allele(name="21", base_pair=210.0, left_bin=0.5, right_bin=0.5),
            ),
        ),
        Marker(
            name="MarkerC",
            dye_row=1,
            alleles=(
                Allele(name="30", base_pair=150.0, left_bin=0.5, right_bin=0.5),
            ),
        ),
    ])


@pytest.fixture
def scaler() -> np.ndarray:
    """Linear scaler: scan position i maps to bp = i * 1.0."""
    return np.arange(300, dtype=float)


@pytest.fixture
def signal_image() -> np.ndarray:
    """2-dye, 300-point signal with some peaks."""
    signal = np.zeros((2, 300), dtype=float)
    signal[0, 98:102] = 500  # near bp=100 (MarkerA allele 10)
    signal[0, 198:202] = 300  # near bp=200 (MarkerB allele 20)
    signal[1, 148:152] = 400  # near bp=150 (MarkerC allele 30)
    return signal


# ---------------------------------------------------------------------------
# AlleleCaller ABC
# ---------------------------------------------------------------------------

class TestAllelCallerABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AlleleCaller()


# ---------------------------------------------------------------------------
# NearestBasePairCaller
# ---------------------------------------------------------------------------

class TestNearestBasePairCaller:
    def test_basic_call(self, simple_panel, scaler, signal_image):
        caller = NearestBasePairCaller(threshold=0.5)
        # Prediction matches signal peaks
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 98:102] = 1.0  # MarkerA allele 10
        pred[1, 148:152] = 1.0  # MarkerC allele 30

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        marker_names = {m.name for m in markers}
        assert "MarkerA" in marker_names
        assert "MarkerC" in marker_names

    def test_allele_name_resolution(self, simple_panel, scaler, signal_image):
        caller = NearestBasePairCaller(threshold=0.5)
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 99:101] = 1.0  # centered around bp=100 -> allele "10"

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        marker_a = [m for m in markers if m.name == "MarkerA"][0]
        allele_names = {a.name for a in marker_a.alleles}
        assert "10" in allele_names

    def test_rfu_extraction(self, simple_panel, scaler, signal_image):
        caller = NearestBasePairCaller(threshold=0.5)
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 98:102] = 1.0

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        marker_a = [m for m in markers if m.name == "MarkerA"][0]
        allele_10 = [a for a in marker_a.alleles if a.name == "10"][0]
        assert allele_10.height == 500

    def test_no_predictions(self, simple_panel, scaler, signal_image):
        caller = NearestBasePairCaller()
        pred = np.zeros((2, 300), dtype=float)
        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        assert len(markers) == 0

    def test_threshold_parameter(self, simple_panel, scaler, signal_image):
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 99:101] = 0.3  # below default threshold

        caller_default = NearestBasePairCaller(threshold=0.5)
        markers_default = caller_default.call_alleles(pred, signal_image, scaler, simple_panel)
        assert len(markers_default) == 0

        caller_low = NearestBasePairCaller(threshold=0.2)
        markers_low = caller_low.call_alleles(pred, signal_image, scaler, simple_panel)
        assert len(markers_low) > 0

    def test_multiple_connected_components(self, simple_panel, scaler, signal_image):
        """Two separate prediction regions on the same dye."""
        caller = NearestBasePairCaller()
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 99:101] = 1.0   # near bp=100 -> MarkerA allele 10
        pred[0, 199:201] = 1.0  # near bp=200 -> MarkerB allele 20

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        names = {m.name for m in markers}
        assert "MarkerA" in names
        assert "MarkerB" in names

    def test_exclude_non_autosomal(self):
        """Non-autosomal markers should be filtered when flag is set."""
        panel = Panel(markers=[
            Marker(
                name="AMEL",
                dye_row=0,
                alleles=(
                    Allele(name="X", base_pair=100.0, left_bin=0.5, right_bin=0.5),
                ),
            ),
            Marker(
                name="D5S818",
                dye_row=0,
                alleles=(
                    Allele(name="13", base_pair=200.0, left_bin=0.5, right_bin=0.5),
                ),
            ),
        ])
        scaler = np.arange(300, dtype=float)
        signal = np.ones((1, 300), dtype=float) * 100
        pred = np.zeros((1, 300), dtype=float)
        pred[0, 99:101] = 1.0  # near AMEL
        pred[0, 199:201] = 1.0  # near D5S818

        caller = NearestBasePairCaller(exclude_non_autosomal=True)
        markers = caller.call_alleles(pred, signal, scaler, panel)
        names = {m.name for m in markers}
        assert "AMEL" not in names
        assert "D5S818" in names

    def test_scaler_1d_and_2d(self, simple_panel, signal_image):
        """Should accept both 1D and 2D scalers."""
        caller = NearestBasePairCaller()
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 99:101] = 1.0

        scaler_1d = np.arange(300, dtype=float)
        scaler_2d = scaler_1d[np.newaxis, :]

        markers_1d = caller.call_alleles(pred, signal_image, scaler_1d, simple_panel)
        markers_2d = caller.call_alleles(pred, signal_image, scaler_2d, simple_panel)

        assert len(markers_1d) == len(markers_2d)

    def test_returns_tuple_of_markers(self, simple_panel, scaler, signal_image):
        caller = NearestBasePairCaller()
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 99:101] = 1.0
        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        assert isinstance(markers, tuple)
        assert all(isinstance(m, Marker) for m in markers)
