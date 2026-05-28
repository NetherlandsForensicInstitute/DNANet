"""Tests for allele calling from segmentation predictions."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from dnanet.core.panel import Panel
from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.evaluation.allele_caller import (
    AlleleCaller,
    ExactBasePairCaller,
    NearestBasePairCaller,
    FromSegmentationImageCaller,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_panel() -> Panel:
    """Panel with 2 markers on dye 0, 1 on dye 1."""
    return Panel(
        markers=[
            Marker(
                name='MarkerA',
                dye_row=0,
                alleles=frozenset(
                    [
                        Allele(name='10', base_pair=100.0, left_bin=0.5, right_bin=0.5),
                        Allele(name='11', base_pair=110.0, left_bin=0.5, right_bin=0.5),
                        Allele(name='12', base_pair=120.0, left_bin=0.5, right_bin=0.5),
                    ]
                ),
            ),
            Marker(
                name='MarkerB',
                dye_row=0,
                alleles=frozenset(
                    [
                        Allele(name='20', base_pair=200.0, left_bin=0.5, right_bin=0.5),
                        Allele(name='21', base_pair=210.0, left_bin=0.5, right_bin=0.5),
                    ]
                ),
            ),
            Marker(
                name='MarkerC',
                dye_row=1,
                alleles=frozenset(
                    [
                        Allele(name='30', base_pair=150.0, left_bin=0.5, right_bin=0.5),
                    ]
                ),
            ),
        ]
    )


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
# FromPredictionImageCaller
# ---------------------------------------------------------------------------


class DummyPredictionCaller(FromSegmentationImageCaller):
    def __init__(
        self,
        threshold: float = 0.5,
        exclude_non_autosomal: bool = False,
        prediction_mode: str = 'binary',
    ) -> None:
        super().__init__(
            threshold=threshold,
            exclude_non_autosomal=exclude_non_autosomal,
            prediction_mode=prediction_mode,
        )

    @staticmethod
    def call_allele_from_basepair(dye_index: int, base_pair: float, panel: Panel) -> Tuple[str, str]:
        if dye_index == 0:
            return 'MarkerA', 'AlleleA'
        return 'AMEL', 'X'


class TestFromPredictionImageCaller:
    def test_returns_tuple_of_markers(self, simple_panel, scaler, signal_image):
        caller = DummyPredictionCaller()
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 99:101] = 1.0
        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        assert isinstance(markers, tuple)
        assert all(isinstance(m, Marker) for m in markers)

    def test_threshold_parameter(self, simple_panel, scaler, signal_image):
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 99:101] = 0.3  # below default threshold

        caller_default = DummyPredictionCaller(threshold=0.5)
        markers_default = caller_default.call_alleles(pred, signal_image, scaler, simple_panel)
        assert len(markers_default) == 0

        caller_low = DummyPredictionCaller(threshold=0.2)
        markers_low = caller_low.call_alleles(pred, signal_image, scaler, simple_panel)
        assert len(markers_low) > 0

    def test_exclude_autosomal_markers(self, simple_panel, scaler, signal_image):
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 99:101] = 1
        pred[1, 99:101] = 1

        caller_exclude_true = DummyPredictionCaller(exclude_non_autosomal=True)
        markers = caller_exclude_true.call_alleles(pred, signal_image, scaler, simple_panel)
        assert [m.name for m in markers] == ['MarkerA']

        caller_exclude_false = DummyPredictionCaller(exclude_non_autosomal=False)
        markers = caller_exclude_false.call_alleles(pred, signal_image, scaler, simple_panel)
        assert sorted([m.name for m in markers]) == ['AMEL', 'MarkerA']

    def test_multiclass_mode_uses_only_allele_class(self, simple_panel, scaler, signal_image):
        caller = DummyPredictionCaller(prediction_mode='multiclass_labels')
        pred = np.zeros((2, 300), dtype=int)
        pred[0, 99:101] = 1
        pred[1, 149:151] = 2

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)

        assert len(markers) == 1
        assert markers[0].name == 'MarkerA'

    def test_invalid_prediction_mode_raises(self):
        with pytest.raises(ValueError, match='prediction_mode'):
            DummyPredictionCaller(prediction_mode='guess')

    def test_auto_mode_treats_float_predictions_as_binary(self, simple_panel, scaler, signal_image):
        caller = DummyPredictionCaller(prediction_mode='auto')
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 99:101] = 0.9

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)

        assert len(markers) == 1
        assert markers[0].name == 'MarkerA'

    def test_auto_mode_treats_boolean_predictions_as_binary(
        self, simple_panel, scaler, signal_image
    ):
        caller = DummyPredictionCaller(prediction_mode='auto')
        pred = np.zeros((2, 300), dtype=bool)
        pred[0, 99:101] = True

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)

        assert len(markers) == 1
        assert markers[0].name == 'MarkerA'

    def test_auto_mode_rejects_ambiguous_integer_predictions(
        self, simple_panel, scaler, signal_image
    ):
        caller = DummyPredictionCaller(prediction_mode='auto')
        pred = np.zeros((2, 300), dtype=int)
        pred[0, 99:101] = 1

        with pytest.raises(ValueError, match='ambiguous'):
            caller.call_alleles(pred, signal_image, scaler, simple_panel)

    def test_prediction_image_must_be_2d(self, simple_panel, scaler, signal_image):
        caller = DummyPredictionCaller()
        pred = np.zeros((2, 300, 2), dtype=float)

        with pytest.raises(ValueError, match='2-D array'):
            caller.call_alleles(pred, signal_image, scaler, simple_panel)


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
        assert 'MarkerA' in marker_names
        assert 'MarkerC' in marker_names

    def test_allele_name_resolution(self, simple_panel, scaler, signal_image):
        caller = NearestBasePairCaller(threshold=0.5)
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 99:101] = 1.0  # centered around bp=100 -> allele "10"

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        marker_a = [m for m in markers if m.name == 'MarkerA'][0]
        allele_names = {a.name for a in marker_a.alleles}
        assert '10' in allele_names

    def test_multiclass_labels_call_only_allele_class(self, simple_panel, scaler, signal_image):
        caller = NearestBasePairCaller(prediction_mode='multiclass_labels')
        pred = np.zeros((2, 300), dtype=int)
        pred[0, 99:101] = 1
        pred[1, 148:152] = 2

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)

        assert {m.name for m in markers} == {'MarkerA'}

    def test_auto_mode_rejects_ambiguous_integer_predictions(
        self, simple_panel, scaler, signal_image
    ):
        caller = NearestBasePairCaller(prediction_mode='auto')
        pred = np.zeros((2, 300), dtype=int)
        pred[0, 99:101] = 1

        with pytest.raises(ValueError, match='ambiguous'):
            caller.call_alleles(pred, signal_image, scaler, simple_panel)

    def test_rfu_extraction(self, simple_panel, scaler, signal_image):
        caller = NearestBasePairCaller(threshold=0.5)
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 98:102] = 1.0

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        marker_a = [m for m in markers if m.name == 'MarkerA'][0]
        allele_10 = [a for a in marker_a.alleles if a.name == '10'][0]
        assert allele_10.height == 500

    def test_no_predictions(self, simple_panel, scaler, signal_image):
        caller = NearestBasePairCaller()
        pred = np.zeros((2, 300), dtype=float)
        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        assert len(markers) == 0

    def test_multiple_connected_components(self, simple_panel, scaler, signal_image):
        """Two separate prediction regions on the same dye."""
        caller = NearestBasePairCaller()
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 99:101] = 1.0  # near bp=100 -> MarkerA allele 10
        pred[0, 199:201] = 1.0  # near bp=200 -> MarkerB allele 20

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        names = {m.name for m in markers}
        assert 'MarkerA' in names
        assert 'MarkerB' in names

    def test_exclude_non_autosomal(self):
        """Non-autosomal markers should be filtered when flag is set."""
        panel = Panel(
            markers=[
                Marker(
                    name='AMEL',
                    dye_row=0,
                    alleles=frozenset(
                        [
                            Allele(name='X', base_pair=100.0, left_bin=0.5, right_bin=0.5),
                        ]
                    ),
                ),
                Marker(
                    name='D5S818',
                    dye_row=0,
                    alleles=frozenset(
                        [
                            Allele(name='13', base_pair=200.0, left_bin=0.5, right_bin=0.5),
                        ]
                    ),
                ),
            ]
        )
        scaler = np.arange(300, dtype=float)
        signal = np.ones((1, 300), dtype=float) * 100
        pred = np.zeros((1, 300), dtype=float)
        pred[0, 99:101] = 1.0  # near AMEL
        pred[0, 199:201] = 1.0  # near D5S818

        caller = NearestBasePairCaller(exclude_non_autosomal=True)
        markers = caller.call_alleles(pred, signal, scaler, panel)
        names = {m.name for m in markers}
        assert 'AMEL' not in names
        assert 'D5S818' in names

    def test_no_dye_mapping(self, simple_panel, scaler):
        signal_image = np.zeros((3, 300), dtype=float)
        signal_image[2, 98:102] = 500

        caller = NearestBasePairCaller(threshold=0.5)
        pred = np.zeros((3, 300), dtype=float)
        pred[2, 100:102] = 1.0

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        assert len(markers) == 1
        called_alleles = [a.name for m in markers for a in m.alleles]
        assert len(called_alleles) == 1
        assert called_alleles[0] == 'Unknown'


class TestExactBasePairCaller:
    def test_basic_call(self, simple_panel, scaler, signal_image):
        caller = ExactBasePairCaller()
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 100:102] = 1.0  # centered around bp=101 -> allele "10"

        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        marker_a = [m for m in markers if m.name == 'MarkerA'][0]
        allele_names = {a.name for a in marker_a.alleles}
        assert '10' in allele_names

    def test_call_ob_allele(self, simple_panel, scaler, signal_image):
        caller = ExactBasePairCaller()
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 105:106] = 1.0
        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)

        assert len(markers) == 1
        assert markers[0].name == 'MarkerA'

        called_alleles = [a for a in markers[0].alleles]
        assert len(called_alleles) == 1
        assert called_alleles[0].name == 'Out of Bin'

    def test_call_ob_allele_and_marker(self, simple_panel, scaler, signal_image):
        caller = ExactBasePairCaller()
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 95:98] = 1.0
        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)

        assert len(markers) == 1
        assert markers[0].name == 'Out of Bin'

        called_alleles = [a for a in markers[0].alleles]
        assert len(called_alleles) == 1
        assert called_alleles[0].name == 'Out of Bin'

    def test_call_multiple_ob_peaks(self, simple_panel, scaler, signal_image):
        caller = ExactBasePairCaller()
        pred = np.zeros((2, 300), dtype=float)
        pred[0, 95:98] = 1.0
        pred[1, 10:20] = 1.0
        markers = caller.call_alleles(pred, signal_image, scaler, simple_panel)
        assert len(markers) == 2

        called_alleles = [a.name for m in markers for a in m.alleles]
        assert len(called_alleles) == 2
        assert set(called_alleles) == {'Out of Bin'}
