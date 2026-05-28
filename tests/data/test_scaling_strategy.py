"""Extended tests for ScalingStrategy — GlobalFiler _attempt_fit edge cases."""

import numpy as np
import pytest

from dnanet.core import Panel
from dnanet.data.image import HIDImage
from dnanet.data.parsing.hid import get_peak_data
from dnanet.data.strategies.scaling.globalfiler import GlobalFilerStrategy
from dnanet.data.strategies.scaling.powerplex_fusion_6c import PowerPlexFusion6CStrategy, MEAN_DIFFS
from tests.conftest import PROVEDIT_DIR


# ---------------------------------------------------------------------------
# GlobalFiler _attempt_fit edge cases
# ---------------------------------------------------------------------------


class TestGlobalFilerAttemptFitEdgeCases:
    def test_too_few_peaks_handles_gracefully(self):
        """With only 2 peaks, should not crash."""
        peak_idxs = np.array([100, 200])
        expected_bps = np.array([60, 160, 260, 360, 460])
        result = GlobalFilerStrategy._attempt_fit(
            peak_idxs,
            expected_bps,
            threshold=5.0,
            max_shrinkages=10,
        )
        trimmed, bps, diff = result
        assert len(trimmed) <= 2
        assert len(bps) == len(trimmed)

    def test_single_peak_handled(self):
        """A single peak should return gracefully."""
        peak_idxs = np.array([100])
        expected_bps = np.array([60, 160, 260])
        result = GlobalFilerStrategy._attempt_fit(
            peak_idxs,
            expected_bps,
            threshold=5.0,
            max_shrinkages=10,
        )
        trimmed, bps, diff = result
        assert len(trimmed) <= 1

    def test_max_shrinkages_returns_best_fit(self):
        """When fit never converges, should return best rather than raise."""
        # Create deliberately bad data
        peak_idxs = np.array([100, 500, 1500, 2000, 3000])
        expected_bps = np.array([60, 80, 100, 120, 140])
        result = GlobalFilerStrategy._attempt_fit(
            peak_idxs,
            expected_bps,
            threshold=0.001,
            max_shrinkages=3,
        )
        trimmed, bps, diff = result
        assert trimmed is not None
        assert bps is not None
        assert len(trimmed) == len(bps)

    def test_shrinkage_reduces_length(self):
        """Each shrinkage should reduce the number of points used."""
        # Make data that needs exactly one shrinkage
        peak_idxs = np.array([100, 200, 300, 400, 500])
        expected_bps = np.array([60, 160, 260, 360, 9999])  # last one is bad

        result = GlobalFilerStrategy._attempt_fit(
            peak_idxs,
            expected_bps,
            threshold=5.0,
            max_shrinkages=10,
        )
        trimmed, bps, diff = result
        assert len(bps) < len(expected_bps)

    def test_perfect_quadratic_fit(self):
        """A perfect quadratic relationship should converge immediately."""
        x = np.array([100, 200, 300, 400, 500])
        # y = 0.001*x^2 + 0.5*x + 10  (quadratic)
        y = 0.001 * x**2 + 0.5 * x + 10
        result = GlobalFilerStrategy._attempt_fit(
            x,
            y,
            threshold=5.0,
            max_shrinkages=10,
        )
        _, _, diff = result
        assert diff < 0.01


# ---------------------------------------------------------------------------
# GlobalFiler parse_size_standard
# ---------------------------------------------------------------------------


class TestGlobalFilerParseSizeStandard:
    @pytest.fixture
    def gf(self):
        return GlobalFilerStrategy()

    def test_parse_with_real_provedit_data(self, gf):
        """Parse size standard from a real ProvedIt ladder file."""
        ladder_path = (
            PROVEDIT_DIR / '5 sec' / 'RD14-0003(020316ADG_5sec)' / 'A01_Ladder-GF_01.5sec.hid'
        )
        if not ladder_path.exists():
            pytest.skip('ProvedIt test resource not available')

        data = get_peak_data(ladder_path, gf, data_loading_strategy='raw')
        assert data is not None
        ss_lane = np.array(data[-1])
        result = gf.parse_size_standard(ss_lane)
        assert result is not None
        assert result.rescaled_indices.shape[0] == 4096
        assert result.scaler.shape[0] == 4096

    def test_parse_returns_correct_shapes(self, gf):
        """Even with imperfect data, result should have correct shapes."""
        ladder_path = (
            PROVEDIT_DIR / '5 sec' / 'RD14-0003(020316ADG_5sec)' / 'A01_Ladder-GF_01.5sec.hid'
        )
        if not ladder_path.exists():
            pytest.skip('ProvedIt test resource not available')

        data = get_peak_data(ladder_path, gf, data_loading_strategy='raw')
        result = gf.parse_size_standard(np.array(data[-1]))
        assert result.rescaled_indices.shape == (4096,)
        assert result.scaler.shape == (4096,)
        assert isinstance(result.fit_error, float)


# ---------------------------------------------------------------------------
# PPF6C validate_ss_peaks
# ---------------------------------------------------------------------------


class TestPPF6CValidateSSPeaks:
    def test_valid_19_peaks(self):
        """19 evenly spaced peaks should pass validation."""
        # Create peaks with ~10 pixels/bp ratio
        peak_idxs = np.arange(19) * 100 + 1000
        expected_bps = np.arange(19) * 10 + 60
        result = PowerPlexFusion6CStrategy._validate_ss_peaks(peak_idxs, expected_bps)
        assert result is True

    def test_wrong_count_fails(self):
        """18 peaks should fail (need exactly 19)."""
        result = PowerPlexFusion6CStrategy._validate_ss_peaks(
            np.arange(18) * 100 + 1000,
            np.arange(18) * 10 + 60,
        )
        assert result is False

    def test_bad_spacing_fails(self):
        """Peaks with pixel/bp ratio outside 7-13 should fail."""
        # Ratio of ~2 pixels/bp (too small)
        peak_idxs = np.arange(19) * 20 + 1000
        expected_bps = np.arange(19) * 10 + 60
        result = PowerPlexFusion6CStrategy._validate_ss_peaks(peak_idxs, expected_bps)
        assert result is False


@pytest.fixture
def ss_lane():
    ss_lane = np.zeros((9300,))
    peak_rfus = [50, 100, 200, 350, 700, 1100, 700, 350, 200, 100, 50]
    ss_lane[8595:8606] = peak_rfus
    idx = 8600
    for mean in MEAN_DIFFS[::-1]:
        new_idx = int(idx - mean)
        ss_lane[new_idx - 5: new_idx + 6] = peak_rfus
        idx = new_idx
    return ss_lane

class TestPPF6CExtractPeaks:
    def test_extract_peaks_happy(self, ss_lane):
        strategy = PowerPlexFusion6CStrategy()
        extracted_peaks = strategy._extract_ss_peaks(ss_lane)
        assert np.array_equal(extracted_peaks, np.array([4460, 4623, 4839, 5044, 5253, 5462, 5666, 5869, 6118, 6360, 6598, 6841, 7075, 7301, 7531, 7750, 7974, 8183, 8394]))

    def test_extract_peaks_too_high_peak(self, ss_lane):
        """Test that abnormal peak heights are not allowed."""
        idxs = np.where(ss_lane > 1000)[0]
        ss_lane[idxs[0]] = 30000
        strategy = PowerPlexFusion6CStrategy()
        assert strategy._extract_ss_peaks(ss_lane).size == 0

    def test_extract_peaks_too_low_peak(self, ss_lane):
        """Test that too low peaks are not allowed."""
        idxs = np.where(ss_lane > 1000)[0]
        ss_lane[idxs[0] - 5 : idxs[0] + 6] = [100, 150, 200, 250, 300, 350, 300, 250, 200, 150, 100]
        strategy = PowerPlexFusion6CStrategy()
        assert strategy._extract_ss_peaks(ss_lane).size == 0

    def test_extract_peaks_invalid_tail_peaks(self, ss_lane):
        """Test that when there is another tail peak, the correct one is selected."""
        idxs = np.where(ss_lane > 1000)[0]
        tail_idx = idxs[-1] + 400
        ss_lane[tail_idx - 5 : tail_idx + 6] = [50, 100, 200, 350, 700, 1100, 700, 350, 200, 100, 50]
        strategy = PowerPlexFusion6CStrategy()
        extracted_peaks = strategy._extract_ss_peaks(ss_lane)
        assert extracted_peaks.size == 19
        assert np.array_equal(extracted_peaks, np.array(
            [4460, 4623, 4839, 5044, 5253, 5462, 5666, 5869, 6118, 6360, 6598, 6841, 7075, 7301, 7531, 7750, 7974, 8183,
             8394]))

    def test_extract_peaks_dipped_peak(self, ss_lane):
        """Test that when we have a dipped peak, the peak top is correctly selected."""
        idxs = np.where(ss_lane > 1000)[0]
        ss_lane[idxs[0] - 5: idxs[0] + 6] = 0
        # Initially the 900 rfu value will be found, but this will be corrected later to the 1100 rfu value..
        ss_lane[idxs[0] - 50 : idxs[0] - 39] = [300, 550, 1100, 450, 500, 900, 500, 350, 200, 150, 100]
        strategy = PowerPlexFusion6CStrategy()
        extracted_peaks = strategy._extract_ss_peaks(ss_lane)
        assert extracted_peaks.size == 19
        assert np.all(ss_lane[extracted_peaks] == 1100)


def test_ss_parsing_hid_image_1A2():
    """Test that for the RD image 1A2_A01_01 the size standard peak extraction returns the expected peaks."""
    panel = Panel.from_xml("resources/kits/SGPanel_PPF6C.xml")
    strategy = PowerPlexFusion6CStrategy()
    hid_im = HIDImage(
        path="tests/resources/profiles/RD/1A2_A01_01.hid",
        adjusted_panel=panel,
        scaling_strategy=strategy,
        include_size_standard=True,
        data_loading_strategy="superior"
    )
    # Extract the raw data as we want the unparsed size standard lane.
    profile = get_peak_data(hid_im.path, hid_im.scaling_strategy, hid_im.data_loading_strategy)
    extracted_peaks = strategy._extract_ss_peaks(profile[-1])
    assert np.array_equal(extracted_peaks, np.array([3702, 3881, 4119, 4348, 4578, 4806, 5033, 5257, 5531, 5798, 6059,
       6323, 6579, 6827, 7077, 7318, 7560, 7788, 8018]))
    assert strategy.parse_size_standard(profile[-1]) is not None
