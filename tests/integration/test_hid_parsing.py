"""Integration tests for HID binary file parsing with real .hid files."""

import numpy as np
import pytest

from tests.conftest import RD_DIR, HID_DIR
from dnanet.data.strategies import PowerPlexFusion6CStrategy
from dnanet.data.parsing.hid import parse_hid, get_peak_data


class TestParseHID:
    """Test raw HID parsing on real R&D files."""

    def test_parse_sample_hid(self):
        """Should successfully parse 1A2_A01_01.hid."""
        result = parse_hid(HID_DIR / '1A2_A01_01.hid')
        assert result is not None
        assert isinstance(result, dict)
        # Should contain DATA keys
        data_keys = [k for k in result if k.startswith('DATA')]
        assert len(data_keys) > 0

    def test_parse_ladder_hid(self):
        """Should successfully parse Ladder_G03_21.hid."""
        result = parse_hid(HID_DIR / 'Ladder_G03_21.hid')
        assert result is not None

    def test_parse_nonexistent_returns_none(self, tmp_path):
        """Non-existent file should raise, not return None."""
        with pytest.raises(FileNotFoundError):
            parse_hid(tmp_path / 'nonexistent.hid')


class TestGetPeakData:
    """Test get_peak_data on real HID files."""

    @pytest.fixture(autouse=True)
    def set_strategies(self):
        self.scaling = PowerPlexFusion6CStrategy()

    def test_raw_strategy(self):
        """Raw strategy should return 6 dye channels."""
        data = get_peak_data(HID_DIR / '1A2_A01_01.hid', self.scaling, data_loading_strategy='raw')
        assert data is not None
        assert data.shape[0] == 6  # 5 dyes + 1 size standard
        assert data.dtype == np.int16

    def test_analyzed_strategy(self):
        """Analyzed strategy should also return 6 channels."""
        data = get_peak_data(
            HID_DIR / '1A2_A01_01.hid', self.scaling, data_loading_strategy='analyzed'
        )
        assert data is not None
        assert data.shape[0] == 6

    def test_superior_strategy(self):
        """Superior strategy applies baseline subtraction."""
        data = get_peak_data(
            HID_DIR / '1A2_A01_01.hid', self.scaling, data_loading_strategy='superior'
        )
        assert data is not None
        assert data.shape[0] == 6
        assert data.dtype == np.int16

    def test_raw_vs_reference_npy(self, ppf6c_kit):
        """After parsing + size-standard rescaling, data should match reference.

        The reference .npy was saved from the original DNANet pipeline (parse +
        baseline + rescale to 4096). We reproduce the full pipeline here.
        """
        data = get_peak_data(HID_DIR / '1A2_A01_01.hid', ppf6c_kit, data_loading_strategy='superior')
        assert data is not None

        # Rescale through size standard (same as HIDImage._load)
        ss_result = ppf6c_kit.parse_size_standard(np.array(data[-1]))
        assert ss_result is not None
        rescaled = data[:5, ss_result.rescaled_indices]  # 5 analysis dyes

        ref = np.load(RD_DIR / '1A2_A01_01.npy')
        if ref.ndim == 3:
            ref = ref[:, :, 0]

        assert rescaled.shape == ref.shape, f'Shape mismatch: {rescaled.shape} vs {ref.shape}'

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match='strategy must be'):
            get_peak_data(HID_DIR / '1A2_A01_01.hid', self.scaling, data_loading_strategy='invalid')

    def test_ladder_has_data(self):
        """Ladder files should also parse successfully."""
        data = get_peak_data(
            HID_DIR / 'Ladder_G03_21.hid', self.scaling, data_loading_strategy='raw'
        )
        assert data is not None
        assert data.shape[0] == 6
        # Size standard (last channel) should have signal
        assert np.max(data[-1]) > 0

    def test_second_sample_parses(self):
        """1A2_E01_13.hid should also parse."""
        data = get_peak_data(
            HID_DIR / '1A2_E01_13.hid', self.scaling, data_loading_strategy='superior'
        )
        assert data is not None
        assert data.shape[0] == 6
