"""Tests for the Ladder class."""

import numpy as np

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.panel import Panel
from dnanet.data.ladders.ladder import Ladder
from dnanet.data.ladders.ladder_allele_catalog import LadderAlleleCatalog


class TestLadderAlleleCatalog:
    def test_from_dict(self):
        catalog = LadderAlleleCatalog(
            alleles_by_dye={
                0: [('AMEL', 'X'), ('AMEL', 'Y'), ('D3S1358', '14')],
                1: [('TH01', '6'), ('TH01', '9.3')],
            }
        )
        assert catalog.expected_count(0) == 3
        assert catalog.expected_count(1) == 2
        assert catalog.expected_count(2) == 0  # unknown dye


class TestLadder:
    def _make_catalog_and_panel(self, num_dyes=2, peaks_per_dye=3):
        """Create a simple catalog and panel for testing."""
        alleles_by_dye = {}
        markers = []

        for dye in range(num_dyes):
            marker_name = f'M{dye}'
            allele_names = [f'A{i}' for i in range(peaks_per_dye)]
            alleles_by_dye[dye] = [(marker_name, a) for a in allele_names]
            markers.append(
                Marker(
                    name=marker_name,
                    dye_row=dye,
                    alleles=frozenset(
                        Allele(
                            name=a,
                            base_pair=100.0 + i * 50.0,
                            left_bin=0.5,
                            right_bin=0.5,
                        )
                        for i, a in enumerate(allele_names)
                    ),
                )
            )

        catalog = LadderAlleleCatalog(alleles_by_dye=alleles_by_dye)
        panel = Panel(markers=tuple(markers))
        return catalog, panel

    def test_successful_ladder(self):
        """Test ladder with perfectly matching peak counts."""
        catalog, panel = self._make_catalog_and_panel(num_dyes=2, peaks_per_dye=3)

        # Create synthetic data with 3 peaks per dye
        signal_len = 1000
        data = np.zeros((2, signal_len, 1))
        for dye in range(2):
            for _, pos in enumerate([200, 500, 800]):
                data[dye, pos, 0] = 1000  # tall peak

        scaler = np.linspace(50, 300, signal_len)

        peak_indices = Ladder._find_all_peaks(
            data=data,
            catalog=catalog,
            num_dyes=2,
            path='test_ladder.hid',
        )
        assert peak_indices is not None

        adjusted_panel = Ladder._adjust_panel(
            peak_indices,
            catalog=catalog,
            scaler=scaler,
            default_panel=panel,
            num_dyes=2,
        )
        assert adjusted_panel is not None

    # def test_mismatched_peaks_returns_none(self):
    #     """If peak count doesn't match catalog, return None."""
    #     catalog, panel = self._make_catalog_and_panel(num_dyes=2, peaks_per_dye=3)

    #     # Only put 2 peaks on dye 0 (expected 3)
    #     signal_len = 1000
    #     data = np.zeros((2, signal_len, 1))
    #     data[0, 200, 0] = 1000
    #     data[0, 500, 0] = 1000
    #     # Dye 1 has 3 peaks
    #     for pos in [200, 500, 800]:
    #         data[1, pos, 0] = 1000

    #     scaler = np.linspace(50, 300, signal_len)

    #     ladder = Ladder(
    #         path="bad_ladder.hid",
    #         data=data,
    #         scaler=scaler,
    #         catalog=catalog,
    #         default_panel=panel,
    #         num_dyes=2,
    #     )
    #     assert ladder is None

    # def test_repr(self):
    #     catalog, panel = self._make_catalog_and_panel(num_dyes=1, peaks_per_dye=2)
    #     signal_len = 1000
    #     data = np.zeros((1, signal_len, 1))
    #     data[0, 300, 0] = 1000
    #     data[0, 700, 0] = 1000
    #     scaler = np.linspace(50, 300, signal_len)

    #     ladder = Ladder.from_hid_data(
    #         path="my_ladder.hid", data=data, scaler=scaler,
    #         catalog=catalog, default_panel=panel, num_dyes=1,
    #     )
    #     assert ladder is not None
    #     assert "my_ladder.hid" in repr(ladder)
    #     assert "adjusted" in repr(ladder)
