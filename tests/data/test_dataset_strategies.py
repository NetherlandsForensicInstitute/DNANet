"""Tests for concrete DatasetStrategy implementations."""

import pytest

from dnanet.data.strategies.datasets.nfi_rnd import NFIRnDStrategy
from dnanet.data.strategies.datasets.provedit import ProvedItStrategy


class TestNFIRnDStrategy:
    """Test NFI R&D dataset strategy file classification."""

    def test_categorize_sample(self):
        assert NFIRnDStrategy.categorize_file('1A2_A01_01.hid') == 'sample'
        assert NFIRnDStrategy.categorize_file('3B5_C03_12.hid') == 'sample'

    def test_categorize_ladder(self):
        assert NFIRnDStrategy.categorize_file('ladder_H03_24.hid') == 'ladder'
        assert NFIRnDStrategy.categorize_file('Ladder_G03_21.hid') == 'ladder'

    def test_categorize_control(self):
        assert NFIRnDStrategy.categorize_file('A01_blanco.hid') == 'control'
        assert NFIRnDStrategy.categorize_file('pocon_test.hid') == 'control'

    def test_get_number_of_contributors(self):
        assert NFIRnDStrategy.get_number_of_contributors('1A2_A01_01.hid') == 2
        assert NFIRnDStrategy.get_number_of_contributors('3B5_C03_12.hid') == 5

    def test_get_number_of_contributors_non_rd_raises(self):
        with pytest.raises(ValueError):
            NFIRnDStrategy.get_number_of_contributors('ladder_01.hid')

    def test_get_sample_id(self):
        assert NFIRnDStrategy.get_sample_id('1A2_A01_01.hid') == '1A2'

    def test_get_sample_id_invalid_raises(self):
        with pytest.raises(ValueError):
            NFIRnDStrategy.get_sample_id('ladder_01.hid')


class TestProvedItStrategy:
    """Test ProvedIt dataset strategy file classification."""

    def test_categorize_unknown(self):
        # This is a single REF profile, not supported in current implementation
        assert ProvedItStrategy.categorize_file(
            "B03_RD14-0003-34d1-0.5IP-Q0.75ng_05sec.hid"
        ) == "unknown"

    def test_categorize_sample(self):
        assert ProvedItStrategy.categorize_file(
            "A01_RD14-0003-36_37_38-1;2;1-M3e-0.06GF-Q1.6_01.5sec.hid"
        ) == "sample"

    def test_categorize_ladder(self):
        assert ProvedItStrategy.categorize_file('B03_Ladder-GF_02.5sec.hid') == 'ladder'

    def test_get_sample_id(self):
        sample_id = ProvedItStrategy.get_sample_id('B03_RD14-0003-34d1-0.5IP-Q0.75ng_05sec.hid')
        assert sample_id == 'RD14-0003-34d1-0.5IP-Q0.75ng'

    def test_get_number_of_contributors(self):
        noc = ProvedItStrategy.get_number_of_contributors(
            'A01_RD14-0003-36_37_38-1;2;1-M3e-0.06GF-Q1.6_01.5sec.hid'
        )
        assert noc == 3
