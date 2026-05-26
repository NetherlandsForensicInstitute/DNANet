"""Tests for the Allele value object."""

import pytest

from dnanet.core import Allele


class TestAllele:
    """Test suite for Allele."""

    def test_creation(self):
        allele = Allele(name='12', base_pair=120.0, left_bin=0.4, right_bin=0.4)
        assert allele.name == '12'
        assert allele.base_pair == 120.0

    def test_defaults(self):
        allele = Allele(name='X')
        assert allele.base_pair is None
        assert allele.height is None

    def test_immutability(self):
        allele = Allele(name='12', base_pair=120.0)
        with pytest.raises(AttributeError):
            allele.name = '13'

    def test_equality_by_value(self):
        a = Allele(name='12', base_pair=120.0)
        b = Allele(name='12', base_pair=120.0)
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality(self):
        a = Allele(name='12', base_pair=120.0)
        b = Allele(name='13', base_pair=130.0)
        assert a != b

    def test_roundtrip_serialization(self, sample_alleles):
        for allele in sample_alleles:
            restored = Allele.from_dict(allele.to_dict())
            assert restored == allele

    def test_from_dict_missing_optional_fields(self):
        allele = Allele.from_dict({'name': 'X'})
        assert allele.name == 'X'
        assert allele.base_pair is None

    def test_usable_in_sets(self):
        a = Allele(name='12', base_pair=120.0)
        b = Allele(name='12', base_pair=120.0)
        assert len({a, b}) == 1
