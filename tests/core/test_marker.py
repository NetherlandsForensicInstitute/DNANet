"""Tests for the Marker composite."""

import pytest

from dnanet.core import Allele, Marker


class TestMarker:
    """Test suite for Marker."""

    def test_autosomal_marker(self, sample_marker):
        assert sample_marker.is_autosomal is True

    def test_amel_is_not_autosomal(self, amel_marker):
        assert amel_marker.is_autosomal is False

    def test_dys_is_not_autosomal(self):
        marker = Marker(name='DYS391', dye_row=3, alleles=frozenset())
        assert marker.is_autosomal is False

    def test_min_max_bp(self, sample_marker):
        assert sample_marker.min_bp < sample_marker.max_bp

    def test_min_bp_empty_alleles(self):
        marker = Marker(name='EMPTY', dye_row=0, alleles=frozenset())
        assert marker.min_bp == float('inf')
        assert marker.max_bp == float('-inf')

    def test_immutability(self, sample_marker):
        with pytest.raises(AttributeError):
            sample_marker.name = 'D5S818'

    def test_roundtrip_serialization(self, sample_marker):
        restored = Marker.from_dict(sample_marker.to_dict())
        assert restored == sample_marker
        assert restored.to_dict() == sample_marker.to_dict()
