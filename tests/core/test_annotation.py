"""Tests for the annotation value objects."""

import numpy as np
import pytest

from dnanet.core.allele import Allele
from dnanet.core.annotation import AlleleAnnotation, ClassAnnotation, ScanpointAnnotation
from dnanet.core.marker import Marker


class TestClassAnnotation:
    def test_stores_label(self):
        ann = ClassAnnotation(data='Allele')
        assert ann.data == 'Allele'

    def test_is_immutable(self):
        ann = ClassAnnotation(data='Allele')
        with pytest.raises(AttributeError):
            ann.data = 'Stutter'


class TestScanpointAnnotation:
    def test_equality_with_image(self):
        img = np.ones((5, 100), dtype=np.float32)
        a = ScanpointAnnotation(data=img)
        b = ScanpointAnnotation(data=img.copy())
        assert a == b

    def test_immutability(self):
        ann = ScanpointAnnotation(data=np.zeros((5, 100), dtype=np.int8))
        with pytest.raises(AttributeError):
            ann.data = np.ones((5, 100), dtype=np.int8)


class TestAlleleAnnotation:
    def test_add_merges_matching_markers(self):
        marker_a = Marker(
            name='D3S1358',
            dye_row=0,
            alleles=frozenset(
                [
                    Allele(name='12', base_pair=120.0, left_bin=0.4, right_bin=0.4),
                ]
            ),
        )
        marker_b = Marker(
            name='D3S1358',
            dye_row=0,
            alleles=frozenset(
                [
                    Allele(name='13', base_pair=124.0, left_bin=0.4, right_bin=0.4),
                ]
            ),
        )

        merged = AlleleAnnotation(data=[marker_a]) + AlleleAnnotation(data=[marker_b])

        assert len(merged.data) == 1
        assert merged.data[0].name == 'D3S1358'
        assert {allele.name for allele in merged.data[0].alleles} == {'12', '13'}
