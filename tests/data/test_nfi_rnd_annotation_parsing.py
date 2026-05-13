"""Tests for NFI R&D annotation parsing helpers."""

from pathlib import Path

import numpy as np

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.constants import LabelCategory
from dnanet.core.annotation import SpanAnnotation, AlleleAnnotation, ScanpointAnnotation
from dnanet.data.strategies.datasets.nfi_rnd import NFIRnDStrategy


class FakeScalingStrategy:
    """Minimal scaling strategy for marker-to-dye lookup in parser tests."""

    def marker_name_to_dye_idx(self) -> dict[str, int]:
        return {'AMEL': 0, 'D3S1358': 0, 'D21S11': 2}


class FakeSpanKit:
    """Minimal kit for scanpoint span annotation tests."""

    num_dyes = 3


class FakeSpanScalingStrategy(FakeScalingStrategy):
    """Minimal scaling strategy for scanpoint span annotation tests."""

    kit = FakeSpanKit()
    scanpoint_resolution = 10


def marker_alleles(annotation: AlleleAnnotation, marker_name: str) -> set[str]:
    marker = next(marker for marker in annotation.data if marker.name == marker_name)
    return {allele.name for allele in marker.alleles}


def write_reference_profile(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.write_text(
        'Marker;Allele1;Allele2\n'
        + ''.join(f'{marker};{allele1};{allele2}\n' for marker, allele1, allele2 in rows),
        encoding='utf-8',
    )


def test_reference_file_stems_for_prefix_uses_dataset_donor_order():
    assert NFIRnDStrategy._reference_file_stems_for_prefix('1A2') == ['1A', '1B']
    assert NFIRnDStrategy._reference_file_stems_for_prefix('3B5') == [
        '3K',
        '3L',
        '3M',
        '3N',
        '3O',
    ]


def test_parse_ground_truth_annotations_merges_donor_reference_profiles(tmp_path):
    references = tmp_path / 'References'
    references.mkdir()
    write_reference_profile(
        references / '1A.csv',
        [
            ('AMEL', 'X', 'Y'),
            ('D3S1358', '14', '15'),
        ],
    )
    write_reference_profile(
        references / '1B.csv',
        [
            ('AMEL', 'X', 'X'),
            ('D3S1358', '15', '17'),
            ('D21S11', '29', '30'),
        ],
    )

    annotations = NFIRnDStrategy._parse_ground_truth_annotations(
        tmp_path,
        [Path('1A2_A01_01.hid')],
        FakeScalingStrategy(),
    )

    annotation = annotations['1A2_A01_01']
    assert marker_alleles(annotation, 'AMEL') == {'X', 'Y'}
    assert marker_alleles(annotation, 'D3S1358') == {'14', '15', '17'}
    assert marker_alleles(annotation, 'D21S11') == {'29', '30'}


def test_parse_ground_truth_annotations_caches_donor_and_prefix_annotations(monkeypatch):
    calls: list[str] = []

    def fake_read_reference_profile(
        cls,
        reference_profiles_path: Path,
        marker_to_dye: dict[str, int],
    ) -> AlleleAnnotation:
        calls.append(reference_profiles_path.name)
        return AlleleAnnotation(
            [
                Marker(
                    name='AMEL',
                    dye_row=marker_to_dye['AMEL'],
                    alleles=frozenset({Allele(name=reference_profiles_path.stem)}),
                )
            ]
        )

    monkeypatch.setattr(
        NFIRnDStrategy,
        '_read_reference_profile',
        classmethod(fake_read_reference_profile),
    )

    annotations = NFIRnDStrategy._parse_ground_truth_annotations(
        Path('/unused'),
        [
            Path('1A2_A01_01.hid'),
            Path('1A2_A01_02.hid'),
            Path('1A3_A01_01.hid'),
        ],
        FakeScalingStrategy(),
    )

    assert calls == ['1A.csv', '1B.csv', '1C.csv']
    assert annotations['1A2_A01_01'] is annotations['1A2_A01_02']
    assert marker_alleles(annotations['1A3_A01_01'], 'AMEL') == {'1A', '1B', '1C'}


def test_parse_span_annotation_returns_scanpoint_annotations(tmp_path):
    span_annotations = tmp_path / 'span_annotations'
    span_annotations.mkdir()
    (span_annotations / 'annotations.csv').write_text(
        'user,date,profile,dye,x0,x1,peak_idx,category,version\n'
        'Jan,2025-11-18 12:35:14,1A2_A01_01.hid,blue,2,5,3,Allele,0.2\n'
        'Jan,2025-11-18 12:35:14,1A2_A01_01.hid,green,6,8,7,BleedThrough,0.2\n',
        encoding='utf-8',
    )

    annotations = NFIRnDStrategy._parse_span_annotation(tmp_path, FakeSpanScalingStrategy())

    annotation = annotations['1A2_A01_01']
    assert isinstance(annotation, SpanAnnotation)
    assert annotation.data.shape == (
        FakeSpanKit.num_dyes,
        FakeSpanScalingStrategy.scanpoint_resolution,
        12,
    )
    # Get class indices by argmax along the last dimension
    class_indices = np.argmax(annotation.data, axis=-1)
    allele_idx = list(LabelCategory).index(LabelCategory.ALLELE)
    # blue dye (index 0) has Allele annotation at positions 2-4
    assert np.all(class_indices[0, 2:5] == allele_idx)
    # green dye (index 1) has BleedThrough annotation at positions 6-7
    bleed_idx = list(LabelCategory).index(LabelCategory.BLEED_THROUGH)
    assert np.all(class_indices[1, 6:8] == bleed_idx)
    # position 1 should be unlabeled
    unlabeled_idx = list(LabelCategory).index(LabelCategory.UNLABELED)
    assert class_indices[0, 1] == unlabeled_idx


def test_parse_span_annotation_merges_multiple_annotators(monkeypatch, tmp_path):
    span_annotations = tmp_path / 'span_annotations'
    span_annotations.mkdir()
    (span_annotations / 'annotations.csv').write_text(
        'user,date,profile,dye,x0,x1,peak_idx,category,version\n'
        'Jan,2025-11-18 12:35:14,1A2_A01_01,blue,2,5,3,Allele,0.2\n'
        'Piet,2025-11-18 12:35:14,1A2_A01_01,blue,6,8,7,Stutter,0.2\n',
        encoding='utf-8',
    )
    merge_calls: list[int] = []

    def fake_merge_span_annotations(
        span_annotations: list[np.ndarray], hid_file_name: str
    ) -> np.ndarray:
        merge_calls.append(len(span_annotations))
        return np.maximum.reduce(span_annotations)

    monkeypatch.setattr(
        NFIRnDStrategy,
        '_merge_span_annotations',
        staticmethod(fake_merge_span_annotations),
    )

    annotations = NFIRnDStrategy._parse_span_annotation(tmp_path, FakeSpanScalingStrategy())

    annotation = annotations['1A2_A01_01']
    assert merge_calls == [2]
    assert annotation is not None
    class_indices = np.argmax(annotation.data, axis=-1)
    allele_idx = list(LabelCategory).index(LabelCategory.ALLELE)
    stutter_idx = list(LabelCategory).index(LabelCategory.STUTTER)
    assert np.all(class_indices[0, 2:5] == allele_idx)
    assert np.all(class_indices[0, 6:8] == stutter_idx)
