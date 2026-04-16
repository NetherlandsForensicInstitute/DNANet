"""Tests for NFI R&D annotation parsing helpers."""

from pathlib import Path

from dnanet.core.allele import Allele
from dnanet.core.annotation import AlleleAnnotation
from dnanet.core.marker import Marker
from dnanet.data.strategies.datasets.nfi_rnd import NFIRnDStrategy


class FakeScalingStrategy:
    """Minimal scaling strategy for marker-to-dye lookup in parser tests."""

    def marker_name_to_dye_idx(self) -> dict[str, int]:
        return {'AMEL': 0, 'D3S1358': 0, 'D21S11': 2}


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
