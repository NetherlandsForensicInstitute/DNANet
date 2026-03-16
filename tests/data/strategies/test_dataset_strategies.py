from pathlib import Path

import pytest

from DNAnet.data.data_models.dna_models import Allele, Marker
from DNAnet.data.data_models.structs import AlleleAnnotation
from DNAnet.data.strategies.dataset_strategies import (
    NFI_RND_DatasetStrategy,
    ProvedItDatasetStrategy,
)
from DNAnet.data.strategies.strategy_registry import StrategyRegistry


@pytest.fixture
def globalfiler_kit():
    StrategyRegistry.configure_kit('GLOBALFILER')


@pytest.fixture
def ppf6c_kit():
    StrategyRegistry.configure_kit('PPF6C')


def test_collect_nfi_rnd_files(ppf6c_kit):
    collected_files = NFI_RND_DatasetStrategy.collect_dataset_files(
        pytest.RESOURCES_DIR / 'profiles/RD'
    )
    collected_files = list(
        set(collected_files)
    )  # Use a set to prevent ordering issues with the assert

    assert len(collected_files) == 2, f'Incorrect amount of files found {len(collected_files)} != 2'
    assert Path(collected_files[0][0]).stem == '1A2_A01_01'
    assert collected_files[0][1] == '1L_11148_1A2'
    assert Path(collected_files[0][2]).stem == 'Ladder_G03_21'

def test_collect_provedit_files(globalfiler_kit):
    collected_files = ProvedItDatasetStrategy.collect_dataset_files(
        pytest.RESOURCES_DIR / 'profiles/RD'
    )
    collected_files = list(
        set(collected_files)
    )  # Use a set to prevent ordering issues with the assert



@pytest.mark.parametrize(
    'read_annotation_heights, first_marker',
    [
        (
            False,
            Marker(
                dye_row=0,
                name='D3S1358',
                alleles=[Allele('14'), Allele('15'), Allele('17')],
            ),
        ),
        (
            True,
            Marker(
                dye_row=0,
                name='D3S1358',
                alleles=[
                    Allele('14', height=2669.0),
                    Allele('15', height=7443.0),
                    Allele('17', height=5312.0),
                ],
            ),
        ),
    ],
)
def test_parse_nfi_rnd_annotations(read_annotation_heights, first_marker, ppf6c_kit):
    provedit_annotations = pytest.RESOURCES_DIR / 'profiles/RD/Dataset 1 DTH_AlleleReport.txt'
    sample_name = '2_09468_1A2'

    NFI_RND_DatasetStrategy.READ_ANNOTATION_HEIGHTS = read_annotation_heights
    annotation_mapping = NFI_RND_DatasetStrategy.parse_annotation_file(
        path=provedit_annotations,
    )

    assert annotation_mapping is not None

    annotation = NFI_RND_DatasetStrategy.create_annotation_for_sample(
        annotation_mapping, sample_name
    )

    assert annotation.annotation[1] == first_marker


def test_parse_provedit_annotations(globalfiler_kit):
    provedit_annotations = (
        pytest.RESOURCES_DIR / 'PROVEDIt_resources/PROVEDIt_RD14-0003 GF Known Genotypes.xlsx'
    )
    sample_name = 'A02_RD14-0003-44_45-1;1-M4d-0.031GF-Q0.9_01.5sec.hid'
    annotation_mapping = ProvedItDatasetStrategy.parse_annotation_file(
        path=provedit_annotations,
    )

    assert annotation_mapping is not None

    annotation = ProvedItDatasetStrategy.create_annotation_for_sample(
        annotation_mapping, sample_name
    )

    assert annotation.annotation[0] == Marker(
        dye_row=0,
        name='D3S1358',
        alleles=[
            Allele('15'),
            Allele('16'),
            Allele('15'),
            Allele('17'),
        ],
    )
