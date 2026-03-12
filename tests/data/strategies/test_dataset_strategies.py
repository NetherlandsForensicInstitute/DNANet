import pytest

from DNAnet.data.data_models.dna_models import Allele, Marker
from DNAnet.data.data_models.structs import AlleleAnnotation
from DNAnet.data.strategies.dataset_strategies import NFI_RND_DatasetStrategy, ProvedItDatasetStrategy
from DNAnet.data.strategies.strategy_registry import StrategyRegistry


@pytest.fixture
def globalfiler_kit():
    StrategyRegistry.configure_kit("GLOBALFILER")
    
@pytest.fixture
def ppf6c_kit():
    StrategyRegistry.configure_kit("PPF6C")


def test_parse_nfi_rnd_annotations(ppf6c_kit):
    provedit_annotations = pytest.RESOURCES_DIR / 'profiles/RD/Dataset 1 DTH_AlleleReport.txt'
    sample_name = "2_09468_1A2"
    annotation_mapping = NFI_RND_DatasetStrategy.parse_annotation_file(
        path=provedit_annotations,
    )
    
    assert annotation_mapping is not None
    
    annotation = NFI_RND_DatasetStrategy.create_annotation_for_sample(annotation_mapping, sample_name)
    
    assert annotation.annotation[1] == Marker(
        dye_row=0, name="D3S1358",
        alleles=[
            Allele("14", height=2669.),
            Allele("15", height=7443.),
            Allele("17", height=5312.),
        ]
    )
    
def test_parse_provedit_annotations(globalfiler_kit):
    provedit_annotations = pytest.RESOURCES_DIR / 'PROVEDIt_resources/PROVEDIt_RD14-0003 GF Known Genotypes.xlsx'
    sample_name = "A02_RD14-0003-44_45-1;1-M4d-0.031GF-Q0.9_01.5sec.hid"
    annotation_mapping = ProvedItDatasetStrategy.parse_annotation_file(
        path=provedit_annotations,
    )
    
    assert annotation_mapping is not None
    
    annotation = ProvedItDatasetStrategy.create_annotation_for_sample(annotation_mapping, sample_name)
    
    assert annotation.annotation[0] == Marker(
        dye_row=0, name="D3S1358",
        alleles=[
            Allele("15"),
            Allele("16"),
            Allele("15"),
            Allele("17"),
        ]
    )