from DNAnet.data.data_models.structs import AlleleAnnotation, AllelePrediction
import pytest

from DNAnet.data.strategies.dataset_strategies import NFI_RND_DatasetStrategy
from DNAnet.data.strategies.strategy_registry import StrategyRegistry


@pytest.fixture
def ppf6c_mapping():
    StrategyRegistry.configure_kit('PPF6C')
    StrategyRegistry.configure_dataset(NFI_RND_DatasetStrategy)
    
    annotation_mapping = StrategyRegistry.get_dataset().parse_annotation_file(
        path=pytest.RESOURCES_DIR / 'profiles/RD/Dataset 1 DTH_AlleleReport.txt',
    )
    return annotation_mapping

def test_allele_annotation_from_file(ppf6c_mapping):
    sample_name = '1_11148_1A2'
    aa = AlleleAnnotation(annotation=ppf6c_mapping[sample_name])
    
    first_marker = aa.annotation[0]
    assert first_marker.name == 'AMEL'
    assert first_marker.alleles[0].name == "X"
    assert first_marker.alleles[0].base_pair is None
