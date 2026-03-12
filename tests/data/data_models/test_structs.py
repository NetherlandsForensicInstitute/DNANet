from DNAnet.data.data_models.structs import AlleleAnnotation, AllelePrediction
import pytest

from DNAnet.data.strategies.dataset_strategies import NFI_RND_DatasetStrategy
from DNAnet.data.strategies.strategy_registry import StrategyRegistry


@pytest.fixture
def ppf6c_strategy():
    StrategyRegistry.configure_kit('PPF6C')
    StrategyRegistry.configure_dataset(NFI_RND_DatasetStrategy)

def test_allele_annotation_from_file(ppf6c_strategy):
    aa = AlleleAnnotation(
        annotation=pytest.RESOURCES_DIR / 'profiles/RD/Dataset 1 DTH_AlleleReport.txt',
        sample_name='1_11148_1A2'
    )
    
    first_marker = aa.annotation[0]
    assert first_marker.name == 'AMEL'
    assert first_marker.alleles[0].name == "X"
    assert first_marker.alleles[0].base_pair is None
    
def test_allele_annotation_from_file_multiple_samples():
    with pytest.raises(ValueError, match="No sample name provided"):
        AlleleAnnotation(
            annotation=pytest.RESOURCES_DIR / 'profiles/RD/Dataset 1 DTH_AlleleReport.txt'
        )


def test_allele_prediction_from_file(ppf6c_strategy):
    aa = AllelePrediction(
        annotation=pytest.RESOURCES_DIR / 'profiles/RD/Dataset 1 DTH_AlleleReport.txt',
        sample_name='1_11148_1A2'
    )
    
    first_marker = aa.annotation[0]
    assert first_marker.name == 'AMEL'
    assert first_marker.alleles[0].name == "X"
    assert first_marker.alleles[0].base_pair is None