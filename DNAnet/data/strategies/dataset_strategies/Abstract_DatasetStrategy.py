from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Literal, Mapping, Iterable, Optional, Sequence
from pathlib import Path

from DNAnet.data.data_models.structs import AlleleAnnotation
from DNAnet.data.data_models.dna_models import Allele, Marker
from DNAnet.data.strategies.strategy_registry import StrategyRegistry


FileCategory = Literal['sample', 'ladder', 'control', 'unknown']


class DatasetStrategy(ABC):
    """Unified strategy interface for dataset-specific behavior.

    This can include file categorization, contributor parsing and allele loading.
    """

    @classmethod
    @abstractmethod
    def collect_dataset_files(
        cls, path: str | Path, **kwargs
    ) -> Tuple[List[Path], Mapping, Mapping]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        """Return the category (sample/ladder/control/unknown) for a given file name."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_contributors(cls, file_name: str) -> List[str]:
        """Derive contributor file stems from the HID filename."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def parse_annotation_file(cls, path: str | Path) -> Dict[str, List[Marker]] | None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_annotation_for_sample(
        cls, annotation_mapping: Dict[str, List[Marker]], sample_name: str
    ) -> AlleleAnnotation:
        raise NotImplementedError

    @classmethod
    def build_marker(
        cls,
        marker_name: str,
        allele_names: Iterable[str],
        allele_heights: Optional[Iterable[float]] = None,
    ) -> Marker:
        _kit = StrategyRegistry.get_kit()

        dye_row = _kit.panel.get_dye_row(marker_name)
        if dye_row is None:
            raise RuntimeError(
                f'Marker {marker_name} not found in panel {_kit.panel}. '
                'Please check the panel or the marker name.'
            )

        allele_names = list(allele_names)
        allele_heights_checked: Iterable[float | None]
        if allele_heights is None:
            allele_heights_checked = [None] * len(allele_names)
        else:
            allele_heights_checked = allele_heights

        return Marker(
            dye_row=dye_row,
            name=marker_name,
            alleles=[
                Allele(name=allele_name, height=allele_height)
                for allele_name, allele_height in zip(
                    allele_names, allele_heights_checked, strict=True
                )
            ],
        )
