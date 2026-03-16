import csv
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional
from typing import List


from DNAnet.data.data_models.dna_models import Allele, Marker, Panel
from DNAnet.data.data_models.structs import AlleleAnnotation
from DNAnet.data.strategies.strategy_registry import StrategyRegistry

FileCategory = Literal["sample", "ladder", "control", "unknown"]


class DatasetStrategy(ABC):
    """
    Unified strategy interface for dataset-specific behavior such as
    file categorization, contributor parsing and allele loading.
    """

    # TODO: Do we really need an instance of this class for the load_donor_alleles function?
    def __init__(self, panel: Panel, genotypes_path: Path):
        self.panel = panel
        self.genotypes_path = Path(genotypes_path)

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
    def create_annotation_for_sample(cls, annotation_mapping: Dict[str, List[Marker]], sample_name: str) -> AlleleAnnotation:
        raise NotImplementedError

    @classmethod
    def build_marker(cls, marker_name: str, allele_names: Iterable[str], allele_heights: Optional[Iterable[float]] = None) -> Marker:
        _scaling_strategy = StrategyRegistry.get_scaling_strategy()
        dye_row = _scaling_strategy.panel.get_dye_row(marker_name)
        if dye_row is None:
            raise RuntimeError(
                f"Marker {marker_name} not found in panel {_scaling_strategy.panel}. "
                "Please check the panel or the marker name."
            )
        if allele_heights is None:
            allele_heights = [None] * len(allele_names)
        return Marker(
            dye_row=dye_row,
            name=marker_name,
            alleles=[
                Allele(name=allele_name, height=allele_height)
                for allele_name, allele_height in zip(allele_names, allele_heights, strict=True)
            ]
        )

    def serialize(self) -> dict:
        """Return a lightweight serialization of the strategy."""
        return {
            "class": self.__class__.__name__,
            "genotypes_path": str(self.genotypes_path),
            "panel_path": str(getattr(self.panel, "_panel_path", "")),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(genotypes_path={self.genotypes_path})"
