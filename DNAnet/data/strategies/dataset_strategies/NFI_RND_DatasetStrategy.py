from DNAnet.data.data_models.dna_models import Allele, Marker
from DNAnet.data.parsing.parse_annotations import _parse_csv_header
from DNAnet.data.strategies.dataset_strategies.Abstract_DatasetStrategy import (
    DatasetStrategy,
    FileCategory,
)
from DNAnet.data.strategies.strategy_registry import StrategyRegistry
from DNAnet.utils import (
    DONORS_PER_DATASET_NR,
    get_prefix_from_filename,
    is_rd_hid_filename,
)


from loguru import logger


import csv
import os
import re
from itertools import groupby
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


class NFI_RND_DatasetStrategy(DatasetStrategy):
    """
    Strategy tailored to the NFI R&D dataset.
    """

    @classmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        OTHER_KITS = ("ppy23", "minifiler", "hdplex")

        fname = file_name.lower()
        # Ladder files
        if "ladder" in fname and not any(kit in fname for kit in OTHER_KITS):
            return "ladder"
        # Controls/blanks (implement your own logic or import from utils)
        if (
            "blanco" in fname
            or "ladder" in fname
            or "pocon" in fname
            or "controle" in fname
            or fname.startswith("a")
        ):
            return "control"
        # Valid sample HID file (using is_rd_hid_filename logic)
        if len(re.findall(r"\d[ABCDEF]\d", file_name[:3])) > 0:
            return "sample"
        # Unknown or unhandled
        return "unknown"

    @classmethod
    def get_contributors(cls, file_name: str) -> list[str]:
        if not is_rd_hid_filename(file_name):
            raise ValueError(
                f"Cannot load donor alleles for non-RD sample. Found file name {file_name}"
            )

        mixture_type = get_prefix_from_filename(file_name)
        dataset_nr, nr_donors = mixture_type[0], int(mixture_type[2])
        return [
            f"{dataset_nr}{letter}"
            for letter in DONORS_PER_DATASET_NR[dataset_nr][:nr_donors]
        ]

    # From previous implementation, but I believe this code is broken or deprecated.
    def build_marker(self, marker_name: str, allele_names: Iterable[str]) -> Marker:
        dye_row = self.panel.get_dye_row(marker_name)
        if not dye_row:
            raise RuntimeError(f"Could not retrieve dye row for {marker_name}")
        return Marker(dye_row, marker_name, [Allele(a) for a in sorted(allele_names)])

    @classmethod
    def parse_annotation_file(
        cls, path: str | Path, sample_name: str | None = None
    ) -> Optional[List[Marker]]:
        # Files can be empty.
        if os.stat(path).st_size == 0:
            logger.debug(f"Found empty file: {path}")
            return None

        kit = StrategyRegistry.get_kit()

        provided_sample_name = sample_name
        markers = []
        with open(path, "r") as file:
            try:
                delimiter, allele_cols, height_cols = _parse_csv_header(file)
            except TypeError as e:
                logger.debug(f"Type error for {path}: {e}")
                return None
            csv_file = csv.reader(file, delimiter=delimiter)
            for sample, results in groupby(csv_file, lambda x: x[0]):
                # If no sample name is provided, we save the first we see
                if sample_name is None:
                    sample_name = sample

                if sample == sample_name:
                    for result in results:
                        marker_name = result[1]
                        dye_row = kit.panel.get_dye_row(marker_name)
                        if dye_row is not None:  # may be missing, e.g. Y-profile
                            marker = Marker(
                                dye_row,
                                marker_name,
                                [
                                    Allele(
                                        name=allele_name,
                                        height=float(result[height_col]),
                                        # TODO: Do we want to have an adjusted panel already here?
                                    )
                                    for allele_col, height_col in zip(
                                        allele_cols, height_cols
                                    )
                                    # 'OB_19.1' should be interpreted as '19.1'
                                    if (allele_name := result[allele_col].strip("OB_"))
                                ],
                            )
                            markers.append(marker)
                elif provided_sample_name is None and sample != sample_name:
                    raise ValueError(
                        "No sample name provided but the file contains multiple!"
                    )
        return markers

    @classmethod
    def _parse_csv_header(cls, file) -> Tuple[str, Iterable[int], Iterable[int]]:
        """
        Retrieve delimiter and indices of columns that have allele names and
        peak heights within csv header

        :param file: file handler of csv file
        :return: delimiter, allele columns indices and height columns indices
        """
        header = next(file)
        for delimiter in [",", ";", "\t"]:
            allele_cols = [
                i
                for i, column in enumerate(header.split(delimiter))
                if column.startswith("Allele")
            ]
            if len(allele_cols) > 0:
                height_cols = [
                    i
                    for i, column in enumerate(header.split(delimiter))
                    if column.startswith("Height")
                ]
                return delimiter, allele_cols, height_cols
        raise TypeError(
            f"No valid delimiter found for file: {file.name} with header {header}."
        )
