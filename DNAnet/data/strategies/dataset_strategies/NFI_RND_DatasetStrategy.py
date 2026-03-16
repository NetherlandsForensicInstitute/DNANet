from collections import defaultdict

from DNAnet.data.data_models.dna_models import Allele, Marker
from DNAnet.data.data_models.structs import AlleleAnnotation
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class NFI_RND_DatasetStrategy(DatasetStrategy):
    """
    Strategy tailored to the NFI R&D dataset.
    """

    READ_ANNOTATION_HEIGHTS: bool = False

    @classmethod
    def collect_dataset_files(cls, path: str | Path, **kwargs) -> Sequence[Tuple[str, str, str]]:
        path = Path(path)
        csv_files = list(path.rglob('*.csv'))

        hid_to_annotation_file_pattern = r'.*hid_to_annotation.*'
        hid_to_annotation_path = None
        hid_to_ladder_pattern = r'.*best_ladder_paths.*'
        hid_to_ladder_path = None

        analysis_treshold_type: str = kwargs.get('analysis_treshold_type', 'DTL')

        for csv_file in csv_files:
            if re.match(hid_to_annotation_file_pattern, csv_file.name):
                hid_to_annotation_path = csv_file
            if re.match(hid_to_ladder_pattern, csv_file.name):
                hid_to_ladder_path = csv_file
        if hid_to_annotation_path is None or hid_to_ladder_path is None:
            raise ValueError(
                'Path does not contain the neccessary mapping files (annotation & ladder)'
            )
        # Hid to Annotation mapping
        hta_header, hta_values = cls._read_csv_file(hid_to_annotation_path)
        analysis_treshold_type_column = [
            i for i, head in enumerate(hta_header) if analysis_treshold_type in head
        ]
        if len(analysis_treshold_type_column) != 1:
            raise RuntimeError(
                f'Could not infer the analysis treshold type column for annotation mapping: {hta_header}'
            )
        hid_to_annotation = {col[0]: col[analysis_treshold_type_column[0]] for col in hta_values}
        # Hid to Ladder mapping
        _, htl_values = cls._read_csv_file(hid_to_ladder_path)
        hid_to_ladder = {hid: ladder for hid, ladder in htl_values}

        hid_files = list(path.rglob('*.hid'))
        samples: List[Tuple[str, str, str]] = []
        for hid_file in hid_files:
            hid_file_category = cls.categorize_file(hid_file.name)
            if hid_file_category != 'sample':
                logger.debug(f'Skipping HID file (not a sample): {hid_file}')
                continue
            hid_file_annotation = hid_to_annotation.get(
                hid_file.name
            )  # The annotation mapping uses the .hid
            hid_file_ladder = hid_to_ladder.get(
                hid_file.stem
            )  # The ladder mapping uses without .hid
            samples.append((str(hid_file), hid_file_annotation, hid_file_ladder))

        return samples

    @classmethod
    def _read_csv_file(cls, csv_file: str | Path) -> Tuple[List[str], List[List[str]]]:
        """Read a csv file with a header row.

        Args:
            csv_file: The path to the csv file to be read

        Returns:
            Tuple with the header row and (filtered on empty) value rows.
        """
        with open(csv_file, 'r') as htap:
            reader = csv.reader(htap)
            rows = list(reader)
        headers, values = rows[0], filter(lambda r: len(r) > 0, rows[1:])
        return headers, list(values)

    @classmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        OTHER_KITS = ('ppy23', 'minifiler', 'hdplex')

        fname = file_name.lower()
        # Ladder files
        if 'ladder' in fname and not any(kit in fname for kit in OTHER_KITS):
            return 'ladder'
        # Controls/blanks (implement your own logic or import from utils)
        if (
            'blanco' in fname
            or 'ladder' in fname
            or 'pocon' in fname
            or 'controle' in fname
            or fname.startswith('a')
        ):
            return 'control'
        # Valid sample HID file (using is_rd_hid_filename logic)
        if len(re.findall(r'\d[ABCDEF]\d', file_name[:3])) > 0:
            return 'sample'
        # Unknown or unhandled
        return 'unknown'

    @classmethod
    def get_contributors(cls, file_name: str) -> list[str]:
        if not is_rd_hid_filename(file_name):
            raise ValueError(
                f'Cannot load donor alleles for non-RD sample. Found file name {file_name}'
            )

        mixture_type = get_prefix_from_filename(file_name)
        dataset_nr, nr_donors = mixture_type[0], int(mixture_type[2])
        return [f'{dataset_nr}{letter}' for letter in DONORS_PER_DATASET_NR[dataset_nr][:nr_donors]]

    @classmethod
    def create_annotation_for_sample(
        cls, annotation_mapping: Dict[str, List[Marker]], sample_name: str
    ) -> AlleleAnnotation:
        return AlleleAnnotation(annotation=annotation_mapping[sample_name])

    @classmethod
    def parse_annotation_file(cls, path: str | Path) -> Optional[Dict[str, List[Marker]]]:
        # Files can be empty.
        if os.stat(path).st_size == 0:
            logger.debug(f'Found empty file: {path}')
            return None

        kit = StrategyRegistry.get_kit()

        markers = defaultdict(list)
        with open(path, 'r') as file:
            try:
                delimiter, allele_cols, height_cols = _parse_csv_header(file)
            except TypeError as e:
                logger.debug(f'Type error for {path}: {e}')
                return None
            csv_file = csv.reader(file, delimiter=delimiter)
            for sample, results in groupby(csv_file, lambda x: x[0]):
                for result in results:
                    marker_name = result[1]
                    dye_row = kit.panel.get_dye_row(marker_name)
                    if dye_row is not None:  # may be missing, e.g. Y-profile
                        allele_names, allele_heights = map(
                            list,
                            zip(
                                *[
                                    (allele_name, float(result[height_col]))
                                    for allele_col, height_col in zip(
                                        allele_cols, height_cols, strict=True
                                    )
                                    if (allele_name := result[allele_col].strip('_OB')) != ''
                                ],
                                strict=True,
                            ),
                        )

                        if cls.READ_ANNOTATION_HEIGHTS:
                            logger.warning(
                                'Reading annotation RFU heights. Beware of RFU variations based on preprocessing during annotation.'
                            )
                        else:
                            allele_heights = None

                        marker = cls.build_marker(
                            marker_name=marker_name,
                            allele_names=allele_names,
                            allele_heights=allele_heights,
                        )
                        markers[sample].append(marker)

        return dict(markers)

    @classmethod
    def _parse_csv_header(cls, file) -> Tuple[str, Iterable[int], Iterable[int]]:
        """
        Retrieve delimiter and indices of columns that have allele names and
        peak heights within csv header

        :param file: file handler of csv file
        :return: delimiter, allele columns indices and height columns indices
        """
        header = next(file)
        for delimiter in [',', ';', '\t']:
            allele_cols = [
                i for i, column in enumerate(header.split(delimiter)) if column.startswith('Allele')
            ]
            if len(allele_cols) > 0:
                height_cols = [
                    i
                    for i, column in enumerate(header.split(delimiter))
                    if column.startswith('Height')
                ]
                return delimiter, allele_cols, height_cols
        raise TypeError(f'No valid delimiter found for file: {file.name} with header {header}.')
