import csv
import os
import re
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

from loguru import logger

from DNAnet.data.data_models.dna_models import Allele, Marker, Panel
from DNAnet.data.data_models.structs import AlleleAnnotation
from DNAnet.data.parsing.parse_annotations import _parse_csv_header
from DNAnet.data.strategies.dataset_strategies.Abstract_DatasetStrategy import (
    FileCategory,
    DatasetStrategy,
)
from DNAnet.data.strategies.strategy_registry import StrategyRegistry
from DNAnet.utils import (
    is_rd_hid_filename,
    get_prefix_from_filename,
)


class NFI_RND_DatasetStrategy(DatasetStrategy):
    """Strategy tailored to the NFI R&D dataset."""

    READ_ANNOTATION_HEIGHTS: bool = False
    DONORS_PER_DATASET_NR = {
        '1': ['A', 'B', 'C', 'D', 'E'],
        '2': ['F', 'G', 'H', 'I', 'J'],
        '3': ['K', 'L', 'M', 'N', 'O'],
        '4': ['P', 'Q', 'R', 'S', 'T'],
        '5': ['U', 'V', 'W', 'X', 'Y'],
        '6': ['Z', 'AA', 'AB', 'AC', 'AD'],
    }

    @classmethod
    def collect_dataset_files(
        cls, path: str | Path, **kwargs
    ) -> List[Tuple[Path, AlleleAnnotation | None, Path | None]]:
        """Collect the dataset files for this dataset with annotation and label mapping.

        Args:
            path: The root path of this dataset
            **kwargs: Additional arguments (none are used here)

        Raises:
            ValueError: When no mapping files could be found in the root
            RuntimeError: When it cannot be determined which annotation mapping column to use (DTH/DTL)

        Returns:
            A list of tuples consisting of:
                * Path to .HID file for the sample
                * AlleleAnnotation with Markers (optional)
                * Path to Ladder file for scaling (optional)
        """
        path = Path(path)
        csv_files = list(path.rglob('*.csv'))

        hid_to_annotation_file_pattern = r'.*hid_to_annotation.*'
        hid_to_annotation_path = None
        hid_to_ladder_pattern = r'.*best_ladder_paths.*'
        hid_to_ladder_path = None

        analysis_treshold_type: str = kwargs.get('analysis_treshold_type', 'DTH')

        for csv_file in csv_files:
            if re.match(hid_to_annotation_file_pattern, csv_file.name):
                hid_to_annotation_path = csv_file
            if re.match(hid_to_ladder_pattern, csv_file.name):
                hid_to_ladder_path = csv_file
        if hid_to_annotation_path is None or hid_to_ladder_path is None:
            raise ValueError(
                'Path does not contain the neccessary mapping files (annotation & ladder)'
            )

        # Allele Report for Annotations
        annotation_txt_files = list(path.rglob('*AlleleReport.txt'))
        annotation_mapping = {}
        for txt_file in annotation_txt_files:
            _annotation = cls.parse_annotation_file(txt_file)

            if _annotation:
                annotation_mapping.update(_annotation)

        # HID to Annotation mapping
        hta_header, hta_values = cls._read_csv_file(hid_to_annotation_path)
        analysis_treshold_type_column = [
            i for i, head in enumerate(hta_header) if analysis_treshold_type in head
        ]
        if len(analysis_treshold_type_column) != 1:
            raise RuntimeError(
                f'Could not infer the analysis treshold type column for annotation mapping: {hta_header}'
            )
        hid_to_annotation_sample = dict([
            (v[0].replace('.hid', ''), annotation_mapping.get(v[analysis_treshold_type_column[0]]))
            for v in hta_values
        ])


        # Hid to Ladder mapping
        _, htl_values = cls._read_csv_file(hid_to_ladder_path)
        hid_to_ladder = {hid: Path(ladder) for hid, ladder in htl_values}

        hid_files = list(path.rglob('*.hid'))
        hid_file_samples = list(filter(lambda x: cls.categorize_file(x.name) == 'sample', hid_files))

        return [
            (
                hid_file,
                cls.create_annotation_for_sample(hid_to_annotation_sample, hid_file.stem),
                hid_to_ladder.get(hid_file.stem)
            )
            for hid_file in hid_file_samples
        ]

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
        """Categorize a ProvedIt filename as ladder, control, or sample."""
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
        """Get the file name meta-information about number of contributors.

        Args:
            file_name: The filename to get information about

        Raises:
            ValueError: When the filename format is not in R&D format.

        Returns:
            A list of dataset number and contributor letter combinations.
        """
        if not is_rd_hid_filename(file_name):
            raise ValueError(
                f'Cannot load donor alleles for non-RD sample. Found file name {file_name}'
            )

        mixture_type = get_prefix_from_filename(file_name)
        dataset_nr, nr_donors = mixture_type[0], int(mixture_type[2])
        return [
            f'{dataset_nr}{letter}' for letter in cls.DONORS_PER_DATASET_NR[dataset_nr][:nr_donors]
        ]

    @classmethod
    def create_annotation_for_sample(
        cls, annotation_mapping: Dict[str, List[Marker]], sample_name: str
    ) -> AlleleAnnotation:
        """Create an AlleleAnnotation for a specific sample.

        Args:
            annotation_mapping: A mapping from sample ID to list of Markers
            sample_name: Name of the sample to build an Annotation for

        Returns:
            AlleleAnnotation with a list of Markers for the provided sample
        """
        return AlleleAnnotation(annotation=annotation_mapping[sample_name])

    @classmethod
    def parse_annotation_file(cls, path: str | Path) -> Optional[Dict[str, List[Marker]]]:
        """Parse the annotation file to a mapping from sample to list of Markers.

        Args:
            path: The path to the annotation file

        Returns:
            A mapping from sample ID to list of Markers
        """
        # Files can be empty.
        if os.stat(path).st_size == 0:
            logger.debug(f'Found empty file: {path}')
            return None

        scaling_strategy = StrategyRegistry.get_scaling_strategy()

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
                    if scaling_strategy.panel.get_dye_row(marker_name) is not None:  # may be missing, e.g. Y-profile
                        allele_heights: None | List[float] = None
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
        """Parse a csv files header.

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

    @staticmethod
    def load_donor_alleles(file_name: str, panel: Panel) -> List[Marker]:
        """
        For R&D files, we know the donors that contributed and the DNA profiles of the donors. For a
        single .hid file, find the donors (from the file name) and return the list of Markers of those
        donors combined.
        :param file_name: .hid file to load actual donors for
        :param panel: the panel to retrieve the dye row of the markers from
        """
        reference_path = "resources/data/2p_5p_Dataset_NFI/References"
        if not is_rd_hid_filename(file_name):
            raise ValueError("Cannot load donor alleles for non-RD sample. "
                             f"Found file name {file_name}")

        mixture_type = get_prefix_from_filename(file_name)  # to retrieve e.g. '1A2'
        dataset_nr, nr_donors = mixture_type[0], int(mixture_type[2])
        # one file contains alleles of one donor, so find files for all donors of the profile
        file_stems = [f"{dataset_nr}{letter}" for letter in
                      NFI_RND_DatasetStrategy.DONORS_PER_DATASET_NR[dataset_nr][:nr_donors]]

        # find the set of all alleles of the donors per marker
        marker_allele_strings = defaultdict(set)
        for file_stem in file_stems:
            reference_profiles_path = os.path.join(reference_path, f'{file_stem}.csv')
            with open(reference_profiles_path, "r") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    marker_allele_strings[row['Marker']].update([row['Allele1'], row['Allele2']])

        # transform into Marker/Allele objects
        markers = []
        for marker_name, alleles in marker_allele_strings.items():
            dye_row = panel.get_dye_row(marker_name)
            markers.append(Marker(dye_row, marker_name, [Allele(a) for a in sorted(alleles)]))

        return markers
