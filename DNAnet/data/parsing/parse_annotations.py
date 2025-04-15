import csv
import logging
import os
from itertools import groupby
from typing import Iterable, List, Optional, Sequence, Tuple

from DNAnet.data.data_models import Allele, Marker, Panel
from DNAnet.typing import PathLike


LOGGER = logging.getLogger("dnanet")


def parse_called_alleles(annotation_file_path: PathLike,
                         panel: Panel,
                         sample_name: str) \
        -> Optional[Sequence[Marker]]:
    """
    Reads annotation (csv) file to retrieve the manually called
    alleles for the sample with `sample_name`. These exist in slightly
    differing versions (txt, tsv, csv - allele/height ordering).

    :param annotation_file_path: path of annotation file
    :param panel: Panel to retrieve information of markers and alleles
    :param sample_name: name of the sample we want to retrieve alleles for
    :return: retrieved markers for this sample is possible/present
    """
    # Files can be empty.
    if os.stat(annotation_file_path).st_size == 0:
        LOGGER.debug(f'Found empty file: {annotation_file_path}')
        return None

    with open(annotation_file_path, 'r') as file:
        try:
            delimiter, allele_cols, height_cols = _parse_csv_header(file)
        except TypeError as e:
            LOGGER.debug(f'Type error for {annotation_file_path}: {e}')
            return None
        csv_file = csv.reader(file, delimiter=delimiter)
        for sample, results in groupby(csv_file, lambda x: x[0]):
            # there exist annotations for multiple hids in one file, so we must
            # select the correct one by checking the sample_name
            if sample == sample_name:
                return _parse_annotations(panel, results, allele_cols, height_cols)
        return None


def _parse_annotations(panel, results, allele_cols, height_cols) \
        -> List[Marker]:
    """
    Parses annotations for a single sample from annotations file
    :return: A list of `Marker` objects containing alleles for a single sample
    """
    markers = []
    for result in results:
        marker_name = result[1]
        dye_row = panel.get_dye_row(marker_name)
        if dye_row is not None:  # may be missing, e.g. Y-profile
            marker = Marker(dye_row,
                            marker_name,
                            [Allele(allele_name,
                                    *panel.get_allele_info(
                                        marker_name, allele_name),
                                    float(result[height_col]))
                             for allele_col, height_col in
                             zip(allele_cols, height_cols)
                             # 'OB_19.1' should be interpreted as '19.1'
                             if (allele_name := result[allele_col].strip('OB_'))])
        markers.append(marker)

    return markers


def _parse_csv_header(file) -> Tuple[str, Iterable[int], Iterable[int]]:
    """
    Retrieve delimiter and indices of columns that have allele names and
    peak heights within csv header

    :param file: file handler of csv file
    :return: delimiter, allele columns indices and height columns indices
    """
    header = next(file)
    for delimiter in [',', ';', '\t']:
        allele_cols = [i for i, column in enumerate(header.split(delimiter))
                       if column.startswith('Allele')]
        if len(allele_cols) > 0:
            height_cols = [i for i, column in
                           enumerate(header.split(delimiter))
                           if column.startswith('Height')]
            return delimiter, allele_cols, height_cols
    raise TypeError(f'No valid delimiter found for file: {file.name} with header {header}.')
