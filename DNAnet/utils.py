import csv
import dataclasses
import json
import logging
import os
import re
from collections import defaultdict
from itertools import islice
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

from DNAnet.data.data_models import Allele, Marker, Panel
from DNAnet.typing import PathLike


LOGGER = logging.getLogger("dnanet")
LOGGER.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
LOGGER.addHandler(console_handler)


DONORS_PER_DATASET_NR = {'1': ['A', 'B', 'C', 'D', 'E'],
                         '2': ['F', 'G', 'H', 'I', 'J'],
                         '3': ['K', 'L', 'M', 'N', 'O'],
                         '4': ['P', 'Q', 'R', 'S', 'T'],
                         '5': ['U', 'V', 'W', 'X', 'Y'],
                         '6': ['Z', 'AA', 'AB', 'AC', 'AD'],
                         }


def is_rd_hid_filename(file_name: str) -> bool:
    """
    Perform simple checks on whether this may be a valid hid filename
    as used in R&D set. We check it starts with something like 2A1
    """
    return len(re.findall(r'\d[ABCDEF]\d', file_name[:3])) > 0


def get_prefix_from_filename(file_name: PathLike) -> str:
    if is_rd_hid_filename(file_name):
        return file_name.split("_")[0]  # take '1A2'
    else:
        raise ValueError(f"Cannot take prefix from provided file name: {file_name}")


def is_non_case_sample_hid_file_name(file_name: str) -> bool:
    """
    Whether we recognise this file name as a control, ladder or other non-
    forensically relevant sample
    """
    return ('blanco' in file_name.lower()
            or 'ladder' in file_name.lower()
            or 'pocon' in file_name.lower()
            or 'controle' in file_name.lower()
            or file_name.startswith('A'))


def get_noc_from_rd_file_name(file_name: PathLike) -> Optional[str]:
    """
    From a hid rd file name like '1A2_A01_01.hid', retrieve the number of
    contributors (as "<noc>p"), that is indicated by the `2` of `1A2`.
    """
    if is_rd_hid_filename(file_name):
        return f"{str(file_name)[2]}p"
    else:
        return None


def marker_list_to_dict(marker_list: Sequence[Marker], as_json: bool = False) -> \
        Union[List[Dict], str]:
    """
    Load the list of Markers as a dictionary. If `as_json` is True, we serialize the dictionary
    with JSON.
    """
    marker_dict = [dataclasses.asdict(m) for m in marker_list]
    return json.dumps(marker_dict) if as_json else marker_dict


def dict_to_marker_list(marker_dict: Union[List[Dict], str], as_json: bool = False) -> \
        Sequence[Marker]:
    """
    Load the marker/alleles dictionary into a list of Markers with Alleles. It is also possible
    that this dict was serialized (by `marker_list_to_dict`) and is in fact a string (indicated with
    `as_json` is True). In that case, we first unserialize this string into a dictionary and then
    load it into Marker and Allele objects.
    """
    marker_dict = json.loads(marker_dict) if as_json else marker_dict
    markers_list = []
    for marker in marker_dict:
        marker['alleles'] = [Allele(**a) for a in marker['alleles']]
        markers_list.append(Marker(**marker))
    return markers_list


def load_donor_alleles(file_name: str, panel: Panel) -> List[Marker]:
    """
    For R&D files, we know the donors that contributed and the DNA profiles of the donors. For a
    single .hid file, find the donors (from the file name) and return the list of Markers of those
    donors combined.
    :param file_name: .hid file to load actual donors for
    :param rd_data_root: root folder of the RD data, containing a Referenties folder
    :param panel: the panel to retrieve the dye row of the markers from
    """
    reference_path = "resources/data/2p_5p_Dataset_NFI/Referenties"
    if not is_rd_hid_filename(file_name):
        raise ValueError("Cannot load donor alleles for non-RD sample. "
                         f"Found file name {file_name}")

    mixture_type = get_prefix_from_filename(file_name)  # to retrieve e.g. '1A2'
    dataset_nr, nr_donors = mixture_type[0], int(mixture_type[2])
    # one file contains alleles of one donor, so find files for all donors of the profile
    file_stems = [f"{dataset_nr}{letter}" for letter in
                  DONORS_PER_DATASET_NR[dataset_nr][:nr_donors]]

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


def chunks(
        iterable: Iterable,
        chunk_size: int,
        skip_remainder: bool = False
) -> Iterator[List]:
    """
    Splits an iterable into chunks. Each element in the returned iterator is
    a collection of `chunk_size` elements of the original iterable if
    `skip_remainder` is True. If `skip_remainder` is False, the last chunk may
    be smaller than `chunk_size`.

    Examples:

    >>> a = list(range(10))
    >>> list(chunks(a, chunk_size=2))
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    >>> a = list(range(11))
    >>> list(chunks(a, chunk_size=2))
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10]]

    >>> a = list(range(11))
    >>> list(chunks(a, chunk_size=2, skip_remainder=True))
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    :param iterable: Iterable
    :param chunk_size: int
    :param skip_remainder: bool, whether to skip last chunk if it is smaller
    :return: Iterator[List]
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk or skip_remainder and len(chunk) < chunk_size:
            return
        yield chunk
