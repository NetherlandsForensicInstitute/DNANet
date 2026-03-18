import dataclasses
import json
import math
import re
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union, Tuple

import coolname
import numpy as np

from DNAnet.data.data_models import Allele, Marker
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.typing import PathLike


def is_rd_hid_filename(file_name: str) -> bool:
    """
    Perform simple checks on whether this may be a valid hid filename
    as used in R&D set. We check it starts with something like 2A1
    """
    return len(re.findall(r'\d[ABCDEF]\d', file_name[:3])) > 0


def is_no_control(file_name: str) -> bool:
    """
    Controls and ladders start with an 'A'
    """
    return not file_name.startswith('A')

def get_prefix_from_filename(file_name: PathLike) -> str:
    if is_rd_hid_filename(file_name):
        return file_name.split("_")[0]  # take '1A2'
    else:
        raise ValueError(f"Cannot take prefix from provided file name: {file_name}")

def generate_random_name() -> str:
    return ''.join([x.capitalize() for x in coolname.generate()])


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


def get_marker_ranges(
    image: HIDImage, tail_size: int = 5
) -> Dict[str, Tuple[int, np.ndarray]]:
    """Using the image's panel, get the ranges of the markers in the image.
    This function retrieves the ranges of the markers in the image,
    which are used for zooming in on specific markers.

    Args:
        image (HIDImage): The HIDImage object containing the image data and metadata.
        tail_size (int, optional):
            Extra space that's added on the left and right side of a marker's bin.
            Defaults to 5.

    Returns:
        Dict[str, Tuple[int, np.ndarray]]: A dictionary containing the marker names as keys,
        and a tuple of the dye row and the bin range as values.
    """
    marker_ranges = {}
    for marker in image._panel._panel:
        marker_name = marker.name
        marker_bin = _get_marker_bin(marker)
        scanpoint_bin = tuple(np.argmin(np.abs(image._scaler - marker_bin), axis=1))
        scanpoint_bin = (
            max(0, scanpoint_bin[0] - tail_size),
            min(4096, scanpoint_bin[1] + tail_size),
        )
        marker_ranges[marker_name] = (marker.dye_row, np.arange(*scanpoint_bin))
    return marker_ranges


def _get_marker_bin(marker):
    left_bin, right_bin = math.inf, -math.inf
    for allele in marker.alleles:
        left_bin = min(left_bin, allele.base_pair - allele.left_bin)
        right_bin = max(right_bin, allele.base_pair + allele.right_bin)

    bins = np.array([left_bin, right_bin])[:, np.newaxis]
    marker_bin = bins + np.array([-1, 1])[:, np.newaxis]

    return marker_bin


def get_allele_bins(
    image: HIDImage
) -> Iterable[Tuple[int, Tuple[int, int]]]:
    """Using the image's panel, get the bins for all the alleles in the image.

    Args:
        image (HIDImage): The HIDImage object containing the image data and metadata.


    Returns:
        Dict[str, Tuple[int, np.ndarray]]: A dictionary containing the marker names as keys,
        and a tuple of the dye row and the bin range as values.
    """
    allele_bins = []
    for marker in image._panel._panel:
        for allele in marker.alleles:
            allele_bin = np.array([allele.base_pair - allele.left_bin,
                                   allele.base_pair + allele.right_bin])[:, np.newaxis]
            scanpoint_bin = tuple(np.argmin(np.abs(image._scaler - allele_bin), axis=1))
            scanpoint_bin = (
                max(0, scanpoint_bin[0]),
                min(4096, scanpoint_bin[1]),
            )
            allele_bins.append((marker.dye_row, scanpoint_bin))
    return allele_bins

def _get_allele_bins(marker):
    """
    returns the set of bins for alle the marker's alleles in the panel
    """
    left_bin, right_bin = math.inf, -math.inf
    for allele in marker.alleles:
        left_bin = allele.base_pair - allele.left_bin
        right_bin = allele.base_pair + allele.right_bin

    bins = np.array([left_bin, right_bin])[:, np.newaxis]
    marker_bin = bins + np.array([-1, 1])[:, np.newaxis]

    return marker_bin


# get all the HID files from the hid_files_path directory
def find_files_by_suffix(root_dir, suffix) -> List[Path]:
    """
    Recursively find all files in root_dir that end with the given suffix.
    """
    return list(Path(root_dir).rglob(f'*{suffix}'))
