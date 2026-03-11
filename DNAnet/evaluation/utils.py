import dataclasses
import logging
from typing import Dict, Mapping, Optional, Sequence, Set, Union, List, Tuple

from scipy.signal import find_peaks

from DNAnet.data.data_models import Marker


LOGGER = logging.getLogger('dnanet')

THRESHOLDS_PER_LOCUS = {
    'AMEL': {'low': 45, 'high': 95},
    'D3S1358': {'low': 45, 'high': 95},
    'D1S1656': {'low': 45, 'high': 95},
    'D2S441': {'low': 45, 'high': 95},
    'D10S1248': {'low': 45, 'high': 95},
    'D13S317': {'low': 45, 'high': 95},
    'Penta E': {'low': 45, 'high': 95},
    'D16S539': {'low': 50, 'high': 140},
    'D18S51': {'low': 50, 'high': 140},
    'D2S1338': {'low': 50, 'high': 140},
    'CSF1PO': {'low': 50, 'high': 140},
    'Penta D': {'low': 50, 'high': 140},
    'TH01': {'low': 45, 'high': 85},
    'vWA': {'low': 45, 'high': 85},
    'D21S11': {'low': 45, 'high': 85},
    'D7S820': {'low': 45, 'high': 85},
    'D5S818': {'low': 45, 'high': 85},
    'TPOX': {'low': 45, 'high': 85},
    'D8S1179': {'low': 80, 'high': 135},
    'D12S391': {'low': 80, 'high': 135},
    'D19S433': {'low': 80, 'high': 135},
    'SE33': {'low': 80, 'high': 135},
    'D22S1045': {'low': 80, 'high': 135},
    'DYS391': {'low': 40, 'high': 95},
    'FGA': {'low': 40, 'high': 95},
    'DYS576': {'low': 40, 'high': 95},
    'DYS570': {'low': 40, 'high': 95},
    'Yindel': {'low': 40, 'high': 95},
}


def flatten_marker_list_to_locusallelename_list(
        markers: Sequence[Union[Marker, Dict]],
        locus: Optional[str] = None,
        min_rfu: Optional[int] = None,
        low_or_high: Optional[str] = None,
        remove_duplicates: bool = False
) -> Set[str]:
    """
    Takes a sequence of Marker objects, each having a sequence of Allele objects. Parses it to a
    dict representation if it is not already a dict.
    Flattens it to a set of strings and filters for locus name and minimal rfu, if provided
    For example:
    (
        Marker(D5, Alleles[13, 15.1]), Marker(D7, Alleles[14, 15.1])) ->
        set('D5_13','D5_15.1','D7_14','D7_15.1')
    )

    raises ValueError if duplicate alleles exist, unless remove_duplicates is True - then it removes them

    """
    min_rfus = get_min_rfu_per_locus(locus, low_or_high, min_rfu)

    if len(markers) == 0:
        return set()

    locus_alleles = []
    markers = map(dataclasses.asdict, markers) if isinstance(markers[0], Marker) else markers
    for marker in markers:
        for allele in marker['alleles']:
            if locus and marker['name'] != locus:
                continue
            # if min_rfu is provided, check if the allele is high enough.
            rfu_threshold = min_rfus[marker['name']]
            if rfu_threshold and allele['height'] is None:
                raise ValueError("Found rfu threshold, but no allele height to compare with. Set "
                                 "`min_rfu` and `low_or_high` to None to discard threshold.")
            if rfu_threshold:
                if isinstance(rfu_threshold, dict):
                    # we are interested in peaks that fall between the low and high threshold
                    if (
                        rfu_threshold["low"] >= allele["height"] or
                        allele["height"] >= rfu_threshold["high"]
                    ):
                        continue
                # we are interested in peaks above a threshold
                elif allele['height'] < rfu_threshold:
                    continue
            locus_alleles.append(f"{marker['name']}_{allele['name']}")
    locus_alleles_set = set(locus_alleles)
    if not remove_duplicates and len(locus_alleles_set) != len(locus_alleles):
        raise ValueError(f"Found non-unique locus-allele combinations for {markers}")
    return locus_alleles_set


def get_min_rfu_per_locus(
    locus: Optional[str],
    low_or_high: Optional[str],
    min_rfu: Optional[int],
) -> Mapping[str, Optional[int]]:
    """
    We will handle multiple options to set thresholds per locus or for a single locus (if `locus`
    is provided`):
    - we can set a min_rfu manually, by providing a value for the `min_rfu` argument. The value
    may also be 'None'. In that case we will not consider any threshold.
    - we can retrieve high or low locus specific thresholds by providing the `low_or_high` argument.
    - it is also possible to set `low_or_high` to `between`, so we retrieve both the low and high
    thresholds for a locus. In this case we are interested in peaks falling between the low
    and high detection threshold.
    If a locus_name is provided, we are thus only interested in that locus.
    """
    if locus and locus not in THRESHOLDS_PER_LOCUS:
        raise ValueError(f"Unknown locus {locus} found when trying to set locus-specific "
                         f"threshold.")
    if low_or_high and low_or_high not in ['low', 'high', 'between']:
        raise ValueError(f"Unexpected argument for `low_or_high`, found {low_or_high}, "
                         f"expected `low`, `high` or `between` (or None).")

    if low_or_high == "between":
        if locus:
            return {locus: THRESHOLDS_PER_LOCUS[locus]}
        return {loci: THRESHOLDS_PER_LOCUS[loci] for loci in THRESHOLDS_PER_LOCUS.keys()}

    if locus and low_or_high:  # set high/low locus specific threshold for this locus
        return {locus: THRESHOLDS_PER_LOCUS[locus][low_or_high]}
    elif locus and not low_or_high:  # set an arbitrary (or None) threshold for this locus
        return {locus: min_rfu}
    elif not locus and low_or_high:  # set high/low locus specific threshold for all loci
        return {locus: value[low_or_high] for locus, value in THRESHOLDS_PER_LOCUS.items()}
    else:  # set the same arbitrary (or None) threshold for all loci
        return {locus: min_rfu for locus in THRESHOLDS_PER_LOCUS.keys()}

def get_peaks(profile, min_rfu: float) -> List[List[Tuple[float, float]]]:
    """
    Crude peak boundary detection per dye by walking until descent stops.

    Parameters
    ----------
    profile : np.ndarray
        Profile array; this function looks at channel 0: profile[:, :, 0].
    min_rfu : float
        Minimum RFU height threshold for a point to count as a peak.

    Returns
    -------
    list of lists (start_idx, end_idx, peak_idx)
        For each dye i, a list of [start, end, peak_idx] index ranges (inclusive).
    """
    simple_peaks: List[List[Tuple[float, float]]] = []
    # Uses channel 0 as in your original code
    for i_dye, dye in enumerate(profile[:, :, 0]):
        peak_ids, _ = find_peaks(dye, height=min_rfu)
        simple_peaks.append([])
        for peak_idx in peak_ids:
            end = peak_idx
            while end + 1 < len(dye) and dye[end] > dye[end + 1]:
                end += 1
            start = peak_idx
            while start > 0 and dye[start - 1] < dye[start]:
                start -= 1
            simple_peaks[i_dye].append((start, end, peak_idx))
    return simple_peaks
