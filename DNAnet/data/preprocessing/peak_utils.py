from typing import Sequence, Optional

import numpy as np
import torch

from DNAnet.data.data_models import Marker, Allele





def build_peak_data(img_data: np.ndarray, dye_index: int, start: int, length: int, include_max_pool_dyes: bool, pad_value: int = 0) -> np.ndarray:
    """
    Creates a 2D array of shape (2, length) if include_max_pool_dyes is True,
    otherwise a 1D array of shape (length,).
    The first row is the slice of the specified dye_index, and the second row
    is the max-pooled slice of all other dyes.
    If include_max_pool_dyes is False, only the slice of the specified dye_index
    is returned.
    The slice is taken from img_data[dye_index, start:start+length], with padding
    if the window falls outside the valid range.
    Args:
        img_data: The image data array of shape (num_dyes, num_scans).
        dye_index: The index of the dye to extract.
        start: The start index of the slice.
        length: The length of the slice.
        include_max_pool_dyes: Whether to include the max-pooled slice of other dyes.
        pad_value: The value to use for padding.

    Returns: np.ndarray: The extracted peak data array.

    """
    if not include_max_pool_dyes:
        return slice_with_padding(img_data, dye_index, start, length, pad_value)

    mask = np.arange(img_data.shape[0]) != dye_index
    other_channels = img_data[mask]
    max_pool = np.max(other_channels, axis=0, keepdims=True)

    data = np.empty((2, length), dtype=img_data.dtype)
    data[0] = slice_with_padding(img_data, dye_index, start, length, pad_value)
    data[1] = slice_with_padding(max_pool, 0, start, length, pad_value)

    return data


def slice_with_padding(arr: np.ndarray, dye_index: int, start: int, length: int, pad_value: int = 0) -> np.ndarray:
    """
    Returns a 1D slice arr[dye_index, start:start+length] with constant padding
    if the window falls outside the valid range.
    """
    N = arr.shape[1]  # assumes the scan/time axis is axis=1, as in arr[dye_index, :]
    s0 = max(0, start)
    s1 = min(N, start + length)
    left_pad  = s0 - start                 # >0 if start < 0
    right_pad = (start + length) - s1      # >0 if end > N

    # Extract valid portion and squeeze to 1D
    sl = arr[dye_index, s0:s1].squeeze()

    # Ensure pad value matches dtype (avoids upcasting for bool/int arrays)
    pad_val = arr.dtype.type(pad_value)

    if left_pad or right_pad:
        sl = np.pad(sl, (left_pad, right_pad), mode="constant", constant_values=pad_val)
    return sl


def markers_contain_allele(actual_alleles: Sequence[Marker], predicted_allele: Allele, predicted_marker: Marker) -> bool:
    """
    Check if the actual alleles contain the predicted allele in the predicted marker.

    Args:
        actual_alleles: Sequence[Marker]: The actual alleles from ground truth.
        predicted_allele: Allele: The predicted allele to check.
        predicted_marker: Marker: The marker of the predicted allele.

    Returns: bool: True if the actual alleles contain the predicted allele in the predicted marker, False otherwise.

    """
    # print(f"Checking if actual alleles contain predicted allele {predicted_allele} in marker {predicted_marker.name}")
    # print(f"Actual alleles: {[ (m.name, [a.name for a in m.alleles]) for m in actual_alleles if m.name == predicted_marker.name ]}")
    for marker in actual_alleles:
        if marker != predicted_marker:
            continue
        for allele in marker.alleles:
            if allele == predicted_allele:
                # print("Found matching allele.")
                return True
    # print("No matching allele found.")
    return False



def find_bin(marker: Marker, base_pair: float, max_offset: float = 0.1) -> Optional[Allele]:
    """
    Find the allele bin for a given marker and base pair value.
    Returns None if no allele is found.
    """
    #TODO Optimize

    for allele in marker.alleles:
        left_bound = allele.base_pair - allele.left_bin - max_offset
        right_bound = allele.base_pair + allele.right_bin + max_offset
        if left_bound <= base_pair <= right_bound:
            return allele
    return None

def get_peak_centers(peaks: Sequence['ExtractedPeak']) -> torch.Tensor:
    """
    Get the original peak center indices from a sequence of ExtractedPeak objects.

    Args:
        peaks: Sequence[ExtractedPeak]: The sequence of ExtractedPeak objects.
    Returns: torch.Tensor: A tensor of shape (N, 2) where N is the number of peaks,
                          containing the dye index and original peak center index for each peak.
    """
    vals = [(peak.dye_index, peak.original_peak_center_index) for peak in peaks]
    return torch.tensor(vals, dtype=torch.long)


def filter_peaks_AT_LT(peaks: Sequence['ExtractedPeak'],
                       analysis_threshold_type: str = "AT",
                       ) -> Sequence['ExtractedPeak']:
    """
    Filter peaks to only those above the AT and hetro imbalance thresholds.

    Args:
        analytical_threshold: list of analytical thresholds per dye
        hetro_imbalance_percentage: percentage of the highest peak to use as threshold
        peaks: peaks in a single profile

    Returns:
        Filtered peaks
    """

    # TODO should these values be different for different kits/datasets?
    thresholds_at = [
        95,  # blue
        140,  # green
        85,  # yellow
        135,  # red
        95,  # purple
    ]

    thresholds_lt = [
        45,  # blue
        50,  # green
        45,  # yellow
        80,  # red
        40,  # purple
    ]

    hetro_imbalance_percentage_at = 0.03
    hetro_imbalance_percentage_lt = 0.02


    if analysis_threshold_type == "NT": # No Threshold, Do not filter
        return peaks
    elif analysis_threshold_type == "DTH":
        hetro_imbalance_percentage = hetro_imbalance_percentage_at
        threshold = thresholds_at
    elif analysis_threshold_type == "DTL":
        hetro_imbalance_percentage = hetro_imbalance_percentage_lt
        threshold = thresholds_lt
    else:
        raise ValueError(f"Unknown analysis threshold type: {analysis_threshold_type}")

    max_height = max(peak.peak_height for peak in peaks)

    filtered_peaks = []
    for peak in peaks:
        # Check against analytical threshold
        try:
            if peak.peak_height < threshold[peak.dye_index]:
                continue
        except IndexError:
            raise f"Dye index {peak.dye_index} out of range for threshold list. Skipping peak."

        # Check against heterozygous imbalance threshold
        imbalance_threshold = hetro_imbalance_percentage * max_height
        if peak.peak_height < imbalance_threshold:
            continue

        filtered_peaks.append(peak)

    return filtered_peaks
