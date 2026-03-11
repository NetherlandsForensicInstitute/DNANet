from typing import Optional, Tuple, Union, Callable, List

import numpy as np
import scipy


CATEGORIES = ["", "Allele", "Stutter", "PullUp", "BleedThrough", "Spike", "DyeBlob", "Artefact",
              "Unclear", "Shoulder", "ForeignDNA", "OverloadingArtifact"]

# TODO remove this, and use the kit specific settings
DNA_CHANNELS = ('blue', 'green', 'black', 'red', 'purple', 'orange')


# deprecated constants, kept for backward compatibility
# These are in fact NOT deprecated and still used in parts of the code
# TODO: Check how these CAN be deprecated based on the KIT strategy
SIZE_STANDARD_BPS: np.ndarray = np.array([65, 80, 100, 120, 140, 160, 180,
                                          200, 225, 250, 275, 300, 325,
                                          350, 375, 400, 425, 450, 475])
BASE_PAIR_START, BASE_PAIR_END = 65, 475
RESCALE_SIZE = 4096


def basepair_to_pixel(scaler: np.ndarray, bp: float) -> float:
    """
    Translate a base pair location to a pixel location using a scaler.
    """
    return float(np.argmin(np.abs(scaler - bp), axis=1))


def assert_image_data_valid_format(data: np.ndarray,
                                   n_color_channels: int = 3):
    """
    Makes sure that the raw image `data` conforms to the correct  format.
    """
    if len(data.shape) != 3:
        raise ValueError(
            f"The image must be 3D `(height, width, num_channels)`, not "
            f"{len(data.shape)}D. Full shape: {data.shape}"
        )
    if data.shape[-1] != n_color_channels:
        raise ValueError(
            f"Image must have {n_color_channels} color channels, not "
            f"{data.shape[-1]}. Full shape: {data.shape}"
        )
    if data.dtype != np.uint8:
        raise ValueError(
            f"dtype of `data` must be `np.uint8` to ensure consistency, "
            f"not {data.dtype}"
        )


def process_image(
        data: np.ndarray,
        channels_first: bool = False
) -> np.ndarray:
    """
    Processes the image data so that it can be fed directly to a model.

    :param data: the raw image data
    :param channels_first: whether to permute the resulting array such that
         the image channels are represented by the first axis rather than the last. The order of
         the dimensions becomes (num_channels, height, width)
    :return: A numpy array with the processed image data.
    """
    # TODO: here we can store any other preprocessing like augmentation or normalization

    # Swap the order of the axes if desired.
    if channels_first:
        return np.transpose(data, (2, 0, 1))

    # Otherwise, return the array as is.
    return data


def validate_ss_peaks(size_standard_peaks_idxs: np.ndarray) -> bool:
    """
    Validate whether the identified peaks in the size standard are likely to
    correspond with the SIZE_STANDARD_BPS array, by checking whether 19 peaks
    have been identified and the relative distances between base pairs is
    between certain values. Returns True if all checks are passed, False otherwise.
    """
    if len(size_standard_peaks_idxs) != 19:
        # The number of selected peaks should be 19
        return False
    distances_between_peaks = np.absolute(
        np.diff((size_standard_peaks_idxs,
                 scipy.ndimage.shift(size_standard_peaks_idxs, shift=1, mode='nearest')
                 ), axis=0))[0, 1:]
    distance_basepairs = np.absolute(
        np.diff((SIZE_STANDARD_BPS,
                 scipy.ndimage.shift(SIZE_STANDARD_BPS, shift=1, mode='nearest')
                 ), axis=0))[0, 1:]
    relative_distances = distances_between_peaks / distance_basepairs
    if not np.all((relative_distances <= 13) & (relative_distances >= 7)):
        # The pixels per basepair are expected to be between 7 and 13
        return False
    return True


def get_interpolated_basepairs(size_standard: np.ndarray) -> (
        Optional)[np.ndarray]:
    """
    Takes the array of the size standard and detects the 19 peaks corresponding
    with the provided list of base pairs. Put these base pairs on their position
    in an array and interpolates all values in between.

    :param size_standard: size standard array
    :return: interpolated base pairs
    """
    # find the peaks in the size standard array
    size_standard_peaks_idxs = extract_ss_peaks(size_standard)
    # only take the last 19 peaks excluding the final peak, validate by comparing their
    # relative distances to SIZE_STANDARD_BPS
    size_standard_peaks_idxs = size_standard_peaks_idxs[-20:-1]
    if not validate_ss_peaks(size_standard_peaks_idxs):
        return None
    # put the peak indices on the base pair values
    interp = basepair_interpolator(indices=size_standard_peaks_idxs,
                                   original_x_values=SIZE_STANDARD_BPS)
    basepairs_interpolated = interp(np.arange(len(size_standard)))
    # interpolate these to the complete array (all values for indices outside
    # the size_standard_peaks_idxs range will be 0)
    return basepairs_interpolated


def find_peaks_above_threshold(array: np.ndarray, threshold: int) -> \
        np.ndarray:
    """
    Find indices of peaks of an array above some threshold. This also includes
    looking for the beginning or end of flat peaks.
    """
    return (np.where((((array >= scipy.ndimage.shift(array, 1)) &
                       (array > scipy.ndimage.shift(array, -1))) |
                      ((array > scipy.ndimage.shift(array, 1)) &
                       (array >= scipy.ndimage.shift(array, -1))))
                     & (array >= threshold)))[0]


def find_peak_boundary(array: np.ndarray, idx: int, threshold: int) \
        -> Tuple[int, int]:
    """
    Find the start and end of a peak, whose peak top is located at `idx`. First
    split the array on `idx`. In the left split, look for the last index where
    the array has a value below `threshold`. In the right part, look similarly
    for the first index, in order to find the closest indices to `idx`.
    If the left split has only values above threshold, we return the beginning
    of the array. If the right split has only values above threshold, we return
    the end of the array. (TODO: improve this by fixed width/higher threshold?)
    """
    # TODO: sometimes the baseline of the array is higher than the threshold,
    # therefore we might want to adjust the threshold manually to some higher
    # value
    # TODO: check whether array[idx] is indeed a peak?
    array = array.flatten()
    # split the array on the peak index
    left_part, right_part = array[:idx], array[idx:]
    # on the left side of the peak, look for the closest index below the threshold
    start = np.where(left_part < threshold)[0][-1] + 1 if \
        any(left_part < threshold) else 0
    # on the right side of the peak, look for the closest index below the threshold
    end = np.where(right_part < threshold)[0][0] + idx - 1 if \
        any(right_part < threshold) else len(array) - 1
    return start, end


def find_peak_near_idx(array: np.ndarray, idx: int) -> np.ndarray:
    """
    Find the index of a peak in the `array` that is closest to the provided
    `idx` and has a peak height above the peak at position `idx`. Returns the
    index of the peak in the provided `array`. When two peaks have equal
    distance, the first peak index is returned.
    # TODO: it may occur that two peaks merge into each other, in that case
    # you ideally want the higher one, now we take the peak closest to `idx`.
    """
    peaks_idxs = find_peaks_above_threshold(array, array[idx])
    return peaks_idxs[np.abs(peaks_idxs - idx).argmin(), np.newaxis]


def find_peak_idx_near_or_in_range(array: np.ndarray, index_range: np.ndarray,
                                   threshold: int) -> np.ndarray:
    """
    Find a (single) peak index in `array` within the `index_range`, or just
    outside (before or after) the `index_range`. It may also be
    possible that no peak is found above the `threshold`. In that case, an
    empty array is returned.
    """
    values_in_range = array[index_range].flatten()
    if np.all(np.diff(values_in_range) > 0):
        # only increasing, search for peak near end of range
        peak_idx = find_peak_near_idx(array.flatten(), index_range[-1])
    elif np.all(np.diff(values_in_range) < 0):
        # only decreasing, search for peak near beginning of range
        peak_idx = find_peak_near_idx(array.flatten(), index_range[0])
    else:  # there must exist any (>=1) peak within the range
        peak_idx = find_peaks_above_threshold(values_in_range,
                                              threshold) + index_range[0]
        if peak_idx.size > 1:
            # multiple peaks found, return the highest
            peak_heights = array[peak_idx].flatten()
            peak_idx = peak_idx[np.argmax(peak_heights), np.newaxis]
    # return only peak index if the peak is above threshold
    return peak_idx if peak_idx.size > 0 and array[peak_idx] >= threshold else np.array([])


def extract_ss_peaks(array: np.ndarray) -> np.ndarray:
    """
    Takes an array and extracts the indices of the size standard peaks, by comparing each
    value with the neighbours and a threshold. We may find 'flat' peaks (e.g.
    [500, 520, 520, 510]) or a peak within a close distance of another
    peak, therefore we filter the found indices based on distance.
    """
    peak_idxs = find_peaks_above_threshold(array, 180)
    # the final two peaks in the size standard are often lower than the other peaks, therefore we
    # try to find those in the end of the array with a lower threshold if we haven't found them yet
    split_idx = 8200  # TODO: can we find this dynamically or something?
    if len(peak_idxs) > 0 and peak_idxs[-1] <= split_idx:
        final_peak_idxs = find_peaks_above_threshold(array[split_idx:], 120) + split_idx
        peak_idxs = np.union1d(peak_idxs, final_peak_idxs)
    # look for peak that are close (within 15 pixels) and delete the peak on the first index. This
    # may go wrong when we have a situation like [1000, 1001, 800, 800, 799], then we
    # ideally want to keep the highest peak (1001), but now this one gets deleted and we keep 1000.
    close_idxs = np.where(np.diff(peak_idxs) <= 15)[0]
    return np.delete(peak_idxs, close_idxs)


def extract_ss_peaks_simple(array: np.ndarray) -> np.ndarray:
    """
    Takes an array and extracts the indices of the size standard peaks, by comparing each
    value with the neighbours and a threshold. We may find 'flat' peaks (e.g.
    [500, 520, 520, 510]) or a peak within a close distance of another
    peak, therefore we filter the found indices based on distance.
    """
    peak_idxs = find_peaks_above_threshold(array, 300)
    # the final two peaks in the size standard are often lower than the other peaks, therefore we
    # try to find those in the end of the array with a lower threshold if we haven't found them yet
    # split_idx = 8200  # TODO: can we find this dynamically or something?
    # if len(peak_idxs) > 0 and peak_idxs[-1] <= split_idx:
    #     final_peak_idxs = find_peaks_above_threshold(array[split_idx:], 120) + split_idx
    #     peak_idxs = np.union1d(peak_idxs, final_peak_idxs)
    # look for peak that are close (within 15 pixels) and delete the peak on the first index. This
    # may go wrong when we have a situation like [1000, 1001, 800, 800, 799], then we
    # ideally want to keep the highest peak (1001), but now this one gets deleted and we keep 1000.
    close_idxs = np.where(np.diff(peak_idxs) <= 15)[0]
    return np.delete(peak_idxs, close_idxs)





def basepair_interpolator(indices: Union[np.ndarray, list[float]],
                          original_x_values: Union[np.ndarray, list[float]],
                          extrapolate: bool = False) \
        -> Callable[[Union[np.ndarray, float]], np.ndarray]:
    """Generate a cubic-spline interpolator for translating pixel indices.

       GeneMarker's sizing uses a cubic spline that passes exactly through every
       size-standard point. To mimic this behaviour we construct a natural cubic
       spline between the detected size-standard peak indices (``indices``) and
       the expected base-pair values (``original_x_values``). Outside of the
       calibrated range we zero-out the values unless explicit extrapolation is
       requested, reproducing the previous behaviour of returning zeros beyond the
       last standard fragment.

       :param indices: indices for which a value is present
       :param original_x_values: known x_values between which to interpolate
       :param extrapolate: whether to use extrapolation or not
       :return: callable that evaluates the spline at the requested positions
       """
    indices = np.asarray(indices)
    original_x_values = np.asarray(original_x_values)

    if indices.ndim != 1 or original_x_values.ndim != 1:
        raise ValueError("`indices` and `original_x_values` must be one-dimensional arrays")

    if indices.size != original_x_values.size:
        raise ValueError("`indices` and `original_x_values` must contain the same number of points")

    sort_order = np.argsort(indices)
    sorted_indices = indices[sort_order]
    sorted_basepairs = original_x_values[sort_order]

    spline = scipy.interpolate.CubicSpline(
        sorted_indices,
        sorted_basepairs,
        bc_type="natural",
        extrapolate=True,
    )

    lower_bound = sorted_indices[0]
    upper_bound = sorted_indices[-1]

    def _interpolate(x_new: Union[np.ndarray, float]) -> np.ndarray:
        x_new_arr = np.asarray(x_new, dtype=float)
        values = np.asarray(spline(x_new_arr))

        if not extrapolate:
            outside_mask = (x_new_arr < lower_bound) | (x_new_arr > upper_bound)
            values = np.where(outside_mask, 0.0, values)

        return np.atleast_1d(values)

    return _interpolate



def rescale_dye_old(basepairs: np.ndarray) -> np.ndarray:
    """
    Rescale the interpolated base pairs of the size standard so that they fit between
    BASE_PAIR_START and BASE_PAIR_END, on exactly RESCALE_SIZE pixels. The output array
    should function as a translator that indicates which pixel indices (of an unscaled dye) should
    be on every pixel location.
    E.g. if the output is np.array([3825, 3826, ..]), then a pixel on index 3825 should be
    scaled to the first pixel, and a pixel on index 3826 should be scaled to the second pixel.

    TODO we essentially either skip rfu values or take them twice, to make sure we end up with
    the right number of values. better would be to interpolate the actual signal.
    """
    target_linspace = np.linspace(BASE_PAIR_START, BASE_PAIR_END, RESCALE_SIZE)

    # Presorting interpolated base pairs
    sort_indices = np.argsort(basepairs)
    sorted_basepairs = basepairs[sort_indices]

    # Find insertion indices
    insertion_indices = np.searchsorted(
        sorted_basepairs,
        target_linspace,
        side='left'
    )

    # Adjust indices for boundary conditions
    insertion_indices = np.clip(
        insertion_indices,
        1,
        len(sorted_basepairs) - 1
    )

    # Determine the closest index prior or after based on value proximity
    left_indices = insertion_indices - 1
    right_indices = insertion_indices

    left_deltas = np.abs(sorted_basepairs[left_indices] - target_linspace)
    right_deltas = np.abs(sorted_basepairs[right_indices] - target_linspace)

    return np.where(
        (left_deltas < right_deltas) | (left_deltas == right_deltas),
        sort_indices[left_indices],
        sort_indices[right_indices],
    )

def fill_lut_range(lut: List[Optional[str]], offset: int, s_k: int, e_k: int,
                    name: str, ranges: List[Tuple[float, float, str]], marker_bp_scale: int) -> None:
    """
    Fill a range in the LUT, handling overlaps by splitting evenly at midpoint.
    """
    for k in range(s_k, e_k + 1):
        idx = k + offset
        if lut[idx] is not None and lut[idx] != name:
            # Overlap detected - find midpoint and split
            existing_name = lut[idx]
            existing_range = next((r for r in ranges if r[2] == existing_name), None)
            if existing_range:
                existing_start, existing_end, _ = existing_range
                current_range = next((r for r in ranges if r[2] == name), None)
                if current_range:
                    start, end, _ = current_range
                    # Calculate overlap region
                    overlap_start = max(start, existing_start)
                    overlap_end = min(end, existing_end)
                    midpoint = (overlap_start + overlap_end) / 2
                    mid_k = int(round(midpoint * marker_bp_scale))

                    # Assign based on which side of midpoint
                    if k < mid_k:
                        if start < existing_start:
                            lut[idx] = name
                    else:
                        if start >= existing_start:
                            lut[idx] = name
        else:
            lut[idx] = name


def rescale_dye(
    basepairs: np.ndarray,
    rescale_size: int,
    target_range: Tuple[int, int]
) -> np.ndarray:
    """Map base-pair positions to scaled pixel indices.

    ``basepairs`` is an array containing the interpolated base-pair value for
    each scan point of the electropherogram. ``target_range`` defines the base
    pair range of the rescaled profile (defaults to the size standard range when
    ``None``).
    """
    bp_start, bp_end = target_range
    target_linspace = np.linspace(bp_start, bp_end, rescale_size)

    # Presorting interpolated base pairs
    sort_indices = np.argsort(basepairs)
    sorted_basepairs = basepairs[sort_indices]

    # Find insertion indices
    insertion_indices = np.searchsorted(
        sorted_basepairs,
        target_linspace,
        side='left'
    )

    # Adjust indices for boundary conditions
    insertion_indices = np.clip(
        insertion_indices,
        1,
        len(sorted_basepairs) - 1
    )

    # Determine the closest index prior or after based on value proximity
    left_indices = insertion_indices - 1
    right_indices = insertion_indices

    left_deltas = np.abs(sorted_basepairs[left_indices] - target_linspace)
    right_deltas = np.abs(sorted_basepairs[right_indices] - target_linspace)

    return np.where(
        (left_deltas < right_deltas) | (left_deltas == right_deltas),
        sort_indices[left_indices],
        sort_indices[right_indices],
    )


def full_annotation_array_to_list(full_annotations_array: np.ndarray,
                                  rfu_array: np.ndarray=None) -> list[list]:
    """
    Converts a numpy array of annotations to a list. Optionally provide the data rfu array
    to get the max rfu for every annotation
    """
    results = []

    # Do not include annotations from size standard
    for row_idx in range(5):
        row = full_annotations_array[row_idx]

        # Find where labels change
        changes = np.concatenate(([True], row[1:] != row[:-1], [True]))
        change_indices = np.where(changes)[0]

        # Extract segments
        for i in range(len(change_indices) - 1):
            start = change_indices[i]
            end = change_indices[i + 1] - 1
            label = row[start]
            if label == 0:
                continue
            else:
                max_rfu=-1
                if not rfu_array is None:
                    max_rfu = int(max(rfu_array[int(row_idx), start:end+1]))
                results.append(
                    [int(row_idx), start, end, int(label), max_rfu])

    return results
