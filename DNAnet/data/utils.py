from typing import Optional, Tuple, Union, Callable

import numpy as np
import scipy


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

