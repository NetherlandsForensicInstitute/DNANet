"""Peak detection and boundary finding for electropherogram signals.

These utilities find fluorescence peaks in EPG dye channels and determine
their boundaries — essential for both allele calling and annotation adjustment.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage


def find_peaks_above_threshold(signal: np.ndarray, threshold: float) -> np.ndarray:
    """Find indices of peaks (including flat-top peaks) above a threshold.

    A peak is defined as a point that is >= its left neighbor AND > its right
    neighbor, or > its left neighbor AND >= its right neighbor. This handles
    plateau peaks (e.g. ``[500, 520, 520, 510]``).

    Args:
        signal: 1D signal array.
        threshold: Minimum peak height.

    Returns:
        Array of peak indices.
    """
    shifted_left = scipy.ndimage.shift(signal, 1)
    shifted_right = scipy.ndimage.shift(signal, -1)

    is_peak = (
        ((signal >= shifted_left) & (signal > shifted_right))
        | ((signal > shifted_left) & (signal >= shifted_right))
    ) & (signal >= threshold)

    return np.where(is_peak)[0]


def find_peak_boundary(signal: np.ndarray, peak_idx: int, threshold: float) -> tuple[int, int]:
    """Find the start and end indices of a peak.

    Starting from ``peak_idx``, walks left to find where the signal drops
    below ``threshold``, then walks right similarly.

    Args:
        signal: 1D signal array.
        peak_idx: Index of the peak top.
        threshold: Height below which the peak boundary is defined.

    Returns:
        Tuple of ``(start_index, end_index)`` inclusive.
    """
    signal = signal.flatten()
    left, right = signal[:peak_idx], signal[peak_idx:]

    start = np.where(left < threshold)[0][-1] + 1 if np.any(left < threshold) else 0
    end = (
        np.where(right < threshold)[0][0] + peak_idx - 1
        if np.any(right < threshold)
        else len(signal) - 1
    )
    return start, end


def find_peak_near_idx(signal: np.ndarray, idx: int) -> np.ndarray:
    """Find the nearest peak that is at least as high as ``signal[idx]``.

    Args:
        signal: 1D signal array.
        idx: Reference index to search around.

    Returns:
        Array containing the single closest peak index.
    """
    peak_idxs = find_peaks_above_threshold(signal, signal[idx])
    return peak_idxs[np.abs(peak_idxs - idx).argmin(), np.newaxis]


def find_valley_idx_in_range(
    signal: np.ndarray, index_range: np.ndarray, threshold: float
) -> np.ndarray:
    """Find the index of the signal minimum within *index_range*.

    Used for classes where the region of interest is identified by a local
    minimum rather than a local maximum (e.g. bleed-through troughs).
    The *threshold* parameter is accepted for a uniform call signature but is
    not used — valleys are defined by position, not height.

    Args:
        signal: 1D signal array.
        index_range: Scanpoint indices of the annotated region.
        threshold: Unused; kept for call-signature parity with
            :func:`find_peak_idx_near_or_in_range`.

    Returns:
        Single-element array with the valley index, or an empty array if
        *index_range* is empty.
    """
    if index_range.size == 0:
        return np.array([])
    values = signal[index_range].flatten()
    valley_local = int(np.argmin(values))
    return np.array([index_range[valley_local]])

def find_absolute_peak_idx_in_range(
    signal: np.ndarray, index_range: np.ndarray, treshold: float
) -> np.ndarray:
    """Find the index of the signal max over the absolute values within *index_range*.
    
    Used for classes where the signal can either be positive (peak) or negative (valley).
    The *treshold* parameter is accepted for a uniform call signature but is not used.

    Args:
        signal: 1D signal array
        index_range: Scanpoint indices of the annotated region.
        treshold: Unused; kept for call-signature parity with
            :func:`find_peak_idx_near_or_in_range`.

    Returns:
        Single-element array with the peak/valley index, or an empty array if 
        *index_range* is empty.
    """
    if index_range.size == 0:
        return np.array([])
    
    values = signal[index_range].flatten()
    absolute_values = np.abs(values)
    absolute_peak = int(np.argmin(absolute_values))
    return np.array([index_range[absolute_peak]])


def find_peak_idx_near_or_in_range(
    signal: np.ndarray, index_range: np.ndarray, threshold: float
) -> np.ndarray:
    """Find the dominant peak within or near a given index range.

    Handles three cases:
    - Monotonically increasing in range → peak is just after the range end.
    - Monotonically decreasing → peak is just before the range start.
    - Mixed → one or more peaks exist within the range; returns the tallest.

    Returns:
        Array with the single peak index, or empty array if none above threshold.
    """
    values = signal[index_range].flatten()

    if np.all(np.diff(values) > 0):
        peak_idx = find_peak_near_idx(signal.flatten(), index_range[-1])
    elif np.all(np.diff(values) < 0):
        peak_idx = find_peak_near_idx(signal.flatten(), index_range[0])
    else:
        peak_idx = find_peaks_above_threshold(values, threshold) + index_range[0]
        if peak_idx.size > 1:
            heights = signal[peak_idx].flatten()
            peak_idx = peak_idx[np.argmax(heights), np.newaxis]

    if peak_idx.size > 0 and signal[peak_idx] >= threshold:
        return peak_idx
    return np.array([])
