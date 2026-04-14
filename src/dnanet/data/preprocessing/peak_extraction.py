"""Peak extraction from EPG profiles.

Provides functions to extract fixed-width signal windows centered on
detected peaks. These windows are the input to the peak classifier and
the local branch of PeakNet.

Two implementations are provided:

- :func:`extract_peak_windows` — NumPy-based, returns
  :class:`~dnanet.data.extracted_peak.ExtractedPeak` objects.
- :func:`extract_peaks_torch` — PyTorch-based, returns raw tensors for
  efficient GPU batching in the combined PeakNet forward pass.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import scipy
import torch

from dnanet.data.strategies import StrategyRegistry
from dnanet.data.extracted_peak import ExtractedPeak


if TYPE_CHECKING:
    from dnanet.core.panel import Panel
    from dnanet.data.image import HIDImage


def setup_marker_to_idx() -> Tuple[Dict[str, int], int]:
    """Generate an index mapping from the kit's loci to an index number.

    Returns:
        A tuple with the index mapping and the number of markers.
    """
    scaling_strategy = StrategyRegistry.get_scaling_strategy()
    marker_to_dye_idx = scaling_strategy.marker_name_to_dye_idx()

    _marker_to_idx = {name: idx + 1 for idx, name in enumerate(marker_to_dye_idx.keys())}
    _marker_to_idx['Out of Bin'] = 0
    _n_markers = len(_marker_to_idx)
    return _marker_to_idx, _n_markers


# ---------------------------------------------------------------------------
# NumPy helpers
# ---------------------------------------------------------------------------


def _slice_with_padding(
    arr: np.ndarray,
    dye_index: int,
    start: int,
    length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Slice a single dye channel with zero-padding for out-of-bounds regions.

    Args:
        arr: Profile array of shape ``(D, L)`` or ``(D, L, 1)``.
        dye_index: Dye channel to slice.
        start: Start index (can be negative).
        length: Number of scan points to extract.
        pad_value: Value for out-of-bounds positions.

    Returns:
        1D array of shape ``(length,)``.
    """
    if arr.ndim == 3:
        signal = arr[dye_index, :, 0]
    else:
        signal = arr[dye_index]

    L = signal.shape[0]
    result = np.full(length, pad_value, dtype=signal.dtype)

    src_start = max(0, start)
    src_end = min(L, start + length)
    dst_start = max(0, -start)
    dst_end = dst_start + (src_end - src_start)

    if src_end > src_start:
        result[dst_start:dst_end] = signal[src_start:src_end]

    return result


def _build_peak_data(
    img_data: np.ndarray,
    dye_index: int,
    start: int,
    length: int,
    include_max_pool_dyes: bool,
    pad_value: int = 0,
) -> np.ndarray:
    """Creates a 2D array of shape (2, length) if include_max_pool_dyes is True,
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
        return _slice_with_padding(img_data, dye_index, start, length, pad_value)

    mask = np.arange(img_data.shape[0]) != dye_index
    other_channels = img_data[mask]
    max_pool = np.max(other_channels, axis=0, keepdims=True)

    data = np.empty((2, length), dtype=img_data.dtype)
    data[0] = _slice_with_padding(img_data, dye_index, start, length, pad_value)
    data[1] = _slice_with_padding(max_pool, 0, start, length, pad_value)

    return data


def _find_marker_for_peak(
    peak_bp: float, dye_index: int, panel: HIDImage | Panel, marker_to_idx: Dict[str, int]
) -> tuple[str | None, int]:
    """Find the marker a peak falls within.

    Args:
        peak_bp: Peak position in base pairs.
        dye_index: Dye channel index.
        panel: Panel object with marker definitions.

    Returns:
        ``(marker_name, marker_index)`` or ``(None, 0)`` if out of bin.
    """
    if panel is None:
        return None, 0

    try:
        markers = panel.adjusted_panel if hasattr(panel, 'adjusted_panel') else panel.markers
    except AttributeError:
        return None, 0

    for marker in markers:
        if getattr(marker, 'dye_row', None) != dye_index:
            continue

        # Check if peak falls within marker bin
        alleles = getattr(marker, 'alleles', [])
        if alleles:
            left = min(a.left_bin for a in alleles if hasattr(a, 'left_bin'))
            right = max(a.right_bin for a in alleles if hasattr(a, 'right_bin'))
            if left <= peak_bp <= right:
                name = getattr(marker, 'name', str(marker))
                return name, marker_to_idx.get(name, 0)

    return None, 0


def _label_peak_from_annotation(
    annotation_image: np.ndarray | None,
    dye_index: int,
    peak_center: int,
    padding: int = 1,
) -> str:
    """Determine peak label from the annotation mask based on the annotation classes specified in the dataset strategy.

    When no annotation is available, we use default class 0.
    When there are multiple annotation classes, we take the most common class in the slice.
    If there is a tie, we take the lowest class index.

    Args:
        annotation_image: Binary mask ``(D, L)`` or ``(D, L, 1)``, or None.
        dye_index: Dye channel index.
        peak_center: Scan-point index of peak apex.
        padding: Number of positions around the center to check.

    Returns:
        The annotation class name corresponding to the most common class in the slice.
    """
    annotation_classes = StrategyRegistry.get_dataset_strategy().get_annotation_classes()

    if annotation_image is None:
        return annotation_classes[
            0
        ]  # default to first class (e.g. "noise") if no annotation available

    if annotation_image.ndim == 3:
        ann = annotation_image[dye_index, :, 0]
    else:
        ann = annotation_image[dye_index]

    return _label_peak_from_annotation_fast(ann, peak_center, padding)


def _label_peak_from_annotation_fast(
    ann_channel: np.ndarray | None,
    peak_center: int,
    padding: int = 1,
) -> str:
    """Fast path that operates on a single channel slice.

    Args:
        ann_channel: 1D binary mask for single dye channel, or None.
        peak_center: Scan-point index of peak apex.
        padding: Number of positions around the center to check.

    Returns:
        ``"allele"`` if any annotated position is within padding of center,
        ``"noise"`` otherwise.
    """
    annotation_classes = StrategyRegistry.get_dataset_strategy().get_annotation_classes()
    if ann_channel is None:
        return annotation_classes[0]

    L = ann_channel.shape[0]
    start = max(0, peak_center - padding)
    end = min(L, peak_center + padding + 1)
    annotation_slice = ann_channel[start:end]
    if np.any(annotation_slice > 0):
        unique, counts = np.unique(annotation_slice, return_counts=True)

        # take the most common class in the slice
        # if there is a tie, take the lowest class index
        # in any case, "noise" should never be the answer here
        # so we filter the unique and counts below
        unique, counts = map(
            np.array, zip(*[(u, c) for u, c in zip(unique, counts, strict=True) if u != 0.0])
        )

        sorted_idx = np.lexsort((unique, -counts))
        most_common_value = int(unique[sorted_idx[0]])

        return annotation_classes[most_common_value]
    return annotation_classes[0]


# ---------------------------------------------------------------------------
# Main extraction functions
# ---------------------------------------------------------------------------


def extract_peak_windows(
    image: HIDImage, threshold: float, window_size: int, include_max_pool_dyes: bool = False
) -> list[ExtractedPeak]:
    """Extract peak windows from a HIDImage using NumPy.

    For each dye channel (excluding the size standard if 6 channels), detects
    peaks above ``threshold`` and extracts a ``window_size``-wide signal window
    centered on each peak.

    Args:
        image: Source DNA profile.
        threshold: Minimum RFU for peak detection.
        window_size: Width of extraction window in scan points.
        include_max_pool_dyes: Add max-pooled other-dyes channel.

    Returns:
        List of :class:`ExtractedPeak` objects.
    """
    if window_size <= 0:
        raise ValueError('window_size must be a positive integer')

    scaling_strategy = StrategyRegistry.get_scaling_strategy()

    data = image.data
    if data is None:
        return []

    if data.ndim == 3:
        data_2d = data[:, :, 0]
    else:
        data_2d = data

    n_dyes = scaling_strategy.kit.num_dyes - 1  # exclude size standard
    assert n_dyes <= data_2d.shape[0], 'Image has fewer dye channels than expected'
    data = data_2d[:n_dyes, :]

    # Cache marker mapping once (fixes Issue 1)
    marker_to_idx, _ = setup_marker_to_idx()

    annotation_image = image.annotation.data if image.annotation is not None else None
    adjusted_panel = getattr(image, '_panel', None)

    # OPTIMIZATION 2: Pre-compute max-pooled signals per dye (fixes Issue 3)
    max_pool_data = None
    if include_max_pool_dyes:
        max_pool_data = np.empty((n_dyes, data.shape[1]), dtype=data.dtype)
        all_indices = np.arange(n_dyes)
        for dye_idx in range(n_dyes):
            mask = all_indices != dye_idx
            max_pool_data[dye_idx] = np.max(data[mask], axis=0)

    # Pre-pad all channels to eliminate per-peak allocations
    half_window = window_size // 2
    pad_left = half_window
    pad_right = window_size - half_window  # Handles both even/odd window sizes

    padded_data = np.pad(data, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    if include_max_pool_dyes:
        padded_max_pool = np.pad(
            array=max_pool_data,
            pad_width=((0, 0), (pad_left, pad_right)),
            mode='constant',
            constant_values=0,
        )

    peaks: list[ExtractedPeak] = []

    # Flattened loop with faster slice extraction
    for dye_index in range(n_dyes):
        dye_data = data[dye_index]

        # Find all peaks for this dye at once
        peak_indices, _ = scipy.signal.find_peaks(dye_data, height=threshold)

        if len(peak_indices) == 0:
            continue

        # Prepare annotation lookup for this dye (avoids repeated ndim checks)
        if annotation_image is not None:
            ann_channel = (
                annotation_image[dye_index, :, 0]
                if annotation_image.ndim == 3
                else annotation_image[dye_index]
            )
        else:
            ann_channel = None

        # Process all peaks for this dye channel
        for peak_scanpoint in peak_indices:
            peak_height = dye_data[peak_scanpoint]
            peak_basepair = image.scaler[peak_scanpoint]

            # Fast window extraction on pre-padded array
            # peak_scanpoint in original maps to peak_scanpoint + pad_left in padded
            start_padded = peak_scanpoint
            end_padded = start_padded + window_size

            if include_max_pool_dyes:
                peak_data = np.empty((2, window_size), dtype=data.dtype)
                peak_data[0] = padded_data[dye_index, start_padded:end_padded]
                peak_data[1] = padded_max_pool[dye_index, start_padded:end_padded]
            else:
                peak_data = padded_data[dye_index, start_padded:end_padded]

            # Use cached marker_to_idx
            peak_marker_name, peak_marker_index = _find_marker_for_peak(
                peak_basepair, dye_index, adjusted_panel, marker_to_idx
            )

            # Fast annotation check without function call overhead
            peak_label = _label_peak_from_annotation_fast(ann_channel, peak_scanpoint, padding=2)

            peaks.append(
                ExtractedPeak(
                    data=peak_data,
                    dye_index=dye_index,
                    peak_center=peak_scanpoint,
                    peak_basepair=peak_basepair,
                    window_size=window_size,
                    peak_height=peak_height,
                    label=peak_label,
                    marker_name=peak_marker_name,
                    marker_index=peak_marker_index,
                )
            )

    return peaks


def extract_peaks_torch(
    image: HIDImage,
    threshold: float,
    window_size: int,
    include_max_pool_dyes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract peak windows from a HIDImage as PyTorch tensors.

    More efficient than :func:`extract_peak_windows` for GPU-based
    training since it skips creating intermediate Python objects.

    Args:
        image: Source DNA profile.
        device: Target device for tensors.
        threshold: Minimum RFU for peak detection.
        window_size: Width of extraction window.
        include_max_pool_dyes: Add max-pooled other-dyes channel.

    Returns:
        Tuple of:
        - ``peak_windows``: ``(P, C, W)`` float32 tensor.
        - ``marker_idxs``: ``(P,)`` long tensor.
        - ``peak_centers``: ``(P, 2)`` long tensor — ``[dye_idx, position]``.
    """
    data = image.data

    scaling_strategy = StrategyRegistry.get_scaling_strategy()
    adjusted_panel = image.adjusted_panel

    if data is None:
        raise ValueError('Cannot extract peaks from an image with no data.')

    if data.ndim == 3:  # (D, L, 1)
        data_2d = data[:, :, 0]  # (D, L)
    else:
        data_2d = data  # (D, L)

    x = torch.from_numpy(data_2d.astype(np.float32))
    n_dyes = scaling_strategy.kit.num_dyes
    n_dyes = n_dyes - 1  # exclude size standard
    x = x[:n_dyes, :]

    peak_centers = _find_peaks_torch_indices(x, threshold)

    peak_windows = _extract_windows_torch(x, peak_centers, window_size, include_max_pool_dyes)

    marker_to_idx = scaling_strategy.marker_name_to_dye_idx()

    marker_idxs = [
        marker_to_idx.get(adjusted_panel.get_marker_name_by_dye_and_bp(dye, bp), len(marker_to_idx))
        for dye, bp in peak_centers.tolist()
    ]
    marker_idxs = torch.tensor(marker_idxs, dtype=torch.long)

    return peak_windows, marker_idxs, peak_centers


def _find_peaks_torch_indices(
    x: torch.Tensor,
    threshold: float,
    dim: int = -1,
) -> torch.Tensor:
    """Plateau-aware peak finder matching SciPy `signal.find_peaks(x, height=threshold)`-style local-max logic:
    - A peak is a local maximum.
    - Flat peaks (plateaus) count as one peak.
    - For a plateau peak, return the middle index (rounded down if even).
    - Excludes peaks touching the signal edges (i.e., runs starting at 0 or ending at n-1).

    Notes:
    - Assumes no NaNs (comparisons with NaNs make plateau/ordering ambiguous).
    """
    if x.numel() == 0:
        return torch.empty((0, x.ndim), dtype=torch.long, device=x.device)

    dim = dim % x.ndim
    if dim != x.ndim - 1:
        raise ValueError(
            'Plateau-aware peak detection currently supports `dim` as the last dimension only.'
        )

    x_moved = x.movedim(dim, -1)  # (..., n)
    n = x_moved.size(-1)
    if n < 3:
        return torch.empty((0, x.ndim), dtype=torch.long, device=x.device)

    outer_shape = x_moved.shape[:-1]
    s = x_moved.reshape(-1, n)  # (B, n)
    B = s.size(0)

    # 1) Assign a run id per position (run = maximal contiguous region of equal values).
    change = s[:, 1:] != s[:, :-1]  # (B, n-1)
    run_id = torch.cat(
        [torch.zeros((B, 1), device=x.device, dtype=torch.long), change.long().cumsum(dim=1)], dim=1
    )  # (B, n), values in [0, n-1]

    # 2) Make run ids unique across batch by offsetting with b*n.
    gid = (run_id + (torch.arange(B, device=x.device, dtype=torch.long).view(B, 1) * n)).reshape(
        -1
    )  # (B*n,)
    pos = (
        torch.arange(n, device=x.device, dtype=torch.long).view(1, n).expand(B, n).reshape(-1)
    )  # (B*n,)

    # 3) For each (batch, run), compute run start=min(pos) and run end=max(pos) using scatter_reduce.
    # scatter_reduce_ supports reductions like 'amin'/'amax' with include_self.
    start = torch.full((B * n,), n, device=x.device, dtype=torch.long)
    end = torch.full((B * n,), -1, device=x.device, dtype=torch.long)
    start.scatter_reduce_(0, gid, pos, reduce='amin', include_self=True)
    end.scatter_reduce_(0, gid, pos, reduce='amax', include_self=True)

    exists = start < n
    if not bool(exists.any()):
        return torch.empty((0, x.ndim), dtype=torch.long, device=x.device)

    slot = torch.nonzero(exists, as_tuple=False).flatten()  # (R,)
    b = slot // n
    rs = start[slot]
    re = end[slot]

    # 4) Peak condition for a run (rs..re):
    # - interior run: rs>0 and re<n-1 (exclude edges)
    # - height >= threshold
    # - height strictly greater than left neighbor and right neighbor
    interior = (rs > 0) & (re < (n - 1))
    if not bool(interior.any()):
        return torch.empty((0, x.ndim), dtype=torch.long, device=x.device)

    b_i = b[interior]
    rs_i = rs[interior]
    re_i = re[interior]

    height = s[b_i, rs_i]  # constant over the run
    left = s[b_i, rs_i - 1]
    right = s[b_i, re_i + 1]

    is_peak = (height >= threshold) & (height > left) & (height > right)
    if not bool(is_peak.any()):
        return torch.empty((0, x.ndim), dtype=torch.long, device=x.device)

    b_p = b_i[is_peak]
    center = (rs_i[is_peak] + re_i[is_peak]) // 2  # midpoint, floor on even length

    # 5) Convert flattened batch index back to outer indices
    if len(outer_shape) == 0:
        return center.view(-1, 1)

    tmp = b_p
    unraveled = []
    for size in reversed(outer_shape):
        unraveled.append(tmp % size)
        tmp = tmp // size
    unraveled = list(reversed(unraveled))

    return torch.stack([*unraveled, center], dim=1)


def _extract_windows_torch(
    x: torch.Tensor,
    indices: torch.Tensor,
    window_size: int,
    include_maxpool_dyes: bool,
) -> torch.Tensor:
    """Extract windows from 2D tensor x centered at specified indices.
    Pads with zeros when out of bounds.

    x:        (D, L) tensor, e.g. (5 or 6, 4096)
    indices:  (p, 2) long/int tensor: [row_index, center_col]
    returns:  (p, C, window_size) windows, zero-padded out of bounds,
              where C=1 or 2 depending on include_maxpool_dyes.
    """
    if x.ndim != 2:
        raise ValueError(f'x must be 2D (C, L), got {tuple(x.shape)}')
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f'indices must be shape (p, 2), got {tuple(indices.shape)}')
    if window_size <= 0:
        raise ValueError('window_size must be > 0')

    C, L = x.shape
    idx = indices.to(device=x.device, dtype=torch.long)
    rows = idx[:, 0]  # (p,)
    centers = idx[:, 1]  # (p,)

    if torch.any(rows < 0) or torch.any(rows >= C):
        raise IndexError('Row indices out of range')

    half = window_size // 2
    starts = centers - half

    offsets = torch.arange(window_size, device=x.device, dtype=torch.long)  # (window_size,)
    pos = starts[:, None] + offsets[None, :]  # (p, window_size)

    mask = (pos >= 0) & (pos < L)
    pos_clamped = pos.clamp(0, L - 1)

    windows = x[rows[:, None], pos_clamped]  # (p, window_size)
    windows = windows * mask.to(dtype=torch.float32)

    if not include_maxpool_dyes:
        return windows.unsqueeze(1)  # (p, 1, window_size)

    # Max-pool over OTHER dyes for the same centered window positions.
    all_dye_windows = x[:, pos_clamped]  # (C, p, window_size)

    row_indices = torch.arange(C, device=x.device)[:, None]
    target_rows = rows[None, :]
    keep_mask = (row_indices != target_rows).unsqueeze(-1)  # (C, p, 1)

    neg_inf = torch.tensor(float('-inf'), device=x.device, dtype=x.dtype)
    masked_windows = torch.where(keep_mask, all_dye_windows, neg_inf)

    other_dyes_max, _ = masked_windows.max(dim=0)  # (p, window_size)
    other_dyes_max = other_dyes_max * mask.to(dtype=x.dtype)

    return torch.stack([windows, other_dyes_max], dim=1)  # (p, 2, window_size)
