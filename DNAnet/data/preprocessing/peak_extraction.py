from typing import Sequence, Dict

import numpy as np
import scipy
import torch

from DNAnet.data.data_models.extracted_peak import ExtractedPeak
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.strategies.strategy_registry import StrategyRegistry


def extract_windows_torch(
        x: torch.Tensor,
        indices: torch.Tensor,
        window_size: int,
        include_maxpool_dyes: bool,
) -> torch.Tensor:
    """
    Extract windows from 2D tensor x centered at specified indices.
    Pads with zeros when out of bounds.

    x:        (D, L) tensor, e.g. (5 or 6, 4096)
    indices:  (p, 2) long/int tensor: [row_index, center_col]
    returns:  (p, C, window_size) windows, zero-padded out of bounds,
              where C=1 or 2 depending on include_maxpool_dyes.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (C, L), got {tuple(x.shape)}")
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"indices must be shape (p, 2), got {tuple(indices.shape)}")
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    C, L = x.shape
    idx = indices.to(device=x.device, dtype=torch.long)
    rows = idx[:, 0]            # (p,)
    centers = idx[:, 1]         # (p,)

    if torch.any(rows < 0) or torch.any(rows >= C):
        raise IndexError("Row indices out of range")

    half = window_size // 2
    starts = centers - half

    offsets = torch.arange(window_size, device=x.device, dtype=torch.long)  # (window_size,)
    pos = starts[:, None] + offsets[None, :]                                # (p, window_size)

    mask = (pos >= 0) & (pos < L)
    pos_clamped = pos.clamp(0, L - 1)

    windows = x[rows[:, None], pos_clamped]  # (p, window_size)
    windows = windows * mask.to(dtype=torch.float32)

    if not include_maxpool_dyes:
        return windows.unsqueeze(1)          # (p, 1, window_size)

    # Max-pool over OTHER dyes for the same centered window positions.
    all_dye_windows = x[:, pos_clamped]      # (C, p, window_size)

    row_indices = torch.arange(C, device=x.device)[:, None]
    target_rows = rows[None, :]
    keep_mask = (row_indices != target_rows).unsqueeze(-1)  # (C, p, 1)

    neg_inf = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    masked_windows = torch.where(keep_mask, all_dye_windows, neg_inf)

    other_dyes_max, _ = masked_windows.max(dim=0)            # (p, window_size)
    other_dyes_max = other_dyes_max * mask.to(dtype=x.dtype)

    return torch.stack([windows, other_dyes_max], dim=1)     # (p, 2, window_size)




def find_peaks_torch_indices(
        x: torch.Tensor,
        threshold: float,
        dim: int = -1,
) -> torch.Tensor:
    """
    Plateau-aware peak finder matching SciPy `signal.find_peaks(x, height=threshold)`-style local-max logic:
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
        raise ValueError("Plateau-aware peak detection currently supports `dim` as the last dimension only.")

    x_moved = x.movedim(dim, -1)  # (..., n)
    n = x_moved.size(-1)
    if n < 3:
        return torch.empty((0, x.ndim), dtype=torch.long, device=x.device)

    outer_shape = x_moved.shape[:-1]
    s = x_moved.reshape(-1, n)  # (B, n)
    B = s.size(0)

    # 1) Assign a run id per position (run = maximal contiguous region of equal values).
    change = s[:, 1:] != s[:, :-1]                          # (B, n-1)
    run_id = torch.cat(
        [torch.zeros((B, 1), device=x.device, dtype=torch.long),
         change.long().cumsum(dim=1)],
        dim=1
    )                                                       # (B, n), values in [0, n-1]

    # 2) Make run ids unique across batch by offsetting with b*n.
    gid = (run_id + (torch.arange(B, device=x.device, dtype=torch.long).view(B, 1) * n)).reshape(-1)  # (B*n,)
    pos = torch.arange(n, device=x.device, dtype=torch.long).view(1, n).expand(B, n).reshape(-1)      # (B*n,)

    # 3) For each (batch, run), compute run start=min(pos) and run end=max(pos) using scatter_reduce.
    # scatter_reduce_ supports reductions like 'amin'/'amax' with include_self [web:1].
    start = torch.full((B * n,), n, device=x.device, dtype=torch.long)
    end = torch.full((B * n,), -1, device=x.device, dtype=torch.long)
    start.scatter_reduce_(0, gid, pos, reduce="amin", include_self=True)
    end.scatter_reduce_(0, gid, pos, reduce="amax", include_self=True)

    exists = start < n
    if not bool(exists.any()):
        return torch.empty((0, x.ndim), dtype=torch.long, device=x.device)

    slot = torch.nonzero(exists, as_tuple=False).flatten()  # (R,)
    b = slot // n
    rs = start[slot]
    re = end[slot]

    # 4) Peak condition for a run (rs..re):
    # - interior run: rs>0 and re<n-1 (exclude edges like your implementation)
    # - height >= threshold
    # - height strictly greater than left neighbor and right neighbor
    interior = (rs > 0) & (re < (n - 1))
    if not bool(interior.any()):
        return torch.empty((0, x.ndim), dtype=torch.long, device=x.device)

    b_i = b[interior]
    rs_i = rs[interior]
    re_i = re[interior]

    height = s[b_i, rs_i]          # constant over the run
    left = s[b_i, rs_i - 1]
    right = s[b_i, re_i + 1]

    is_peak = (height >= threshold) & (height > left) & (height > right)
    if not bool(is_peak.any()):
        return torch.empty((0, x.ndim), dtype=torch.long, device=x.device)

    b_p = b_i[is_peak]
    center = (rs_i[is_peak] + re_i[is_peak]) // 2  # midpoint, floor on even length

    # 5) Convert flattened batch index back to outer indices (same as your current code).
    if len(outer_shape) == 0:
        return center.view(-1, 1)

    tmp = b_p
    unraveled = []
    for size in reversed(outer_shape):
        unraveled.append(tmp % size)
        tmp = tmp // size
    unraveled = list(reversed(unraveled))

    return torch.stack([*unraveled, center], dim=1)



def extract_peaks_torch(img: HIDImage,
                        device,
                        threshold: float,
                        window_size: int,
                        include_max_pool_dyes: bool = False,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract peak windows from HIDImage using PyTorch.
    """
    data = img.data[:5] if img.include_size_standard else img.data
    image = torch.from_numpy(data).to(device=device, dtype=torch.float32) # (6, 4096, 1)
    image = image.squeeze() # (6, 4096)
    marker_to_idx = StrategyRegistry.get_scaling_strategy().marker_name_to_dye_idx()
    # print(f"Image shape: {image.shape}, dtype: {image.dtype}, device: {image.device}")

    peak_centers = find_peaks_torch_indices(image, threshold) # (num_peaks, 2) where each row is (dye_index, position)
    # print(f"Found {peak_centers.shape[0]} peaks above threshold {threshold}.")
    # print(indices)

    peak_tensors = extract_windows_torch(image, peak_centers, window_size, include_max_pool_dyes) # (num_peaks, window_size)
    # print(f"Extracted peak windows shape: {peak_tensors.shape}")
    #
    # print(img._panel._panel)

    marker_idx = [marker_to_idx.get(img._panel.get_marker_name_by_dye_and_bp(dye, bp), len(marker_to_idx)) for dye, bp in peak_centers.tolist()]
    marker_idx = torch.tensor(marker_idx, device=device, dtype=torch.long) # (num_peaks,)

    # Should return peak_tensors, marker_idx, peak_centers
    return peak_tensors, marker_idx, peak_centers



def extract_peak_windows(img: HIDImage,
                         threshold: float,
                         window_size: int,
                         use_ground_truth_labels: bool = True,
                         include_raw_annotation: bool = False,
                         include_max_pool_dyes: bool = False) -> Sequence[ExtractedPeak]:
    """
    Extract windows centered on a single representative peak per contiguous run
    above 'threshold'. Flat-topped (plateau) peaks count as one peak.

    Parameters
    ----------
    x : array_like, shape (N,)
        1D input signal.
    threshold : float
        Points strictly greater than this value are considered "above threshold".
    window_size : int
        Length of each extracted window (must be positive).
        The peak lands at index window_size//2.
    pad_value : float, optional
        Value used to pad windows near the edges (default 0.0).

    Returns
    -------
    windows : ndarray, shape (num_peaks, window_size)
        Each row is a window centered on one detected peak index.
    """
    peaks = []
    data = img.data[:5] if img.include_size_standard else img.data # do not include peaks in size standard
    for (dye_index, dye_data) in enumerate(data):
        assert dye_index < 5, f"Dye index out of range. include size standard: {img.include_size_standard}. Shape: {img.data.shape}"
        x = dye_data.squeeze()

        N = x.size

        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if N == 0:
            return np.empty((0, window_size), dtype=x.dtype)


        peak_idxs = scipy.signal.find_peaks(x, height=threshold)

        for peak_idx in peak_idxs[0]:
            peak_height = x[peak_idx]
            peak = ExtractedPeak(img, dye_index, peak_idx, window_size, peak_height,
                                 include_raw_annotation=include_raw_annotation, use_ground_truth=use_ground_truth_labels,
                                 include_max_pool_dyes=include_max_pool_dyes)
            peaks.append(peak)

    # LOGGER.debug(f"Extracted {len(peaks)} peaks from image {img.path.name} with threshold {threshold}.")

    return peaks

