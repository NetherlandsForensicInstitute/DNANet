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

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from scipy.signal import find_peaks

from dnanet.data.extracted_peak import ExtractedPeak


if TYPE_CHECKING:
    from dnanet.data.image import HIDImage


# ---------------------------------------------------------------------------
# Marker-to-index mapping (shared with peak classifier embedding table)
# ---------------------------------------------------------------------------

# Default marker names for PPF6C 6-dye kit (27 markers + out-of-bin = 28).
# Index 0 is reserved for "out of bin" / unknown marker.
_DEFAULT_MARKERS: list[str] = [
    "D3S1358", "TH01", "D21S11", "D18S51",
    "D10S1248", "D1S1656", "D2S1338", "D16S539",
    "D22S1045", "vWA", "D8S1179", "FGA",
    "D2S441", "D12S391", "D19S433", "SE33",
    "D7S820", "CSF1PO", "D13S317", "D5S818",
    "D6S1043", "AMEL", "DYS391", "DYS576",
    "DYS570", "YINDEL", "DYS389II",
]

MARKER_TO_IDX: dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(_DEFAULT_MARKERS)
}
MARKER_TO_IDX["Out of Bin"] = 0

N_MARKERS: int = len(MARKER_TO_IDX)  # 28 by default


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
    include_max_pool_dyes: bool = False,
) -> np.ndarray:
    """Build the data array for a single peak window.

    Args:
        img_data: Full profile ``(D, L)`` or ``(D, L, 1)``.
        dye_index: Dye channel of the peak.
        start: Window start index.
        length: Window width.
        include_max_pool_dyes: If True, adds a second row with the
            max of all *other* dye channels.

    Returns:
        Array of shape ``(1, length)`` or ``(2, length)``.
    """
    dye_slice = _slice_with_padding(img_data, dye_index, start, length)

    if not include_max_pool_dyes:
        return dye_slice[np.newaxis, :]

    if img_data.ndim == 3:
        data_2d = img_data[:, :, 0]
    else:
        data_2d = img_data

    n_dyes = data_2d.shape[0]
    other_indices = [i for i in range(n_dyes) if i != dye_index]
    if other_indices:
        others = np.stack([
            _slice_with_padding(img_data, i, start, length)
            for i in other_indices
        ])
        max_pooled = others.max(axis=0)
    else:
        max_pooled = np.zeros(length, dtype=dye_slice.dtype)

    return np.stack([dye_slice, max_pooled])


def _find_marker_for_peak(
    peak_bp: float,
    dye_index: int,
    panel,
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
        markers = panel._panel if hasattr(panel, "_panel") else panel.markers
    except AttributeError:
        return None, 0

    for marker in markers:
        if getattr(marker, "dye_row", None) != dye_index:
            continue

        # Check if peak falls within marker bin
        alleles = getattr(marker, "alleles", [])
        if alleles:
            left = min(a.left_bin for a in alleles if hasattr(a, "left_bin"))
            right = max(a.right_bin for a in alleles if hasattr(a, "right_bin"))
            if left <= peak_bp <= right:
                name = getattr(marker, "name", str(marker))
                return name, MARKER_TO_IDX.get(name, 0)

    return None, 0


def _label_peak_from_annotation(
    annotation_image: np.ndarray | None,
    dye_index: int,
    peak_center: int,
    padding: int = 1,
) -> str:
    """Determine peak label from the annotation mask.

    Args:
        annotation_image: Binary mask ``(D, L)`` or ``(D, L, 1)``, or None.
        dye_index: Dye channel index.
        peak_center: Scan-point index of peak apex.
        padding: Number of positions around the center to check.

    Returns:
        ``"allele"`` if any annotated position is within padding of center,
        ``"noise"`` otherwise.
    """
    if annotation_image is None:
        return "noise"

    if annotation_image.ndim == 3:
        ann = annotation_image[dye_index, :, 0]
    else:
        ann = annotation_image[dye_index]

    L = ann.shape[0]
    start = max(0, peak_center - padding)
    end = min(L, peak_center + padding + 1)

    if np.any(ann[start:end] > 0):
        return "allele"
    return "noise"


# ---------------------------------------------------------------------------
# Main extraction functions
# ---------------------------------------------------------------------------

def extract_peak_windows(
    image: HIDImage,
    threshold: float,
    window_size: int,
    include_max_pool_dyes: bool = False,
    use_ground_truth: bool = True,
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
        use_ground_truth: If True, label peaks based on the annotation image.

    Returns:
        List of :class:`ExtractedPeak` objects.
    """
    from dnanet.data.extracted_peak import SCAN_TO_BP

    data = image.data
    if data is None:
        return []

    if data.ndim == 3:
        data_2d = data[:, :, 0]
    else:
        data_2d = data

    n_dyes = data_2d.shape[0]
    # Exclude size standard channel (typically channel 6 = index 5)
    dye_count = min(n_dyes, 5)

    annotation_image = None
    if use_ground_truth and image.annotation is not None:
        annotation_image = image.annotation.image

    panel = getattr(image, "_panel", None)

    peaks: list[ExtractedPeak] = []

    for dye_idx in range(dye_count):
        signal = data_2d[dye_idx]
        peak_indices, _ = find_peaks(signal, height=threshold)

        for peak_idx in peak_indices:
            start = peak_idx - window_size // 2
            peak_data = _build_peak_data(
                data, dye_idx, start, window_size, include_max_pool_dyes,
            )

            label = _label_peak_from_annotation(
                annotation_image, dye_idx, peak_idx,
            )

            peak_bp = float(SCAN_TO_BP(peak_idx))
            marker_name, marker_index = _find_marker_for_peak(
                peak_bp, dye_idx, panel,
            )

            peaks.append(ExtractedPeak(
                data=peak_data,
                dye_index=dye_idx,
                peak_center=int(peak_idx),
                window_size=window_size,
                peak_height=float(signal[peak_idx]),
                label=label,
                marker_name=marker_name,
                marker_index=marker_index,
            ))

    return peaks


def extract_peaks_torch(
    image: HIDImage,
    device: torch.device | str,
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
    if data is None:
        empty_w = torch.zeros(0, 1, window_size, device=device)
        empty_m = torch.zeros(0, dtype=torch.long, device=device)
        empty_c = torch.zeros(0, 2, dtype=torch.long, device=device)
        return empty_w, empty_m, empty_c

    if data.ndim == 3:
        data_2d = data[:, :, 0]
    else:
        data_2d = data

    x = torch.from_numpy(data_2d.astype(np.float32)).to(device)
    n_dyes = min(x.shape[0], 5)  # exclude size standard

    all_windows = []
    all_markers = []
    all_centers = []

    panel = getattr(image, "_panel", None)
    half = window_size // 2

    for dye_idx in range(n_dyes):
        signal_np = data_2d[dye_idx]
        peak_indices, _ = find_peaks(signal_np, height=threshold)

        for peak_idx in peak_indices:
            # Extract window with padding
            start = peak_idx - half
            end = start + window_size

            # Build window on CPU first for simplicity
            window_np = _build_peak_data(
                data, dye_idx, start, window_size, include_max_pool_dyes,
            )
            all_windows.append(
                torch.from_numpy(window_np.astype(np.float32))
            )

            # Marker lookup
            from dnanet.data.extracted_peak import SCAN_TO_BP
            bp = float(SCAN_TO_BP(peak_idx))
            _, marker_idx = _find_marker_for_peak(bp, dye_idx, panel)
            all_markers.append(marker_idx)
            all_centers.append([dye_idx, peak_idx])

    if not all_windows:
        c = 2 if include_max_pool_dyes else 1
        return (
            torch.zeros(0, c, window_size, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, 2, dtype=torch.long, device=device),
        )

    peak_windows = torch.stack(all_windows).to(device)
    marker_idxs = torch.tensor(all_markers, dtype=torch.long, device=device)
    peak_centers = torch.tensor(all_centers, dtype=torch.long, device=device)

    return peak_windows, marker_idxs, peak_centers
