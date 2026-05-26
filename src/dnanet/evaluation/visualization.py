"""Visualization functions for DNA electropherogram profiles.

Provides matplotlib-based plotting for EPG profiles with optional
ground-truth annotations and prediction overlays.

Design pattern: **Pure Functions**
    All plotting functions accept data arrays and domain objects as inputs,
    returning matplotlib Figure objects. No side effects beyond figure
    creation.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from loguru import logger

# Standard dye channel colors for forensic EPG visualization
DYE_COLORS: tuple[str, ...] = ("blue", "green", "black", "red", "purple", "orange")


def plot_profile(
    signal: np.ndarray,
    *,
    annotation: np.ndarray | None = None,
    prediction: np.ndarray | None = None,
    prediction_as_mask: bool = True,
    title: str | None = None,
    dye_colors: Sequence[str] | None = None,
    figsize: tuple[int, int] = (20, 20),
) -> Figure:
    """Plot a full DNA profile with optional annotations and predictions.

    Args:
        signal: (C, L) EPG signal data (one row per dye channel).
        annotation: (C, L) binary ground-truth annotation mask.
        prediction: (C, L) predicted mask (probabilities or binary).
        prediction_as_mask: If True, show predictions as colored regions.
            If False, show as an overlaid line.
        title: Optional figure title.
        dye_colors: Colors for each dye channel. Defaults to standard
            forensic dye colors.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure.
    """
    colors = dye_colors or DYE_COLORS
    n_dyes = len(signal)
    dyes_max = signal.max(axis=1)

    fig, axs = plt.subplots(n_dyes, figsize=figsize)
    if n_dyes == 1:
        axs = [axs]

    # Plot signal
    _plot_lines(axs, signal, colors)

    # Plot annotation as green background regions
    if annotation is not None:
        _plot_segmentation_mask(axs, annotation, dyes_max, color="green", alpha=0.4)

    # Plot prediction
    if prediction is not None:
        if prediction_as_mask:
            pred_binary = (prediction > 0.5).astype(int)
            _plot_segmentation_mask(axs, pred_binary, dyes_max, color="orange", alpha=0.5)
        else:
            _plot_lines(axs, prediction, "orange", scale_to=dyes_max)

    if title:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    return fig


def plot_profile_marker(
    signal: np.ndarray,
    scaler: np.ndarray,
    marker_bp_range: tuple[float, float],
    dye_row: int,
    *,
    annotation: np.ndarray | None = None,
    prediction: np.ndarray | None = None,
    title: str | None = None,
    zoom: tuple[int, int] | None = None,
) -> Figure:
    """Plot a single marker region of an EPG profile.

    Args:
        signal: (C, L) full EPG signal.
        scaler: (L,) scan-to-base-pair mapping array.
        marker_bp_range: (left_bp, right_bp) bin range of the marker.
        dye_row: Dye channel index for this marker.
        annotation: (C, L) ground-truth annotation mask.
        prediction: (C, L) predicted mask.
        title: Plot title (usually the marker name).
        zoom: Optional (start, end) scan indices to zoom into.

    Returns:
        The matplotlib Figure.
    """
    left_bp, right_bp = marker_bp_range
    # Find scan indices corresponding to the marker bin range
    bp_indices = np.argmin(
        np.abs(scaler[np.newaxis, :] - np.array([[left_bp], [right_bp]])),
        axis=1,
    )
    scan_slice = slice(bp_indices[0], bp_indices[1])
    if zoom:
        # Apply zoom within the marker range
        start = bp_indices[0] + zoom[0]
        end = min(bp_indices[0] + zoom[1], bp_indices[1])
        scan_slice = slice(start, end)

    marker_signal = signal[dye_row, scan_slice]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    color = DYE_COLORS[dye_row] if dye_row < len(DYE_COLORS) else "black"
    ax.plot(marker_signal, color=color)

    if annotation is not None:
        ann_slice = annotation[dye_row, scan_slice]
        ax.fill_between(
            np.arange(len(ann_slice)), 0, marker_signal.max(),
            where=ann_slice.flatten() == 1,
            color="green", alpha=0.4,
            transform=ax.get_xaxis_transform(),
        )

    if prediction is not None:
        pred_slice = prediction[dye_row, scan_slice]
        ax.fill_between(
            np.arange(len(pred_slice)), 0, marker_signal.max(),
            where=(pred_slice > 0.5).astype(int).flatten(),
            color="orange", alpha=0.4,
            transform=ax.get_xaxis_transform(),
        )

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _plot_lines(
    axs: Sequence[plt.Axes],
    data: np.ndarray,
    color: Any = "black",
    scale_to: np.ndarray | None = None,
    alpha: float = 1.0,
) -> None:
    """Plot a line on each dye channel.

    Args:
        axs: One axis per dye channel.
        data: (C, L) data to plot.
        color: Single color or per-dye color sequence.
        scale_to: If provided, rescale each row to this max value.
        alpha: Line alpha.
    """
    if scale_to is None:
        scale_to = [None] * len(data)

    if isinstance(color, str) or (isinstance(color, tuple) and isinstance(color[0], float)):
        color = [color] * len(data)

    for i, (c, row, max_val) in enumerate(zip(color, data, scale_to)):
        y = row.copy()
        if max_val is not None and max_val > 0:
            y_max = max(y.max(), 1)
            y = y * (max_val / y_max)
        axs[i].plot(y, c=c, alpha=alpha)


def _plot_segmentation_mask(
    axs: Sequence[plt.Axes],
    mask: np.ndarray,
    max_values: np.ndarray,
    color: str = "green",
    alpha: float = 0.5,
) -> None:
    """Plot segmentation mask as colored background regions.

    Args:
        axs: One axis per dye channel.
        mask: (C, L) binary mask.
        max_values: Max signal value per dye (for fill height).
        color: Fill color.
        alpha: Fill alpha.
    """
    for i, (dye_mask, max_val) in enumerate(zip(mask, max_values)):
        axs[i].fill_between(
            np.arange(len(dye_mask)), 0, max_val,
            where=dye_mask.flatten() == 1,
            color=color, alpha=alpha,
            transform=axs[i].get_xaxis_transform(),
        )
