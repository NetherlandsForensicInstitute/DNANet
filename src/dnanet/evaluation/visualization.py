"""Visualization functions for DNA electropherogram profiles.

Provides matplotlib-based plotting for EPG profiles with optional
ground-truth annotations and prediction overlays.

Design pattern: **Pure Functions**
    All plotting functions accept data arrays and domain objects as inputs,
    returning matplotlib Figure objects. No side effects beyond figure
    creation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import matplotlib


matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from dnanet.core.constants import LabelCategory


if TYPE_CHECKING:
    from matplotlib.figure import Figure


# Standard dye channel colors for forensic EPG visualization
DYE_COLORS: tuple[str, ...] = ("blue", "green", "black", "red", "purple", "orange")


def plot_profile(
    signal: np.ndarray,
    *,
    annotation: np.ndarray | None = None,
    prediction: np.ndarray | None = None,
    title: str | None = None,
    dye_colors: Sequence[str] | None = None,
    figsize: tuple[int, int] = (20, 20),
) -> Figure:
    """Plot a full DNA profile with optional multiclass annotations.

    Args:
        signal: (C, L) EPG signal data (one row per dye channel).
        annotation: (C, L) class-index ground-truth annotation map.
        prediction: (C, L) class-index prediction map.
        title: Optional figure title.
        dye_colors: Colors for each dye channel. Defaults to standard
            forensic dye colors.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure.
    """
    signal = _coerce_signal(signal)
    colors = dye_colors or DYE_COLORS
    n_dyes, signal_length = signal.shape

    annotation_map = None
    if annotation is not None:
        annotation_map = coerce_class_map(
            annotation,
            signal_shape=signal.shape,
            source="annotation",
        )

    prediction_map = None
    if prediction is not None:
        prediction_map = coerce_class_map(
            prediction,
            signal_shape=signal.shape,
            source="prediction",
        )

    track_names = ["signal"]
    track_height_ratios = {"signal": 6, "annotation": 1, "prediction": 1}
    if annotation_map is not None:
        track_names.append("annotation")
    if prediction_map is not None:
        track_names.append("prediction")

    num_tracks = len(track_names)
    fig, axes = plt.subplots(
        n_dyes * num_tracks,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                track_height_ratios[track_name]
                for _ in range(n_dyes)
                for track_name in track_names
            ],
        },
    )
    axes = np.atleast_1d(axes).tolist()
    axes_by_track = {
        track_name: axes[track_index::num_tracks]
        for track_index, track_name in enumerate(track_names)
    }
    signal_axes = axes_by_track["signal"]

    _plot_lines(signal_axes, signal, colors)
    for signal_ax in signal_axes:
        signal_ax.set_xlim(-0.5, signal_length - 0.5)
        signal_ax.margins(x=0)
        signal_ax.spines["top"].set_visible(False)
        signal_ax.spines["right"].set_visible(False)

    present_classes: set[int] = set()
    if annotation_map is not None:
        present_classes.update(
            _plot_class_tracks(
                axes_by_track["annotation"],
                annotation_map,
                lane_label="Ann",
            )
        )
    if prediction_map is not None:
        present_classes.update(
            _plot_class_tracks(
                axes_by_track["prediction"],
                prediction_map,
                lane_label="Pred",
            )
        )

    if title:
        fig.suptitle(title, fontsize=16)

    if present_classes:
        fig.legend(
            handles=[
                Patch(
                    facecolor=LabelCategory.from_index(class_idx).color,
                    edgecolor="none",
                    label=LabelCategory.from_index(class_idx).display_name,
                )
                for class_idx in sorted(present_classes)
            ],
            loc="upper right",
        )

    axes[-1].set_xlabel("Scanpoint")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def coerce_class_map(
    data: np.ndarray,
    *,
    signal_shape: tuple[int, int],
    source: str,
) -> np.ndarray:
    """Normalize plotting labels to a 2-D class-index map.

    Supports:
      - (C, L) integer class maps
      - (K, C, L) class-first logits/one-hot arrays
      - (C, L, K) class-last logits/one-hot arrays
      - singleton class/channel variants of the shapes above

    Binary probability maps are thresholded at 0.5 only when there is a
    single class channel. Multiclass arrays are reduced with argmax.
    """
    result = np.asarray(data)

    if result.ndim == 1:
        result = result[np.newaxis, :]

    if result.ndim == 2:
        return _coerce_integer_class_map(
            result,
            signal_shape=signal_shape,
            source=source,
            threshold_binary=source == "prediction",
        )

    if result.ndim == 3 and result.shape[1:] == signal_shape:
        if result.shape[0] == 1:
            return _coerce_integer_class_map(
                result[0],
                signal_shape=signal_shape,
                source=source,
                threshold_binary=source == "prediction",
            )
        return _coerce_integer_class_map(
            result.argmax(axis=0),
            signal_shape=signal_shape,
            source=source,
        )

    if result.ndim == 3 and result.shape[:2] == signal_shape:
        if result.shape[-1] == 1:
            return _coerce_integer_class_map(
                result[..., 0],
                signal_shape=signal_shape,
                source=source,
                threshold_binary=source == "prediction",
            )
        return _coerce_integer_class_map(
            result.argmax(axis=-1),
            signal_shape=signal_shape,
            source=source,
        )

    raise ValueError(
        f"{source} must have shape {signal_shape}, (classes, *{signal_shape}), "
        f"or (*{signal_shape}, classes); got {result.shape}."
    )


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
        _plot_segmentation_mask(
            [ax],
            ann_slice[np.newaxis, :],
            color="green",
            alpha=0.55,
            y_range=(0.88, 0.94),
            min_width=3,
        )

    if prediction is not None:
        pred_slice = prediction[dye_row, scan_slice]
        _plot_segmentation_mask(
            [ax],
            ((pred_slice > 0.5).astype(int))[np.newaxis, :],
            color="orange",
            alpha=0.75,
            y_range=(0.95, 1.0),
            min_width=3,
        )

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _coerce_signal(signal: np.ndarray) -> np.ndarray:
    """Return signal as a 2-D ``(num_dyes, scanpoints)`` array."""
    result = np.asarray(signal)
    if result.ndim == 1:
        result = result[np.newaxis, :]
    if result.ndim == 3 and result.shape[-1] == 1:
        result = result[..., 0]
    if result.ndim != 2:
        raise ValueError(
            f"signal must be a 2-D array of shape (num_dyes, scanpoints); got {result.shape}."
        )
    return result


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

    for i, (c, row, max_val) in enumerate(zip(color, data, scale_to, strict=True)):
        y = row.copy()
        if max_val is not None and max_val > 0:
            y_max = max(y.max(), 1)
            y = y * (max_val / y_max)
        axs[i].plot(y, c=c, alpha=alpha)


def _plot_class_tracks(
    axs: Sequence[plt.Axes],
    class_map: np.ndarray | None,
    *,
    lane_label: str,
) -> set[int]:
    """Plot per-class mini-lanes and return the set of drawn class indices."""
    present_classes: set[int] = set()

    for ax in axs:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_ylabel(lane_label, rotation=0, labelpad=18, va="center")
        ax.margins(x=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if class_map is None:
        return present_classes

    for ax, class_row in zip(axs, class_map, strict=True):
        for start, end, class_idx in _iter_class_spans(class_row):
            category = LabelCategory.from_index(class_idx)
            width = end - start
            present_classes.add(class_idx)

            if width == 1:
                ax.scatter(
                    [start],
                    [0.5],
                    marker="s",
                    s=36,
                    color=category.color,
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=3,
                )
            else:
                ax.barh(
                    y=0.5,
                    width=width,
                    left=start - 0.5,
                    height=0.7,
                    color=category.color,
                    edgecolor="none",
                    align="center",
                )

    return present_classes


def _coerce_integer_class_map(
    class_map: np.ndarray,
    *,
    signal_shape: tuple[int, int],
    source: str,
    threshold_binary: bool = False,
) -> np.ndarray:
    """Validate a 2-D class map and convert it to integer indices."""
    if class_map.shape != signal_shape:
        raise ValueError(
            f"{source} must match signal shape {signal_shape}; got {class_map.shape}."
        )

    rounded = np.rint(class_map)
    if np.allclose(class_map, rounded):
        class_indices = rounded.astype(np.int32, copy=False)
    elif threshold_binary:
        class_indices = (np.asarray(class_map) > 0.5).astype(np.int32, copy=False)
    else:
        raise ValueError(
            f"{source} must be an integer class map for plotting; got non-integer values."
        )

    num_classes = len(LabelCategory)
    if np.any(class_indices < 0) or np.any(class_indices >= num_classes):
        raise ValueError(
            f"{source} class indices must be in [0, {num_classes}); "
            f"got range [{class_indices.min()}, {class_indices.max()}]."
        )

    return class_indices


def _iter_class_spans(class_row: np.ndarray) -> list[tuple[int, int, int]]:
    """Return contiguous non-zero class spans as ``(start, end, class_idx)``."""
    labels = np.asarray(class_row).astype(np.int32, copy=False).flatten()
    if labels.size == 0:
        return []

    spans: list[tuple[int, int, int]] = []
    start = 0
    current = int(labels[0])

    for index in range(1, labels.size + 1):
        if index < labels.size and int(labels[index]) == current:
            continue

        if current != 0:
            spans.append((start, index, current))

        if index < labels.size:
            start = index
            current = int(labels[index])

    return spans


def _plot_segmentation_mask(
    axs: Sequence[plt.Axes],
    mask: np.ndarray,
    color: str = "green",
    alpha: float = 0.5,
    y_range: tuple[float, float] = (0.0, 1.0),
    min_width: int = 1,
) -> None:
    """Plot segmentation mask as colored scanpoint tracks.

    Args:
        axs: One axis per dye channel.
        mask: (C, L) binary mask.
        color: Fill color.
        alpha: Fill alpha.
        y_range: Vertical band in axis coordinates.
        min_width: Minimum visible width of a positive region in scanpoints.
    """
    for ax, dye_mask in zip(axs, mask, strict=True):
        for start, end in _iter_mask_spans(dye_mask, min_width=min_width):
            ax.axvspan(
                start,
                end,
                ymin=y_range[0],
                ymax=y_range[1],
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
            )


def _iter_mask_spans(
    mask_row: np.ndarray,
    *,
    min_width: int = 1,
) -> list[tuple[float, float]]:
    """Return contiguous positive mask spans with optional display widening."""
    binary = np.asarray(mask_row).astype(bool).flatten()
    if binary.size == 0 or not np.any(binary):
        return []

    padded = np.pad(binary.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    spans: list[tuple[float, float]] = []

    for start, end in zip(starts, ends, strict=True):
        width = end - start
        if width < min_width:
            missing = min_width - width
            left_extra = missing // 2
            right_extra = missing - left_extra
            start = max(0, start - left_extra)
            end = min(binary.size, end + right_extra)

            # If span hit boundary, rebalance to keep requested minimum width.
            width = end - start
            if width < min_width:
                if start == 0:
                    end = min(binary.size, min_width)
                else:
                    start = max(0, binary.size - min_width)

        spans.append((float(start), float(end)))

    return spans
