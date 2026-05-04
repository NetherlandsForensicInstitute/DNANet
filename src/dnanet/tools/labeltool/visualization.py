"""Visualization functions for the interactive label tool.

Extends the base ``dnanet.evaluation.visualization`` module with features
needed for interactive annotation: automatic peak detection, span creation,
base-pair axis formatting, and multi-annotator display.
"""

from __future__ import annotations

from typing import Any, Tuple, Iterable, Sequence, TYPE_CHECKING

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.patches import Rectangle


from dnanet.tools.labeltool.tool import bp_to_scan, scan_to_bp

if TYPE_CHECKING:
    from dnanet.data import HIDDataset
    from dnanet.data.image import HIDImage
    from dnanet.tools.labeltool.interactivity import Interactivity
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from functools import partial

# Canonical dye channel names used in CSV annotation files.
# These are fixed identifiers (NOT the same as plot colors from the scaling
# strategy, which may differ — e.g. "black" instead of "yellow").
DNA_CHANNELS: list[str] = [
    "blue", "green", "yellow", "red", "purple", "orange",
]


def get_peaks(
    profile: np.ndarray,
    min_rfu: float = 200,
) -> list[list[tuple[int, int, int]]]:
    """Detect peaks and their boundaries in an EPG profile.

    Uses scipy's ``find_peaks`` for detection, then walks left/right
    to find where descent stops for boundary estimation.

    Args:
        profile: (C, L, 1) or (C, L) EPG signal array.
        min_rfu: Minimum RFU height threshold.

    Returns:
        List of lists (one per dye), each containing
        ``(start_idx, end_idx, peak_idx)`` tuples.
    """
    if profile.ndim == 3:
        data = profile[:, :, 0]
    else:
        data = profile

    all_peaks: list[list[tuple[int, int, int]]] = []
    for dye in data:
        peak_ids, _ = find_peaks(dye, height=min_rfu)
        dye_peaks: list[tuple[int, int, int]] = []
        for peak_idx in peak_ids:
            # Walk right while descending
            end = peak_idx
            while end + 1 < len(dye) and dye[end] > dye[end + 1]:
                end += 1
            # Walk left while ascending
            start = peak_idx
            while start > 0 and dye[start - 1] < dye[start]:
                start -= 1
            dye_peaks.append((start, end, peak_idx))
        all_peaks.append(dye_peaks)
    return all_peaks


def add_initial_spans(
    axs: Sequence[Axes],
    peak_ranges: list[list[tuple[int, int, int]]],
) -> list[dict[str, Any]]:
    """Create initial unlabeled spans from detected peaks.

    Args:
        axs: One axis per dye channel.
        peak_ranges: Output of :func:`get_peaks`.

    Returns:
        List of span dicts.
    """
    spans: list[dict[str, Any]] = []
    for i, ax in enumerate(axs):
        if i >= len(peak_ranges):
            break
        for start_x, end_x, peak_idx in peak_ranges[i]:
            x0, x1 = sorted([start_x, end_x])
            artist = ax.axvspan(x0, x1, color="gray", alpha=0.2)
            spans.append({
                "artist": artist,
                "ax": ax,
                "category": None,
                "x0": x0,
                "x1": x1,
                "peak_idx": peak_idx,
            })
    return spans


def plot_profile_interactive(
    hid_images: HIDDataset,
    *,
    interactive: type[Interactivity] | partial | None = None,
    spans_by_profile: dict[str, list[dict[str, Any]]] | None = None,
    min_rfu_peak_detection: int | None = 200,
    plot_allele_bins: bool = False,
    title: bool = True,
) -> Figure | None:
    """Plot EPG profiles with optional interactive annotation.

    This is the main plotting entry point for the label tool, combining
    profile visualization with span management and interactivity.

    Args:
        hid_images: Sequence of HIDImage objects to plot.
        interactive: Optional Interactivity subclass (or partial) for
            interactive annotation.
        spans_by_profile: Pre-existing annotations grouped by profile name.
        min_rfu_peak_detection: RFU threshold for automatic peak detection.
            None disables auto-detection.
        plot_allele_bins: Whether to show allele bin positions.
        title: Whether to add profile name as title.

    Returns:
        The last matplotlib Figure, or None if no images had data.
    """
    # Plot colors from the scaling strategy (for line rendering)
    dye_colors = ['blue', 'green', 'black', 'red', 'purple', 'orange']

    fig = None
    for image in hid_images:
        img = image.data
        if img is None:
            logger.warning("No image data for {}", image.path)
            continue

        # Squeeze trailing dimension if present: (C, L, 1) -> (C, L)
        if img.ndim == 3:
            img_2d = img[:, :, 0]
        else:
            img_2d = img

        dyes_max = img_2d.max(axis=1)
        n_dyes = len(img_2d)
        fig, axs = plt.subplots(n_dyes, figsize=(20, 20), sharex=True)
        if n_dyes == 1:
            axs = [axs]

        # Plot DNA profile lines
        for i, (color, dye) in enumerate(zip(dye_colors, img_2d, strict=True)):
            axs[i].plot(dye, c=color)
            axs[i].set_ylim(0, dyes_max[i])

        # Plot ground-truth annotation
        if image.annotation is not None:
            ann = image.annotation.data
            if ann.ndim == 3:
                ann = ann[:, :, 0]
            for i, (dye_ann, max_val) in enumerate(zip(ann, dyes_max)):
                axs[i].fill_between(
                    np.arange(len(dye_ann)), 0, max_val,
                    where=dye_ann.flatten() == 1,
                    color="green", alpha=0.5,
                    transform=axs[i].get_xaxis_transform(),
                )

        # Load pre-existing spans
        profile_name = image.path.stem.split("/")[-1]
        spans = None
        users = None

        if spans_by_profile and profile_name in spans_by_profile:
            spans = spans_by_profile[profile_name]
            users = list({span["annotator"] for span in spans})

            # Build lookup from canonical dye name -> axis index.
            # The span["dye"] stores a canonical channel name (e.g. "yellow")
            # which may differ from the plot color (e.g. "black").
            dye_name_to_idx: dict[str, int] = {}
            for idx, name in enumerate(DNA_CHANNELS[:n_dyes]):
                dye_name_to_idx[name] = idx
            # Also map plot colors so spans saved with either convention work
            for idx, color in enumerate(dye_colors[:n_dyes]):
                dye_name_to_idx.setdefault(color, idx)

            for span in spans:
                i_dye = dye_name_to_idx.get(span["dye"])
                if i_dye is None:
                    logger.warning(
                        "Unknown dye '{}' in span, skipping", span["dye"],
                    )
                    continue
                span["ax"] = axs[i_dye]
                if len(users) == 1:
                    span["artist"] = axs[i_dye].axvspan(
                        span["x0"], span["x1"],
                        color=span["color"], alpha=span["alpha"],
                    )
                else:
                    # Multi-annotator: stack rectangles vertically
                    y_min, y_max = axs[i_dye].get_ylim()
                    y_range = y_max - y_min
                    rect_height = y_range / len(users)
                    idx = users.index(span["annotator"])
                    y_bottom = y_min + idx * rect_height
                    rect = Rectangle(
                        (span["x0"], y_bottom),
                        width=span["x1"] - span["x0"],
                        height=rect_height,
                        facecolor=span["color"],
                        alpha=0.7,
                    )
                    span["artist"] = axs[i_dye].add_patch(rect)

            # Remove spans that couldn't be mapped to an axis
            if spans:
                spans = [s for s in spans if s.get("ax") is not None]
                logger.info("Loaded {} annotations for {}", len(spans), profile_name)

        # Auto-detect peaks if no spans
        if not spans and min_rfu_peak_detection is not None:
            spans = add_initial_spans(
                axs=axs,
                peak_ranges=get_peaks(profile=img, min_rfu=min_rfu_peak_detection),
            )
            logger.info(
                "Auto-detected {} peaks for {} (threshold={})",
                len(spans), profile_name, min_rfu_peak_detection,
            )

        # Title
        title_string = ""
        if title:
            title_string = f"{profile_name}\n"
            if users:
                title_string += " ".join(users)
            fig.suptitle(title_string, fontsize=16)

        # Activate interactivity — pass canonical dye channel names for CSV I/O
        interactive_instance = None
        if interactive is not None:
            interactive_instance = interactive(
                fig, axs, spans, dye_names=DNA_CHANNELS[:n_dyes],
            )
            interactive_instance.activate_interactivity()

        # Base-pair x-axis formatting
        major_bp = np.arange(0, 500, 25)
        major_scan = bp_to_scan(major_bp)
        for ax in axs:
            ax.xaxis.set_major_locator(FixedLocator(major_scan))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"{scan_to_bp(x):.0f}")
            )

        plt.show()

        # If the user pressed 'q', stop iterating through profiles.
        if interactive_instance is not None and interactive_instance.quit_requested:
            logger.info("User requested quit — exiting.")
            break

    return fig
