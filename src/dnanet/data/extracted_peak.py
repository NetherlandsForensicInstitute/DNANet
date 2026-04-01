"""ExtractedPeak — data model for a single peak window.

An :class:`ExtractedPeak` represents a fixed-width window of signal data
centered on a detected fluorescence peak. It is the basic unit of data
for the peak classifier and combined PeakNet model.

Each peak carries:
- The signal window (1 or 2 channels × ``window_size`` scan points)
- Its dye channel index and scan-point position
- The marker it falls within (if any)
- An annotation label (``"allele"`` or ``"noise"``)

Design pattern: **Value Object**
    Peaks are lightweight, immutable-ish data carriers. They are
    produced in bulk by :func:`extract_peak_windows` and consumed
    by the peak classification data module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.interpolate

from dnanet.core import Annotation, ClassAnnotation

if TYPE_CHECKING:
    from dnanet.data.image import HIDImage, TrainableElement
    from dnanet.core.marker import Marker



class ExtractedPeak(TrainableElement):
    """A single extracted peak window from a DNA profile.

    Args:
        data: Peak signal array of shape ``(C, W)`` where *C* is 1 (single
            dye) or 2 (dye + max-pooled other dyes), and *W* is the window
            width in scan points.
        dye_index: Index of the dye channel this peak was detected in.
        peak_center: Scan-point position of the peak apex in the full profile.
        window_size: Width of the extraction window.
        peak_height: RFU amplitude at the peak apex.
        label: Annotation label (e.g. ``"allele"``, ``"noise"``).
        marker_name: STR marker name (e.g. ``"D3S1358"``), or ``None``.
        marker_index: Integer index of the marker in the panel, or -1.
    """

    __slots__ = (
        "_data",
        "dye_index",
        "peak_center",
        "window_size",
        "peak_height",
        "label",
        "marker_name",
        "marker_index",
        "peak_basepair",
        "window_start",
    )

    def __init__(
        self,
        data: np.ndarray,
        dye_index: int,
        peak_center: int,
        window_size: int,
        peak_height: float,
        label: str | None = None,
        marker_name: str | None = None,
        peak_basepair: float | None = None,
        marker_index: int = -1,
    ) -> None:
        self._data = data
        self.dye_index = dye_index
        self.peak_center = peak_center
        self.window_size = window_size
        self.peak_height = float(peak_height)
        self.label = label
        self.marker_name = marker_name
        self.marker_index = marker_index
        self.peak_basepair = peak_basepair
        self.window_start: int = peak_center - window_size // 2

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def annotation(self) -> Annotation | ClassAnnotation | None:
        if self.label is None:
            return None
        return ClassAnnotation(self.label)

    @property
    def is_allele(self) -> bool:
        """Whether this peak is labeled as an allele."""
        return self.label == "allele"

    def __repr__(self) -> str:
        return (
            f"ExtractedPeak(dye={self.dye_index}, center={self.peak_center}, "
            f"bp={self.peak_basepair:.1f}, label={self.label!r}, "
            f"marker={self.marker_name!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtractedPeak):
            return NotImplemented
        return (
            self.dye_index == other.dye_index
            and self.peak_center == other.peak_center
            and self.window_size == other.window_size
        )

    def __hash__(self) -> int:
        return hash((self.dye_index, self.peak_center, self.window_size))
