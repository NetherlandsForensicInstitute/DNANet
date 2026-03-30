"""Ladder — base-pair calibration via ladder profiles.

Design pattern: **Composition over Inheritance**
    The original DNANet had ``Ladder(HIDImage)`` — a subclass of HIDImage.
    This was problematic because a Ladder IS NOT a sample; it has a different
    purpose (calibration) and different data flow (peak finding → panel
    adjustment). Using composition, a Ladder HAS an HIDImage for its raw data,
    but exposes a completely different interface.

Design pattern: **Factory Method** (``from_hid_file``)
    The constructor does heavy work (peak finding, panel adjustment), so
    we provide a factory method that can return ``None`` on failure instead
    of raising exceptions for every bad ladder file in a directory of hundreds.

How ladder-based panel adjustment works:
    1. The ladder contains (almost) all alleles present in the kit panel.
    2. We detect peaks in each dye channel of the ladder profile.
    3. We match detected peaks to known alleles (by count per dye).
    4. For each allele, the peak's pixel position → base-pair position
       via the size-standard scaler gives us the *actual* base-pair location.
    5. Alleles not in the ladder are inter/extrapolated from neighboring alleles.
    6. The result is an "adjusted panel" where base-pair values reflect the
       real electrophoresis conditions of this run.

    See: https://softgenetics.com/PDF/CalibratingPanelautoadjust_icon.pdf
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.signal import find_peaks

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.panel import Panel
from dnanet.core.types import PathLike
from dnanet.data.strategies.scaling import ScalingStrategy


# ---------------------------------------------------------------------------
# Ladder allele catalog (loaded from CSV once)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderAlleleCatalog:
    """Catalog of alleles expected in a ladder, organized by dye row.

    Loaded from a CSV file with columns: Marker, Allele, Dye.
    """

    alleles_by_dye: dict[int, list[tuple[str, str]]] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: PathLike) -> LadderAlleleCatalog:
        """Load ladder allele definitions from a CSV file.

        Args:
            path: Path to CSV with columns ``Marker``, ``Allele``, ``Dye``.
        """
        alleles: dict[int, list[tuple[str, str]]] = defaultdict(list)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dye = int(row["Dye"])
                alleles[dye].append((row["Marker"], row["Allele"]))
        return cls(alleles_by_dye=dict(alleles))

    def expected_count(self, dye_row: int) -> int:
        """Number of alleles expected on a given dye row."""
        return len(self.alleles_by_dye.get(dye_row, []))


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

class Ladder:
    """Calibration ladder for base-pair panel adjustment.

    A Ladder wraps a ladder HID file and uses it to create an *adjusted panel*
    where allele base-pair positions reflect the actual electrophoresis
    conditions of the run.

    Attributes:
        path: Path to the ladder HID file.
        peak_indices: Detected peak indices per dye (list of arrays).
        adjusted_panel: Panel with base-pairs adjusted from ladder peaks.
                        ``None`` if peak detection failed.
    """

    def __init__(
        self,
        path: Path,
        data: np.ndarray,
        scaler: np.ndarray,
        catalog: LadderAlleleCatalog,
        num_dyes: int,
        peak_indices: list[np.ndarray],
        adjusted_panel: Panel | None,
    ) -> None:
        self.path = path
        self._data = data
        self._scaler = scaler
        self._catalog = catalog
        self._num_dyes = num_dyes
        self.peak_indices = peak_indices
        self.adjusted_panel = adjusted_panel

    @classmethod
    def from_hid_data(
        cls,
        path: PathLike,
        data: np.ndarray,
        scaler: np.ndarray,
        catalog: LadderAlleleCatalog,
        default_panel: Panel,
        num_dyes: int = 5,
    ) -> Ladder | None:
        """Build a Ladder from pre-loaded HID data.

        Args:
            path: Path to the ladder HID file (for logging).
            data: Signal array of shape ``(num_dyes+1, signal_length, 1)``.
            scaler: Base-pair values per pixel, shape ``(signal_length,)``.
            catalog: Expected alleles per dye row.
            default_panel: The default (unadjusted) panel.
            num_dyes: Number of analysis dye channels (excl. size standard).

        Returns:
            A ``Ladder`` instance, or ``None`` if peak detection fails.
        """
        path = Path(path)

        # Find peaks on each dye
        peak_indices = cls._find_all_peaks(data, catalog, num_dyes, path)
        if peak_indices is None:
            return None

        # Adjust the panel using detected peak positions
        adjusted = cls._adjust_panel(
            peak_indices, catalog, scaler, default_panel, num_dyes
        )

        return cls(
            path=path,
            data=data,
            scaler=scaler,
            catalog=catalog,
            num_dyes=num_dyes,
            peak_indices=peak_indices,
            adjusted_panel=adjusted,
        )

    # -- Peak detection --------------------------------------------------- #

    @classmethod
    def _find_all_peaks(
        cls,
        data: np.ndarray,
        catalog: LadderAlleleCatalog,
        num_dyes: int,
        path: Path,
    ) -> list[np.ndarray] | None:
        """Find peaks on every dye and validate against expected counts."""
        all_peaks: list[np.ndarray] = []

        for dye_row in range(num_dyes):
            dye = data[dye_row, :, 0] if data.ndim == 3 else data[dye_row, :]
            dynamic_threshold = np.max(dye) * 0.6

            # Pad to detect peaks at index 0
            padded = np.concatenate((np.zeros(1), dye))
            peaks, _ = find_peaks(padded, height=dynamic_threshold)
            peaks = peaks - 1  # remove padding offset

            expected = catalog.expected_count(dye_row)
            if len(peaks) != expected:
                logger.warning(
                    "Ladder {}: dye {} expected {} peaks, found {}",
                    path.name, dye_row, expected, len(peaks),
                )
                return None

            all_peaks.append(peaks)

        return all_peaks

    # -- Panel adjustment ------------------------------------------------- #

    @classmethod
    def _adjust_panel(
        cls,
        peak_indices: list[np.ndarray],
        catalog: LadderAlleleCatalog,
        scaler: np.ndarray,
        default_panel: Panel,
        num_dyes: int,
    ) -> Panel:
        """Create an adjusted panel using ladder peak positions.

        For each allele:
        - If it was found in the ladder → use the scaler to get its real bp
        - If not → inter/extrapolate from neighboring alleles on the same marker
        """
        adjusted_markers: list[Marker] = []

        for dye_row in range(num_dyes):
            catalog_alleles = catalog.alleles_by_dye.get(dye_row, [])
            panel_markers = [
                m for m in default_panel.markers if m.dye_row == dye_row
            ]

            # Build mapping: (marker_name, allele_name) → base_pair from ladder
            marker_allele_to_bp: dict[tuple[str, str], float] = {}
            for (marker_name, allele_name), peak_idx in zip(
                catalog_alleles, peak_indices[dye_row]
            ):
                marker_allele_to_bp[(marker_name, allele_name)] = float(
                    scaler[peak_idx]
                )

            for marker in panel_markers:
                adjusted_alleles = cls._adjust_marker_alleles(
                    marker, marker_allele_to_bp
                )
                adjusted_markers.append(
                    Marker(
                        name=marker.name,
                        dye_row=dye_row,
                        alleles=frozenset(adjusted_alleles),
                    )
                )

        return Panel(markers=tuple(adjusted_markers))

    @classmethod
    def _adjust_marker_alleles(
        cls,
        marker: Marker,
        marker_allele_to_bp: dict[tuple[str, str], float],
    ) -> list[Allele]:
        """Adjust allele base-pair positions for a single marker."""
        adjusted: list[Allele] = []
        seen_names: set[str] = set()

        for idx, allele in enumerate(marker.alleles):
            if allele.name in seen_names:
                continue
            seen_names.add(allele.name)

            ladder_bp = marker_allele_to_bp.get((marker.name, allele.name))

            if ladder_bp is not None:
                # Direct match: use the ladder-calibrated bp
                adjusted.append(
                    Allele(
                        name=allele.name,
                        base_pair=ladder_bp,
                        left_bin=allele.left_bin,
                        right_bin=allele.right_bin,
                    )
                )
            else:
                # Inter/extrapolate from neighboring alleles
                extrapolated = cls._extrapolate_allele(
                    marker, allele, idx, marker_allele_to_bp
                )
                adjusted.append(extrapolated)

        return adjusted

    @classmethod
    def _extrapolate_allele(
        cls,
        marker: Marker,
        allele: Allele,
        index: int,
        marker_allele_to_bp: dict[tuple[str, str], float],
    ) -> Allele:
        """Inter/extrapolate a missing allele's bp from known neighbors.

        Uses linear interpolation between two known alleles, or linear
        extrapolation if the allele is at the edge of the marker range.
        """
        # Collect alleles with known ladder bp
        known_panel_bp: list[float] = []
        known_ladder_bp: list[float] = []

        for a in marker.alleles:
            lbp = marker_allele_to_bp.get((marker.name, a.name))
            if lbp is not None and a.base_pair is not None:
                known_panel_bp.append(a.base_pair)
                known_ladder_bp.append(lbp)

        if len(known_panel_bp) < 2:
            # Not enough points to inter/extrapolate — keep original
            return allele

        # Linear fit: ladder_bp = a * panel_bp + b
        coeffs = np.polyfit(known_panel_bp, known_ladder_bp, 1)
        estimated_bp = float(np.polyval(coeffs, allele.base_pair)) if allele.base_pair else None

        return Allele(
            name=allele.name,
            base_pair=estimated_bp,
            left_bin=allele.left_bin,
            right_bin=allele.right_bin,
        )

    # -- Dunder ----------------------------------------------------------- #

    def __repr__(self) -> str:
        status = "adjusted" if self.adjusted_panel else "failed"
        return f"Ladder({self.path.name}, {status})"
