"""Result data structures for inference output.

Design pattern: **Value Object**
    All result classes are frozen dataclasses — immutable, serializable,
    and hashable where applicable. This makes them safe to pass between
    pipeline stages and easy to serialize to JSON.
"""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from dataclasses import field, asdict, dataclass


@dataclass(frozen=True, slots=True)
class AlleleCall:
    """A single called allele with metadata.

    Attributes:
        name: Allele designation (e.g. "12", "13.2", "X").
        base_pair: Calibrated base-pair position.
        height: Peak height in RFU (from the original signal).
        confidence: Model confidence for this allele call (0.0–1.0).
    """

    name: str
    base_pair: float
    height: float
    confidence: float


@dataclass(frozen=True, slots=True)
class MarkerResult:
    """Called alleles for a single marker (locus).

    Attributes:
        name: Marker designation (e.g. "D3S1358", "AMEL").
        dye_row: 0-based dye channel index.
        alleles: Called alleles at this marker.
    """

    name: str
    dye_row: int
    alleles: list[AlleleCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'dye_row': self.dye_row,
            'alleles': [asdict(a) for a in self.alleles],
        }


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Inference results for a single DNA profile.

    Attributes:
        sample: Sample identifier (derived from HID filename).
        hid_path: Path to the source HID file.
        ladder_path: Path to the ladder HID file used for panel adjustment.
        markers: Called markers with alleles.
        warnings: Any warnings encountered during inference.
        signal: Raw signal data (num_dyes, scanpoints), if requested.
        prediction: Scanpoint-level prediction probabilities
                    (num_dyes, scanpoints), if requested.
        scaler: Base-pair calibration array (scanpoints,), if requested.
    """

    sample: str
    hid_path: str
    ladder_path: str | None = None
    markers: list[MarkerResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    signal: list[list[float]] | None = None
    prediction: list[list[float]] | None = None
    scaler: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            'sample': self.sample,
            'hid_path': self.hid_path,
            'markers': [m.to_dict() for m in self.markers],
        }
        if self.ladder_path is not None:
            result['ladder_path'] = self.ladder_path
        if self.warnings:
            result['warnings'] = self.warnings
        return result

    @property
    def allele_count(self) -> int:
        return sum(len(m.alleles) for m in self.markers)

    @property
    def marker_count(self) -> int:
        return len(self.markers)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Complete inference results across all profiles.

    Attributes:
        checkpoint: Path to the checkpoint used.
        kit: Kit name from the scaling strategy.
        profiles: Results for each profile.
        timing: Per-profile timing info (if collected).
    """

    checkpoint: str
    kit: str
    profiles: list[ProfileResult] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            'checkpoint': self.checkpoint,
            'kit': self.kit,
            'total_profiles': len(self.profiles),
            'total_alleles': sum(p.allele_count for p in self.profiles),
            'profiles': [p.to_dict() for p in self.profiles],
        }
        if self.timing:
            result['timing'] = self.timing
        return result

    def save_json(self, path: str | Path) -> Path:
        """Save results to a JSON file.

        Args:
            path: Output file path.

        Returns:
            The path the results were saved to.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2))
        return output_path

    @property
    def total_markers_called(self) -> int:
        return sum(p.marker_count for p in self.profiles)

    @property
    def total_alleles(self) -> int:
        return sum(p.allele_count for p in self.profiles)

    @property
    def total_profiles(self) -> int:
        return len(self.profiles)


def save_epg_plot(
    signal: list[list[float]],
    prediction: list[list[float]] | None = None,
    *,
    title: str | None = None,
    output_path: str | Path,
) -> Path:
    """Save an EPG profile plot to disk.

    Args:
        signal: (num_dyes, scanpoints) fluorescence signal.
        prediction: (num_dyes, scanpoints) prediction overlay.
        title: Plot title.
        output_path: Where to save the PNG.

    Returns:
        The path the plot was saved to.
    """
    import numpy as np

    from dnanet.evaluation.visualization import plot_profile

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    signal_arr = np.array(signal, dtype=np.float32)
    prediction_arr = np.array(prediction, dtype=np.float32) if prediction else None

    fig = plot_profile(
        signal=signal_arr,
        prediction=prediction_arr,
        title=title,
    )
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    import matplotlib.pyplot as plt

    plt.close(fig)
    return output_path
