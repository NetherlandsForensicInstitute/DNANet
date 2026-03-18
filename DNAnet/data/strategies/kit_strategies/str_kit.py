"""
Kit definitions: capture the user-facing choice of kit (name) and all
kit-specific settings such as size standard and panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from DNAnet.data.data_models.dna_models import Panel
from DNAnet.data.strategies.kit_strategies.internal_size_standard import InternalSizeStandard


@dataclass(frozen=True)
class STRKit:
    """
    Describes a DNA profiling kit and its key configuration.

    Attributes:
        name: Human-readable kit name/identifier (e.g. "provedit", "nfi", "globalfiler").
        size_standard: Internal size standard used by this kit.
        panel_path: Optional path to the panel file describing markers/alleles.
        panel: Panel object describing markers/alleles.
        markers: Optional list of marker names used by this kit (for quick checks or validation).
        description: Optional free-text description.
    """

    name: str
    size_standard: InternalSizeStandard
    panel: Panel
    num_dyes: int = 6
    panel_path: Optional[Path] = None
    markers: Optional[Sequence[str]] = None
    description: Optional[str] = None


