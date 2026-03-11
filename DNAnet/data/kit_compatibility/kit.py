"""
Kit definitions: capture the user-facing choice of kit (name) and all
kit-specific settings such as size standard and panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from DNAnet.data.data_models.dna_models import Panel
from DNAnet.data.kit_compatibility.lane_standards import InternalSizeStandard


@dataclass(frozen=True)
class Kit:
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
    panel_path: Optional[Path] = None
    panel: Optional[Panel] = None
    markers: Optional[Sequence[str]] = None
    description: Optional[str] = None



# Example pre-defined kits; extend as needed.
POWER_PLEX_FUSION_6C_PANEL_PATH = Path("resources/data/SGPanel_PPF6C.xml")

POWER_PLEX_FUSION_6C_KIT = Kit(
    name="PPF6C",
    size_standard=InternalSizeStandard.WEN_ILS,
    panel_path=POWER_PLEX_FUSION_6C_PANEL_PATH,
    panel=Panel(POWER_PLEX_FUSION_6C_PANEL_PATH),  # fill in when you load the panel
    markers=None,  # fill in with marker names for quick validation
    description="ProvedIt dataset kit using WEN_ILS size standard.",
)

POWERPLEX_Y23 = Kit(
    name="POWERPLEX_Y23",
    size_standard=InternalSizeStandard.WEN_ILS,
    panel_path=None,
    panel=None,
    markers=None,
    description="POWERPLEX_Y23 kit using WEN_ILS size standard.",
)




GLOBALFILER_PANEL_PATH = Path("resources/data/SGPanel_Globalfiler_Panel.xml")

GLOBALFILER_KIT = Kit(
    name="GlobalFiler",
    size_standard=InternalSizeStandard.GENESCAN_600_LIZ,
    panel_path=GLOBALFILER_PANEL_PATH,
    panel=Panel(GLOBALFILER_PANEL_PATH),
    markers=None,  # fill in with marker names for quick validation
    description="GlobalFiler kit using GENESCAN_600_LIZ size standard.",
)
