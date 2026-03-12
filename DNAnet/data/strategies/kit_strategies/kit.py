"""
Kit definitions: capture the user-facing choice of kit (name) and all
kit-specific settings such as size standard and panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from DNAnet.data.data_models.dna_models import Panel
from DNAnet.data.strategies.kit_strategies.lane_standards import InternalSizeStandard


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
    panel: Panel
    num_dyes: int = 6
    panel_path: Optional[Path] = None
    markers: Optional[Sequence[str]] = None
    description: Optional[str] = None


_POWER_PLEX_FUSION_6C_PANEL_PATH = Path("resources/kit_panels/SGPanel_PPF6C.xml")
_GLOBALFILER_PANEL_PATH = Path("resources/kit_panels/SGPanel_Globalfiler_Panel.xml")


class KitOptions(Enum):
    PPF6C = "POWER_PLEX_FUSION_6C_KIT"
    GLOBALFILER = "GLOBALFILER_KIT"


def get_standard_kit(kit_name: str) -> Kit:
    if kit_name == KitOptions.PPF6C.name:
        return Kit(
            name="PPF6C",
            size_standard=InternalSizeStandard.WEN_ILS,
            num_dyes=6,
            panel_path=_POWER_PLEX_FUSION_6C_PANEL_PATH,
            panel=Panel(
                _POWER_PLEX_FUSION_6C_PANEL_PATH
            ),  # fill in when you load the panel
            markers=None,  # fill in with marker names for quick validation
            description="ProvedIt dataset kit using WEN_ILS size standard.",
        )

    if kit_name == KitOptions.GLOBALFILER.name:
        return Kit(
            name="GlobalFiler",
            size_standard=InternalSizeStandard.GENESCAN_600_LIZ,
            num_dyes=6,
            panel_path=_GLOBALFILER_PANEL_PATH,
            panel=Panel(_GLOBALFILER_PANEL_PATH),
            markers=None,  # fill in with marker names for quick validation
            description="GlobalFiler kit using GENESCAN_600_LIZ size standard.",
        )

    raise ValueError(f"Kit name is not in the standards: {kit_name}")
