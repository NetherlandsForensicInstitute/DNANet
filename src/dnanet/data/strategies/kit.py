"""STR Kit definition — captures kit identity and configuration.

Design pattern: **Value Object**
    An STR kit is immutable configuration data: once defined, it never changes
    during a run. Frozen dataclass enforces this.

Note:
    ``num_dyes`` is the single source of truth for the number of fluorescence
    channels. Code that previously used the hard-coded ``NUM_DYES = 5``
    constant should read ``kit.num_dyes`` instead. This allows kits with
    5, 6, or 7 dye channels to work seamlessly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dnanet.core.panel import Panel
from dnanet.data.strategies.size_standard import SizeStandard


@dataclass(frozen=True)
class STRKit:
    """Configuration for a forensic DNA profiling kit.

    Attributes:
        name: Kit identifier (e.g. "PPF6C", "GlobalFiler").
        size_standard: The internal size standard used by this kit.
        panel: The allele/marker panel for this kit.
        num_dyes: Number of fluorescence channels (including size standard).
        hid_dye_mapping: Maps 1-based HID dye indices to 0-based channel rows.
                         Kit-specific because some kits skip dye numbers
                         (e.g. PPF6C skips dye 5, mapping {1:0, 2:1, 3:2, 4:3, 6:4}).
        panel_path: Path to the panel XML file (for reference).
        description: Human-readable description.
    """

    name: str
    size_standard: SizeStandard
    panel: Panel
    num_dyes: int = 5
    hid_dye_mapping: dict[int, int] = field(default_factory=lambda: {
        1: 0, 2: 1, 3: 2, 4: 3, 6: 4,
    })
    panel_path: Path | None = None
    description: str = ""
    hid_file_data_columns_analyzed: list[str] | None = None
    hid_file_data_columns_raw: list[str] | None = None

    def dye_row_from_hid_index(self, hid_index: int) -> int:
        """Convert a 1-based HID dye index to a 0-based channel row.

        Raises:
            ValueError: If the HID index doesn't map to a known dye for this kit.
        """
        row = self.hid_dye_mapping.get(hid_index)
        if row is None:
            raise ValueError(
                f"Unknown HID dye index {hid_index} for kit {self.name}. "
                f"Valid indices: {list(self.hid_dye_mapping.keys())}"
            )
        return row
