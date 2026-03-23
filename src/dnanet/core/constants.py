"""Domain constants for forensic DNA analysis.

Design pattern: **Enum as a Value Object**
    Instead of the original dictionary-of-dictionaries (`LABEL_CATEGORIES`),
    we model each label category as an `enum.Enum` member. This gives us:
    - Type safety: you can't pass an arbitrary string where a label is expected.
    - Discoverability: IDE autocomplete shows all valid labels.
    - Single source of truth: color, index, and display name live together.

Note:
    Kit-specific constants (dye count, dye-index mapping) live on ``STRKit``
    and ``ScalingStrategy``, NOT here. This module only holds constants that
    are truly universal across all kits and datasets.
"""

from enum import Enum, unique


@unique
class LabelCategory(Enum):
    """Peak label categories used in interactive annotation and classification.

    Each member stores its display metadata as a tuple:
        (color, alpha, shortcut_key)
    """

    UNLABELED = ("gray", 0.2, "0")
    ALLELE = ("green", 0.4, "1")
    STUTTER = ("yellow", 0.4, "2")
    PULL_UP = ("blue", 0.4, "3")
    BLEED_THROUGH = ("red", 0.4, "4")
    SPIKE = ("cyan", 0.4, "5")
    DYE_BLOB = ("purple", 0.4, "6")
    ARTEFACT = ("pink", 0.4, "7")
    UNCLEAR = ("tab:orange", 0.4, "8")
    SHOULDER = ("tab:olive", 0.4, "9")
    FOREIGN_DNA = ("tab:brown", 0.4, "f")
    OVERLOADING_ARTEFACT = ("lime", 0.4, "o")

    def __init__(self, color: str, alpha: float, shortcut: str) -> None:
        self.color = color
        self.alpha = alpha
        self.shortcut = shortcut

    @property
    def label_name(self) -> str:
        """Human-readable label name, e.g. 'Allele', 'PullUp'.

        Matches the original `pyval` strings for backward compatibility.
        """
        # UNLABELED has no label name (was `None` in the original)
        if self is LabelCategory.UNLABELED:
            return ""
        # Convert PULL_UP -> PullUp, BLEED_THROUGH -> BleedThrough, etc.
        result = "".join(part.capitalize() for part in self.name.split("_"))
        # Preserve "DNA" acronym (capitalize() lowercases it to "Dna")
        return result.replace("Dna", "DNA")

    @property
    def display_name(self) -> str:
        """Human-readable name with spaces, e.g. 'Pull Up', 'Foreign DNA'.

        Used in radio button labels for the interactive label tool.
        """
        _NAMES: dict[str, str] = {
            "UNLABELED": "Unlabeled",
            "ALLELE": "Allele",
            "STUTTER": "Stutter",
            "PULL_UP": "Pull Up",
            "BLEED_THROUGH": "Bleed Through",
            "SPIKE": "Spike",
            "DYE_BLOB": "Dye Blob",
            "ARTEFACT": "Artefact",
            "UNCLEAR": "Unclear",
            "SHOULDER": "Shoulder",
            "FOREIGN_DNA": "Foreign DNA",
            "OVERLOADING_ARTEFACT": "Overloading artefact",
        }
        return _NAMES[self.name]

    @property
    def color_display(self) -> str:
        """Human-readable color name, e.g. 'Green', 'Orange'."""
        return self.color.replace("tab:", "").capitalize()

    @property
    def radio_label(self) -> str:
        """Full radio-button label, e.g. ``'Allele (Green - 1)'``."""
        return f"{self.display_name} ({self.color_display} - {self.shortcut.upper()})"

    @classmethod
    def from_index(cls, index: int) -> "LabelCategory":
        """Look up a category by its integer position (0-based)."""
        return list(cls)[index]

    @classmethod
    def label_names(cls) -> list[str]:
        """Return all label names in order (matches original LABEL_CATEGORIES_STR)."""
        return [member.label_name for member in cls]


# Default signal length (number of scan points per dye channel)
DEFAULT_SIGNAL_LENGTH: int = 4096

# Label tool format version
LABELTOOL_VERSION: str = "1.0"

# Markers that are not autosomal (used for filtering in allele calling)
NON_AUTOSOMAL_PREFIXES: tuple[str, ...] = ("DYS",)
NON_AUTOSOMAL_MARKERS: frozenset[str] = frozenset({"AMEL"})
