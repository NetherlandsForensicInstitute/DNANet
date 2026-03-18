"""Domain constants for forensic DNA analysis.

Design pattern: **Enum as a Value Object**
    Instead of the original dictionary-of-dictionaries (`LABEL_CATEGORIES`),
    we model each label category as an `enum.Enum` member. This gives us:
    - Type safety: you can't pass an arbitrary string where a label is expected.
    - Discoverability: IDE autocomplete shows all valid labels.
    - Single source of truth: color, index, and display name live together.

    The `DyeIndex` enum similarly replaces the hard-coded `{1: 0, 2: 1, ...}`
    mapping that was buried inside `Panel._parse_panel`.
"""

from enum import Enum, IntEnum, unique


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
        return "".join(part.capitalize() for part in self.name.split("_"))

    @classmethod
    def from_index(cls, index: int) -> "LabelCategory":
        """Look up a category by its integer position (0-based)."""
        return list(cls)[index]

    @classmethod
    def label_names(cls) -> list[str]:
        """Return all label names in order (matches original LABEL_CATEGORIES_STR)."""
        return [member.label_name for member in cls]


@unique
class DyeIndex(IntEnum):
    """Mapping from 1-based HID dye indices to 0-based channel rows.

    The HID file format uses 1-based dye numbering with a gap (no dye 5).
    This enum makes that mapping explicit instead of hiding it in a dict
    literal inside the panel parser.
    """

    BLUE = 0       # HID dye 1
    GREEN = 1      # HID dye 2
    YELLOW = 2     # HID dye 3
    RED = 3        # HID dye 4
    ORANGE = 4     # HID dye 6 (dye 5 is skipped in PPF6C kit)

    @classmethod
    def from_hid_index(cls, hid_index: int) -> "DyeIndex":
        """Convert a 1-based HID dye index to the 0-based channel row.

        Args:
            hid_index: The dye index as stored in the HID/panel file (1-based).

        Returns:
            The corresponding DyeIndex enum member.

        Raises:
            ValueError: If the HID index doesn't map to a known dye.
        """
        mapping = {1: cls.BLUE, 2: cls.GREEN, 3: cls.YELLOW, 4: cls.RED, 6: cls.ORANGE}
        if hid_index not in mapping:
            raise ValueError(
                f"Unknown HID dye index {hid_index}. "
                f"Valid indices: {list(mapping.keys())}"
            )
        return mapping[hid_index]


# Number of fluorescence channels (dyes) in a standard EPG
NUM_DYES: int = 5

# Default signal length (number of scan points per dye channel)
DEFAULT_SIGNAL_LENGTH: int = 4096

# Markers that are not autosomal (used for filtering in allele calling)
NON_AUTOSOMAL_PREFIXES: tuple[str, ...] = ("DYS",)
NON_AUTOSOMAL_MARKERS: frozenset[str] = frozenset({"AMEL"})
