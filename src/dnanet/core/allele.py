"""Allele domain model.

Design pattern: **Value Object**
    An Allele is a *value object* — it is defined entirely by its data, has no
    identity beyond its fields, and is ideally treated as immutable after
    creation. We use `@dataclass(frozen=True, slots=True)` to enforce this:

    - `frozen=True`: prevents accidental mutation after construction. If you
      need a modified copy, use `dataclasses.replace(allele, height=500)`.
    - `slots=True`: uses `__slots__` instead of `__dict__`, saving memory when
      you have thousands of Allele instances (common in a full panel).

    Compared to the original, we:
    1. Removed the `bin` property that returned a numpy array — mixing numpy
       into a pure domain model couples it to a heavy dependency. Bin ranges
       are computed where needed (in preprocessing).
    2. Made serialization explicit with `to_dict` / `from_dict` (preserved
       for backward-compatible JSON I/O).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Allele:
    """A single measured allele within a DNA marker.

    Attributes:
        name: Allele designation (e.g. "12", "13.2", "X").
        base_pair: Size in base pairs (the peak's center position).
        left_bin: Distance (in bp) from `base_pair` to the left bin edge.
        right_bin: Distance (in bp) from `base_pair` to the right bin edge.
        height: Peak height in Relative Fluorescence Units (RFU).
    """

    name: str
    base_pair: float | None = field(default=None, compare=False)
    left_bin: float | None = field(default=None, compare=False)
    right_bin: float | None = field(default=None, compare=False)
    height: float | None = field(default=None, compare=False)

    # -- Serialization (Value Object pattern: reconstruct from plain data) --

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Allele:
        """Deserialize from a plain dictionary."""
        return cls(
            name=data["name"],
            base_pair=data.get("base_pair"),
            left_bin=data.get("left_bin"),
            right_bin=data.get("right_bin"),
            height=data.get("height"),
        )
