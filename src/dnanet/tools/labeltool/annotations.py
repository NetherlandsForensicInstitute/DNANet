"""Annotation persistence layer — Repository pattern for CSV I/O.

Design pattern: **Repository**
    ``AnnotationStore`` encapsulates all CSV read/write logic so the label
    tool and analysis scripts never deal with file formats directly.
"""

from __future__ import annotations

import csv
from typing import Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from loguru import logger

from dnanet.core.constants import LabelCategory


# Current annotation format version
LABELTOOL_VERSION = "1.0"

# CSV column order
HEADER = [
    "user", "date", "profile", "dye", "x0", "x1",
    "peak_idx", "category", "version",
]


class AnnotationStore:
    """CSV-backed repository for scan-point annotations.

    Supports reading from a single CSV or a folder of CSVs, and writing
    back with automatic deduplication (per user+profile).

    Args:
        path: Path to a CSV file or directory of CSVs.

    Example::

        store = AnnotationStore(Path("annotations.csv"))
        spans = store.load_spans(profile="1A2_A01_01")
        # ... user modifies spans ...
        store.save_spans("1A2_A01_01", spans, user="Alice")
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_all(self) -> list[dict[str, Any]]:
        """Read all annotation rows from the store.

        Returns:
            List of dicts, one per annotation row.
        """
        rows: list[dict[str, Any]] = []
        for csv_file in self._csv_files():
            rows.extend(self._read_csv(csv_file))
        return rows

    def load_spans_by_profile(
        self,
        *,
        user: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Load annotations grouped by profile name.

        Args:
            user: If provided, only return annotations from this user.

        Returns:
            Dict mapping profile name to list of span dicts.
        """
        rows = self.read_all()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

        label_lookup = _build_label_lookup()

        for row in rows:
            if user and row.get("user") != user:
                continue

            category = row.get("category", "")
            color, alpha = label_lookup.get(category, ("gray", 0.2))

            try:
                x0 = int(row["x0"])
                x1 = int(row["x1"])
                peak_idx = int(row.get("peak_idx", -1))
            except (ValueError, TypeError):
                logger.warning("Skipping malformed row: {}", row)
                continue

            grouped[row["profile"]].append({
                "annotator": row["user"],
                "artist": None,
                "ax": None,
                "dye": row["dye"],
                "category": category,
                "x0": x0,
                "x1": x1,
                "peak_idx": peak_idx,
                "color": color,
                "alpha": alpha,
            })

        return dict(grouped)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def save_spans(
        self,
        profile_name: str,
        spans: list[dict[str, Any]],
        *,
        user: str,
        dye_names: list[str],
        axes: Any = None,
    ) -> None:
        """Save spans for a profile, replacing previous entries for this user+profile.

        Args:
            profile_name: The sample/profile identifier.
            spans: List of span dicts (must have ax, x0, x1, peak_idx, category).
            user: Annotator identifier.
            dye_names: Ordered dye channel names.
            axes: Matplotlib axes array (used to map span axis to dye index).
        """
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_rows: list[list[str]] = []
        for span in spans:
            if axes is not None:
                import numpy as np
                dye_idx = int(np.where(axes == span["ax"])[0][0])
            else:
                dye_idx = dye_names.index(span.get("dye", dye_names[0]))
            dye_name = dye_names[dye_idx]

            new_rows.append([
                user, date, profile_name, dye_name,
                str(int(span["x0"])),
                str(int(span["x1"])),
                str(int(span.get("peak_idx", -1))),
                span.get("category", ""),
                LABELTOOL_VERSION,
            ])

        # Read existing, filter out same profile+user
        existing: list[list[str]] = []
        if self.path.is_file() and self.path.exists():
            with open(self.path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not (row["profile"] == profile_name and row["user"] == user):
                        existing.append([
                            row["user"], row["date"], row["profile"], row["dye"],
                            row["x0"], row["x1"],
                            row.get("peak_idx", "-1"),
                            row.get("category", ""),
                            row.get("version", "0.1"),
                        ])

        all_rows = existing + new_rows

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(all_rows)

        logger.info(
            "Saved {} annotations for {} (user={}), file={}",
            len(new_rows), profile_name, user, self.path,
        )

    def ensure_file(self) -> None:
        """Create the CSV with headers if it doesn't exist."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(HEADER)
            logger.info("Created new annotation file: {}", self.path)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _csv_files(self) -> list[Path]:
        """Resolve the path to a list of CSV files."""
        if self.path.is_file():
            return [self.path]
        elif self.path.is_dir():
            return sorted(self.path.rglob("*.csv"))
        elif not self.path.exists():
            return []
        return []

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, Any]]:
        """Read a single CSV file into a list of dicts."""
        rows: list[dict[str, Any]] = []
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                rows.extend(reader)
        except PermissionError:
            logger.warning("Permission denied reading {}", path)
        return rows


def _build_label_lookup() -> dict[str, tuple[str, float]]:
    """Build a mapping from label name -> (color, alpha)."""
    lookup: dict[str, tuple[str, float]] = {}
    for cat in LabelCategory:
        lookup[cat.label_name] = (cat.color, cat.alpha)
    # Also map empty string for UNLABELED
    lookup[""] = ("gray", 0.2)
    lookup[None] = ("gray", 0.2)  # type: ignore[index]
    return lookup
