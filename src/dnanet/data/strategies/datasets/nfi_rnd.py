"""NFI R&D dataset strategy.

Handles the NFI Research & Development dataset conventions:
- File naming: ``{NOC}{batch}{contributors}_{well}_{rep}.hid``
  e.g. ``1A2_A01_01.hid`` means NOC=2, batch=A, contributors=1A2
- Ladders: filenames starting with ``ladder_`` (case-insensitive)
- Controls: ``blanco``, ``pocon``, ``controle``, or filenames starting with ``A``
- Annotations: Tab-separated TXT files (AlleleReport format), one per injection
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from loguru import logger

from dnanet.core.marker import Marker
from dnanet.data.strategies.dataset import DatasetStrategy, FileCategory


# R&D filename pattern: digit + letter + digit (e.g. "1A2")
_RD_PREFIX_RE = re.compile(r"^\d[A-F]\d")


class NFIRnDStrategy(DatasetStrategy):
    """Strategy for the NFI R&D mixture dataset."""

    @classmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        """Classify based on NFI R&D naming conventions.

        - Ladders start with ``ladder_``
        - Controls contain ``blanco``, ``pocon``, ``controle``, or start with ``A``
        - Samples match the ``\\d[A-F]\\d`` prefix pattern
        """
        lower = file_name.lower()
        stem = Path(file_name).stem

        if lower.startswith("ladder") or "ladder" in lower:
            return "ladder"
        if "blanco" in lower or "pocon" in lower or "controle" in lower:
            return "control"
        if file_name.startswith("A"):
            return "control"
        if _RD_PREFIX_RE.match(stem):
            return "sample"
        return "unknown"

    @classmethod
    def get_contributors(cls, file_name: str) -> str | None:
        """Extract NOC from R&D filename: ``1A2`` → ``"2p"``."""
        stem = Path(file_name).stem
        if _RD_PREFIX_RE.match(stem):
            return f"{stem[2]}p"
        return None

    @classmethod
    def get_sample_id(cls, file_name: str) -> str:
        """Extract profile prefix: ``1A2_A01_01.hid`` → ``"1A2"``."""
        stem = Path(file_name).stem
        if _RD_PREFIX_RE.match(stem):
            return stem.split("_")[0]
        raise ValueError(f"Cannot extract sample ID from R&D filename: {file_name}")

    @classmethod
    def find_annotation_file(
        cls, sample_path: Path, annotation_dir: Path
    ) -> Path | None:
        """Find the TXT annotation file that contains this sample.

        R&D annotations are stored as ``{injection_name}.txt`` files,
        typically in the same directory structure as the HID files.
        """
        # The annotation file lives alongside the sample's injection directory
        injection_dir = sample_path.parent
        txt_files = list(annotation_dir.glob("*.txt"))
        if not txt_files:
            txt_files = list(injection_dir.glob("*.txt"))
        return txt_files[0] if txt_files else None

    @classmethod
    def load_annotations(
        cls,
        annotation_source: Path,
        sample_name: str,
    ) -> list[Marker]:
        """Load called alleles from a tab-separated AlleleReport TXT file.

        This delegates to the annotation parser for the actual parsing.
        The R&D format has columns: Sample Name, Marker, Allele 1, Allele 2, ...
        """
        # Defer to the existing annotation parser
        from dnanet.data.parsing.annotations import parse_called_alleles

        try:
            panel = cls._get_active_panel()
        except RuntimeError:
            logger.warning("No panel configured; cannot load annotations")
            return []

        return parse_called_alleles(annotation_source, panel, sample_name)

    @classmethod
    def find_ladder_for_sample(
        cls, sample_path: Path, ladder_mapping: dict[str, Path] | None = None
    ) -> Path | None:
        """Find the ladder for an R&D sample.

        Uses the pre-built mapping (from ``best_ladder_paths_DTH.csv``)
        if provided, otherwise looks in the same injection directory.
        """
        stem = sample_path.stem
        if ladder_mapping and stem in ladder_mapping:
            return ladder_mapping[stem]

        # Fallback: look for ladder files in the same directory
        parent = sample_path.parent
        ladders = [f for f in parent.glob("*.hid") if "ladder" in f.name.lower()]
        return ladders[0] if ladders else None

    @staticmethod
    def load_ladder_mapping(csv_path: Path) -> dict[str, Path]:
        """Load a sample→ladder mapping from ``best_ladder_paths_DTH.csv``.

        Returns:
            Dict mapping sample stems to ladder paths.
        """
        mapping = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row["image_path"]] = Path(row["ladder_path"])
        return mapping

    @staticmethod
    def _get_active_panel():
        """Get the active panel from the strategy registry."""
        from dnanet.data.strategies.registry import StrategyRegistry
        return StrategyRegistry.get_scaling_strategy().panel

    @staticmethod
    def get_annotation_classes() -> list[str]:
        return ["noise", "allele"]
