"""ProvedIt dataset strategy.

Handles the ProvedIt dataset conventions:
- File naming: ``{well}_{sample-description}_{injection-time}.hid``
  e.g. ``B03_RD14-0003-34d1-0.5IP-Q0.75ng_05sec.hid``
  The sample description encodes contributor count, ratio, quality, etc.
- Ladders: filenames containing ``Ladder-GF`` (GlobalFiler ladder)
- Annotations: Single XLSX genotype file for the entire dataset
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from dnanet.core.marker import Marker
from dnanet.data.strategies.dataset import FileCategory, DatasetStrategy


# ProvedIt ladder pattern
_LADDER_RE = re.compile(r"Ladder", re.IGNORECASE)


class ProvedItStrategy(DatasetStrategy):
    """Strategy for the ProvedIt dataset (GlobalFiler kit)."""

    @classmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        """Classify based on ProvedIt naming conventions.

        - Ladders contain ``Ladder`` (e.g. ``B03_Ladder-GF_02.5sec.hid``)
        - Negative controls contain ``neg`` or ``NTC``
        - Everything else is a sample
        """
        stem = Path(file_name).stem

        if _LADDER_RE.search(stem):
            return "ladder"
        lower = stem.lower()
        if "neg" in lower or "ntc" in lower:
            return "control"
        return "sample"

    @classmethod
    def get_number_of_contributors(cls, file_name: str) -> int:
        """Extract contributor count from ProvedIt naming.

        ProvedIt encodes contributors in the sample description,
        e.g. ``RD14-0003-34d1`` → 2 contributors (digits 3 and 4 in ``34``).
        The exact parsing depends on the specific ProvedIt naming convention.

        Returns:
            String like ``"2p"`` or ``None`` if not determinable.
        """
        stem = Path(file_name).stem
        # Example filename:
        # A02_RD14-0003-31_32-1;1-M2c-0.03GF-Q2.0_01.5sec
        # We want to extract the 31_32 here to conclude 2 contributors (with 1:1 mixture prop.)
        parts = stem.split("-")
        if len(parts) < 2:
            raise ValueError(f'Could not extract NoC from {file_name=}')

        # We check each of the file's parts
        for p in parts:
            # And if it matches the pattern of:
            # contributor 1 + (underscore + contributer{i}) * i_contributors (i ranging from 1 to 4)
            if re.match(rf'(\d{1,2})(_\d{1,2})+', p):
                # We return the number of distinct contributor ID's
                return len(p.split('_'))
        raise ValueError(f'Could not extract NoC from {file_name=}')

    @classmethod
    def get_sample_id(cls, file_name: str) -> str:
        """Extract sample identifier from ProvedIt filename.

        Returns the description part (between well and injection time),
        e.g. ``B03_RD14-0003-34d1-0.5IP-Q0.75ng_05sec.hid``
        → ``"RD14-0003-34d1-0.5IP-Q0.75ng"``
        """
        stem = Path(file_name).stem
        parts = stem.split("_")
        if len(parts) >= 3:
            # Join everything between well and injection time
            return "_".join(parts[1:-1])
        return stem

    @classmethod
    def find_annotation_file(
        cls, sample_path: Path, annotation_dir: Path
    ) -> Path | None:
        """Find the genotype XLSX file for ProvedIt.

        ProvedIt uses a single XLSX file for all genotype data.
        """
        xlsx_files = list(annotation_dir.glob("*.xlsx"))
        return xlsx_files[0] if xlsx_files else None

    @classmethod
    def parse_annotations(
        cls,
        annotation_source: Path,
        sample_name: str,
    ) -> list[Marker]:
        """Load called alleles from the ProvedIt XLSX genotype file.

        The ProvedIt XLSX contains genotype data for all samples.
        Each row typically maps sample→marker→alleles.

        Note:
            This is a stub — full XLSX parsing will be implemented when
            ProvedIt integration is tested end-to-end.
        """
        logger.warning(
            "ProvedIt XLSX annotation loading not yet fully implemented "
            "for sample {}", sample_name
        )
        return []

    @classmethod
    def find_ladder_for_sample(
        cls, sample_path: Path, ladder_mapping: dict[str, Path] | None = None
    ) -> Path | None:
        """Find the ladder for a ProvedIt sample.

        ProvedIt ladders share the same injection directory and well prefix.
        E.g. sample ``B03_RD14-...`` uses ladder ``B03_Ladder-GF_...`` from
        the same run.
        """
        stem = sample_path.stem
        if ladder_mapping and stem in ladder_mapping:
            return ladder_mapping[stem]

        # Extract well prefix (e.g. "B03")
        well = stem.split("_")[0] if "_" in stem else None
        if well is None:
            return None

        # Look in same directory for a ladder with matching well
        parent = sample_path.parent
        for f in parent.glob("*.hid"):
            if "ladder" in f.name.lower() and f.stem.startswith(well):
                return f

        # Fallback: any ladder in the directory
        ladders = [f for f in parent.glob("*.hid") if "ladder" in f.name.lower()]
        return ladders[0] if ladders else None

    @staticmethod
    def get_annotation_classes() -> list[str]:
        return ["noise", "allele"]
