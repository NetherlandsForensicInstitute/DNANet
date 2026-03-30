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
from itertools import groupby
import os
import re
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Tuple

from loguru import logger

from dnanet.core.allele import Allele
from dnanet.core.annotation import AlleleAnnotation, Annotation
from dnanet.core.marker import Marker
from dnanet.core.types import PathLike
from dnanet.data.strategies.dataset import DatasetStrategy, FileCategory


# R&D filename pattern: digit + letter + digit (e.g. "1A2")
_RD_PREFIX_RE = re.compile(r"^\d[A-F]\d")


class NFIRnDStrategy(DatasetStrategy):
    """Strategy for the NFI R&D mixture dataset."""
    
    READ_ANNOTATION_HEIGHTS: bool = False
    
    @classmethod
    def collect_dataset_files(cls, root_path: PathLike, **kwargs) -> Generator[Tuple[Path, Annotation | None, Path | None]]:
        """Collect the dataset files for this specific dataset strategy. 
        
        Creates a generator that yields paths to HIDImage's, Annotations (optional),
        and corresponding Ladder file (optional).

        Args:
            root_path: The path to the root of this dataset
        """
        path = Path(root_path)
        csv_files = list(path.rglob('*.csv'))

        hid_to_annotation_file_pattern = r'.*hid_to_annotation.*'
        hid_to_annotation_path = None
        hid_to_ladder_pattern = r'.*best_ladder_paths.*'
        hid_to_ladder_path = None

        analysis_treshold_type: str = kwargs.get('analysis_treshold_type', 'DTH')
        logger.info(f"Using treshold type: {analysis_treshold_type}")

        for csv_file in csv_files:
            if re.match(hid_to_annotation_file_pattern, csv_file.name):
                hid_to_annotation_path = csv_file
            if re.match(hid_to_ladder_pattern, csv_file.name):
                hid_to_ladder_path = csv_file
        if hid_to_annotation_path is None or hid_to_ladder_path is None:
            raise ValueError(
                'Path does not contain the neccessary mapping files (annotation & ladder)'
            )

        # Allele Report for Annotations
        annotation_txt_files = list(path.rglob('*AlleleReport.txt'))
        annotation_name_to_annotation: Dict[str, Annotation] = {}
        for txt_file in annotation_txt_files:
            _annotation = cls.parse_annotations(txt_file)

            if _annotation:
                annotation_name_to_annotation.update(_annotation)

        # HID to Annotation mapping
        hta_header, hta_values = cls._read_csv_file(hid_to_annotation_path)
        analysis_treshold_type_column = [
            i for i, head in enumerate(hta_header) if analysis_treshold_type in head
        ]
        if len(analysis_treshold_type_column) != 1:
            raise RuntimeError(
                f'Could not infer the analysis treshold type column for annotation mapping: {hta_header}'
            )
        hid_to_annotation = dict([
            (v[0].replace('.hid', ''), annotation_name_to_annotation.get(v[analysis_treshold_type_column[0]]))
            for v in hta_values
        ])


        # Hid to Ladder mapping
        _, htl_values = cls._read_csv_file(hid_to_ladder_path)
        hid_to_ladder = {hid: Path(ladder) for hid, ladder in htl_values}

        hid_files = list(path.rglob('*.hid'))
        hid_file_samples = list(filter(lambda x: cls.categorize_file(x.name) == 'sample', hid_files))

        for hid_file in hid_file_samples:
            yield (
                hid_file,
                hid_to_annotation.get(str(hid_file.stem)),
                hid_to_ladder.get(hid_file.stem)
            )
        

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
    def _get_scaling_strategy():
        """Get the active panel from the strategy registry."""
        from dnanet.data.strategies.registry import StrategyRegistry
        return StrategyRegistry.get_scaling_strategy()

    @classmethod
    def parse_annotations(
        cls,
        annotation_source: PathLike,
    ) -> Dict[str, Annotation]:
        """Parse manually called alleles from an annotation text file.

        The annotation file may contain calls for multiple samples. This function
        finds the rows matching ``sample_name`` and returns the parsed markers.

        Args:
            annotation_file: Path to the annotation CSV/TSV/TXT file.

        Returns:
            List of Markers with their alleles, or ``None`` if not found.
        """
        if os.stat(annotation_source).st_size == 0:
            logger.debug("Empty annotation file: {}", annotation_source)
            raise RuntimeError("Annotations file is emtpy")

        annotation_mapping: Dict[str, Annotation] = {}
        with open(annotation_source, "r") as f:
            try:
                delimiter, allele_cols, height_cols = cls._parse_csv_header(f)
            except TypeError as e:
                logger.debug("Could not parse header of {}: {}", annotation_source, e)
                raise e

            reader = csv.reader(f, delimiter=delimiter)
            for sample, rows in groupby(reader, lambda row: row[0]):
                sample_annotation = cls._parse_sample_annotations(rows, allele_cols, height_cols)
                annotation_mapping[sample] = AlleleAnnotation(sample_annotation)

        return annotation_mapping

    @classmethod
    def _parse_sample_annotations(
        cls,
        rows,
        allele_cols: Iterable[int],
        height_cols: Iterable[int],
    ) -> list[Marker]:
        """Parse annotation rows for a single sample into Markers."""
        markers: list[Marker] = []

        marker_2_dye = cls._get_scaling_strategy().marker_name_to_dye_idx()
        for row in rows:
            marker_name = row[1]
            dye_row = marker_2_dye[marker_name]
            if dye_row is None:
                continue

            alleles = frozenset(
                Allele(
                    name=allele_name,
                    height=float(row[height_col]) if cls.READ_ANNOTATION_HEIGHTS else None,
                )
                for allele_col, height_col in zip(allele_cols, height_cols)
                if (allele_name := row[allele_col].strip("OB_"))
            )
            markers.append(Marker(name=marker_name, dye_row=dye_row, alleles=alleles))

        return markers

    @classmethod
    def _parse_csv_header(cls, file) -> tuple[str, list[int], list[int]]:
        """Detect delimiter and locate Allele/Height columns in the header."""
        header = next(file)

        for delimiter in [",", ";", "\t"]:
            columns = header.split(delimiter)
            allele_cols = [i for i, col in enumerate(columns) if col.startswith("Allele")]
            if allele_cols:
                height_cols = [i for i, col in enumerate(columns) if col.startswith("Height")]
                return delimiter, allele_cols, height_cols

        raise TypeError(f"No valid delimiter found in header: {header!r}")
    
    @classmethod
    def _read_csv_file(cls, csv_file: str | Path) -> Tuple[List[str], List[List[str]]]:
        """Read a csv file with a header row.

        Args:
            csv_file: The path to the csv file to be read

        Returns:
            Tuple with the header row and (filtered on empty) value rows.
        """
        with open(csv_file, 'r') as htap:
            reader = csv.reader(htap)
            rows = list(reader)
        headers, values = rows[0], filter(lambda r: len(r) > 0, rows[1:])
        return headers, list(values)