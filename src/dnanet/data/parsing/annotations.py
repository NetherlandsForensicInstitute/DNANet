"""Annotation file parsing for manually called alleles.

Reads tab/comma/semicolon-delimited text files produced by forensic analysts,
extracting allele calls and peak heights for a specific sample.
"""

from __future__ import annotations

import csv
import os
from itertools import groupby
from typing import Iterable

from loguru import logger

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.panel import Panel
from dnanet.core.types import PathLike


def parse_called_alleles(
    annotation_file: PathLike,
    panel: Panel,
    sample_name: str,
) -> list[Marker] | None:
    """Parse manually called alleles from an annotation text file.

    The annotation file may contain calls for multiple samples. This function
    finds the rows matching ``sample_name`` and returns the parsed markers.

    Args:
        annotation_file: Path to the annotation CSV/TSV/TXT file.
        panel: Panel used to look up allele bin positions.
        sample_name: The sample identifier to extract.

    Returns:
        List of Markers with their alleles, or ``None`` if not found.
    """
    if os.stat(annotation_file).st_size == 0:
        logger.debug("Empty annotation file: {}", annotation_file)
        return None

    with open(annotation_file, "r") as f:
        try:
            delimiter, allele_cols, height_cols = _parse_csv_header(f)
        except TypeError as e:
            logger.debug("Could not parse header of {}: {}", annotation_file, e)
            return None

        reader = csv.reader(f, delimiter=delimiter)
        for sample, rows in groupby(reader, lambda row: row[0]):
            if sample == sample_name:
                return _parse_sample_annotations(panel, rows, allele_cols, height_cols)

    return None


def _parse_sample_annotations(
    panel: Panel,
    rows,
    allele_cols: Iterable[int],
    height_cols: Iterable[int],
) -> list[Marker]:
    """Parse annotation rows for a single sample into Markers."""
    markers: list[Marker] = []

    for row in rows:
        marker_name = row[1]
        dye_row = panel.get_dye_row(marker_name)
        if dye_row is None:
            continue

        alleles = frozenset(
            Allele(
                name=allele_name,
                base_pair=bp,
                left_bin=bp - left,
                right_bin=right - bp,
                height=float(row[height_col]),
            )
            for allele_col, height_col in zip(allele_cols, height_cols)
            if (allele_name := row[allele_col].strip("OB_"))
            for bp, left, right in [
                panel.get_allele_basepair_and_bins(marker_name, allele_name)
            ]
        )
        markers.append(Marker(name=marker_name, dye_row=dye_row, alleles=alleles))

    return markers


def _parse_csv_header(file) -> tuple[str, list[int], list[int]]:
    """Detect delimiter and locate Allele/Height columns in the header."""
    header = next(file)

    for delimiter in [",", ";", "\t"]:
        columns = header.split(delimiter)
        allele_cols = [i for i, col in enumerate(columns) if col.startswith("Allele")]
        if allele_cols:
            height_cols = [i for i, col in enumerate(columns) if col.startswith("Height")]
            return delimiter, allele_cols, height_cols

    raise TypeError(f"No valid delimiter found in header: {header!r}")
