"""Evaluation utility functions.

Provides per-locus RFU thresholds and allele flattening for
allele-level metric computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from loguru import logger


if TYPE_CHECKING:
    from dnanet.core.marker import Marker


def flatten_markers_to_allele_names(
    markers: Sequence[Marker],
    *,
    locus: str | None = None,
) -> frozenset[str]:
    """Flatten a sequence of Markers to a set of ``"MarkerName_AlleleName"`` strings.

    Optionally filters by locus name.

    Example::

        >>> markers = [Marker("D5S818", 0, frozenset(Allele("13", height=500), Allele("15", height=200)))]
        >>> flatten_markers_to_allele_names(markers)
        frozenset({'D5S818_13', 'D5S818_15'})

    Args:
        markers: Sequence of Marker objects to flatten.
        locus: If provided, only include alleles from this locus.

    Returns:
        Frozen set of ``"marker_allele"`` strings.

    Raises:
        ValueError: If an RFU threshold is set but allele height is None.
    """
    result: list[str] = []
    for marker in markers:
        if locus and marker.name != locus:
            continue

        for allele in marker.alleles:
            result.append(f"{marker.name}_{allele.name}")

    result_set = frozenset(result)
    if len(result_set) != len(result):
        logger.warning("Non-unique locus-allele combinations found")
    return result_set
