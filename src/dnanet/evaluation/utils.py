"""Evaluation utility functions.

Provides per-locus RFU thresholds and allele flattening for
allele-level metric computation.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from loguru import logger

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker


# RFU detection thresholds per locus from the GlobalFiler kit.
# Keys: marker name. Values: dict with 'low' and 'high' thresholds.
THRESHOLDS_PER_LOCUS: dict[str, dict[str, int]] = {
    "AMEL": {"low": 45, "high": 95},
    "D3S1358": {"low": 45, "high": 95},
    "D1S1656": {"low": 45, "high": 95},
    "D2S441": {"low": 45, "high": 95},
    "D10S1248": {"low": 45, "high": 95},
    "D13S317": {"low": 45, "high": 95},
    "Penta E": {"low": 45, "high": 95},
    "D16S539": {"low": 50, "high": 140},
    "D18S51": {"low": 50, "high": 140},
    "D2S1338": {"low": 50, "high": 140},
    "CSF1PO": {"low": 50, "high": 140},
    "Penta D": {"low": 50, "high": 140},
    "TH01": {"low": 45, "high": 85},
    "vWA": {"low": 45, "high": 85},
    "D21S11": {"low": 45, "high": 85},
    "D7S820": {"low": 45, "high": 85},
    "D5S818": {"low": 45, "high": 85},
    "TPOX": {"low": 45, "high": 85},
    "D8S1179": {"low": 80, "high": 135},
    "D12S391": {"low": 80, "high": 135},
    "D19S433": {"low": 80, "high": 135},
    "SE33": {"low": 80, "high": 135},
    "D22S1045": {"low": 80, "high": 135},
    "DYS391": {"low": 40, "high": 95},
    "FGA": {"low": 40, "high": 95},
    "DYS576": {"low": 40, "high": 95},
    "DYS570": {"low": 40, "high": 95},
}


def get_rfu_thresholds(
    locus: str | None = None,
    low_or_high: str | None = None,
    min_rfu: int | None = None,
) -> Mapping[str, int | dict[str, int] | None]:
    """Get per-locus RFU thresholds for allele filtering.

    Supports multiple modes:

    - ``low_or_high="low"`` or ``"high"``: Use kit-specific thresholds.
    - ``low_or_high="between"``: Returns both low and high as a dict
      (for filtering peaks that fall *between* the thresholds).
    - ``min_rfu=<value>``: Use a uniform threshold for all loci.
    - Both ``None``: No threshold (returns ``None`` for all loci).

    Args:
        locus: If provided, return threshold(s) for this locus only.
        low_or_high: Which kit threshold to use — ``"low"``, ``"high"``,
            or ``"between"``.
        min_rfu: Manual uniform threshold (overrides ``low_or_high``
            when ``low_or_high`` is ``None``).

    Returns:
        Mapping from locus name to threshold value (int, dict, or None).

    Raises:
        ValueError: If ``locus`` is unknown or ``low_or_high`` is invalid.
    """
    if locus and locus not in THRESHOLDS_PER_LOCUS:
        raise ValueError(f"Unknown locus: {locus!r}")
    if low_or_high and low_or_high not in ("low", "high", "between"):
        raise ValueError(
            f"Expected 'low', 'high', or 'between', got {low_or_high!r}"
        )

    if low_or_high == "between":
        if locus:
            return {locus: THRESHOLDS_PER_LOCUS[locus]}
        return dict(THRESHOLDS_PER_LOCUS)

    if locus and low_or_high:
        return {locus: THRESHOLDS_PER_LOCUS[locus][low_or_high]}
    elif locus and not low_or_high:
        return {locus: min_rfu}
    elif not locus and low_or_high:
        return {k: v[low_or_high] for k, v in THRESHOLDS_PER_LOCUS.items()}
    else:
        return {k: min_rfu for k in THRESHOLDS_PER_LOCUS}


def flatten_markers_to_allele_names(
    markers: Sequence[Marker],
    *,
    locus: str | None = None,
    min_rfu: int | None = None,
    low_or_high: str | None = None,
) -> frozenset[str]:
    """Flatten a sequence of Markers to a set of ``"MarkerName_AlleleName"`` strings.

    Optionally filters by locus name and/or RFU threshold.

    Example::

        >>> markers = [Marker("D5S818", 0, (Allele("13", height=500), Allele("15", height=200)))]
        >>> flatten_markers_to_allele_names(markers)
        frozenset({'D5S818_13', 'D5S818_15'})

    Args:
        markers: Sequence of Marker objects to flatten.
        locus: If provided, only include alleles from this locus.
        min_rfu: Manual RFU threshold.
        low_or_high: Kit-specific threshold mode (``"low"``, ``"high"``,
            or ``"between"``).

    Returns:
        Frozen set of ``"marker_allele"`` strings.

    Raises:
        ValueError: If an RFU threshold is set but allele height is None.
    """
    thresholds = get_rfu_thresholds(locus, low_or_high, min_rfu)

    result: list[str] = []
    for marker in markers:
        if locus and marker.name != locus:
            continue

        threshold = thresholds.get(marker.name)

        for allele in marker.alleles:
            if not _passes_rfu_filter(allele, threshold):
                continue
            result.append(f"{marker.name}_{allele.name}")

    result_set = frozenset(result)
    if len(result_set) != len(result):
        logger.warning("Non-unique locus-allele combinations found")
    return result_set


def _passes_rfu_filter(
    allele: Allele,
    threshold: int | dict[str, int] | None,
) -> bool:
    """Check if an allele passes the RFU threshold filter.

    Args:
        allele: The allele to check.
        threshold: An int (minimum RFU), a dict with 'low'/'high'
            (between filter), or None (no filter).

    Returns:
        True if the allele passes the filter.

    Raises:
        ValueError: If a threshold is set but ``allele.height`` is None.
    """
    if threshold is None:
        return True

    if allele.height is None:
        raise ValueError(
            f"RFU threshold is set but allele '{allele.name}' has no height. "
            "Set min_rfu and low_or_high to None to skip thresholding."
        )

    if isinstance(threshold, dict):
        # "between" mode: keep if low < height < high
        return threshold["low"] < allele.height < threshold["high"]

    # Simple minimum threshold
    return allele.height >= threshold
