"""Allele calling from segmentation predictions.

Translates pixel-level segmentation masks into allele calls by finding
connected components of positive predictions and mapping their positions
to the nearest allele in the reference panel.

Design pattern: **Strategy**
    :class:`AlleleCaller` defines the interface; implementations provide
    different allele-calling algorithms. Currently:

    - :class:`NearestBasePairCaller` — Assigns each predicted region to the
      nearest allele by base-pair distance.
"""

from __future__ import annotations

import abc
from collections import defaultdict

import numpy as np
from loguru import logger

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.panel import Panel


class AlleleCaller(abc.ABC):
    """Abstract base class for allele calling strategies.

    Implementations translate a pixel-level segmentation mask into a
    sequence of :class:`~dnanet.core.marker.Marker` objects with called
    alleles.
    """

    @abc.abstractmethod
    def call_alleles(
        self,
        prediction_image: np.ndarray,
        signal_image: np.ndarray,
        scaler: np.ndarray,
        panel: Panel,
    ) -> tuple[Marker, ...]:
        """Call alleles from a segmentation prediction.

        Args:
            prediction_image: (C, L) predicted mask (probabilities or binary).
            signal_image: (C, L) raw EPG signal data (for RFU extraction).
            scaler: (L, ) or (1, L) array mapping scan positions to base pairs.
            panel: Reference panel with allele definitions.

        Returns:
            Tuple of Markers with called alleles.
        """


class NearestBasePairCaller(AlleleCaller):
    """Call alleles by nearest base-pair matching.

    For each connected component of positive predictions:
    1. Compute the mean base-pair position (via the scaler).
    2. Find the closest allele in the panel for that dye channel.
    3. Record the peak height (max RFU in the component).

    Args:
        threshold: Probability threshold for positive predictions.
        exclude_non_autosomal: If True, filter out non-autosomal markers
            from the results.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        exclude_non_autosomal: bool = False,
    ) -> None:
        self.threshold = threshold
        self.exclude_non_autosomal = exclude_non_autosomal

    def call_alleles(
        self,
        prediction_image: np.ndarray,
        signal_image: np.ndarray,
        scaler: np.ndarray,
        panel: Panel,
    ) -> tuple[Marker, ...]:
        markers = self._translate_pixels_to_alleles(
            scaler, prediction_image, signal_image, panel,
        )

        if self.exclude_non_autosomal:
            markers = tuple(m for m in markers if m.is_autosomal)

        return markers

    def _translate_pixels_to_alleles(
        self,
        scaler: np.ndarray,
        prediction_image: np.ndarray,
        signal_image: np.ndarray,
        panel: Panel,
    ) -> tuple[Marker, ...]:
        """Translate pixel predictions to marker/allele names.

        Finds connected components of positive predictions, maps each to a
        base-pair position via the scaler, then looks up the nearest allele
        in the panel.
        """
        # Ensure scaler is 2D: (1, L)
        if scaler.ndim == 1:
            scaler = scaler[np.newaxis, :]

        loci_dict: dict[tuple[int, str], set[tuple[str, float]]] = defaultdict(set)
        rfus: dict[tuple[str, str], int] = defaultdict(int)

        for dye_index, dye_pred in enumerate(prediction_image):
            positives = np.where(dye_pred >= self.threshold)[0]
            if positives.size == 0:
                logger.debug("No predictions in dye row {}", dye_index)
                continue

            # Split into connected components (consecutive indices)
            predicted_bins = np.split(
                positives,
                np.where(np.diff(positives) != 1)[0] + 1,
            )

            for prediction_bin in predicted_bins:
                bin_basepairs = scaler[:, prediction_bin]
                bin_start, bin_end = np.min(bin_basepairs), np.max(bin_basepairs)

                mean_bp = float(np.mean(bin_basepairs))
                marker_name, allele_name = self._get_nearest_allele(
                    dye_index, mean_bp, panel,
                )
                _, allele_start, allele_end = panel.get_allele_basepair_and_bins(marker_name, allele_name)
                if not (bin_start >= allele_start and bin_end <= allele_end):
                    # Predicted bin does not fall inside matched allele bin, therefore set name to "Out of Bin".
                    allele_name = "OB"
                # Also store the mean_bp the allele was found on, as we may find multiple "OB" peaks.
                loci_dict[(dye_index, marker_name)].add((allele_name, mean_bp))

                # Track highest RFU for this allele
                max_rfu = int(np.max(signal_image[dye_index, prediction_bin]))
                rfus[(marker_name, allele_name)] = max(
                    rfus[(marker_name, allele_name)], max_rfu,
                )

        return tuple(
            Marker(
                name=marker_name,
                dye_row=dye_index,
                alleles=frozenset(
                    Allele(name=allele_name, height=rfus[(marker_name, allele_name)], base_pair=bp)
                    for allele_name, bp in sorted(alleles)
                ),
            )
            for (dye_index, marker_name), alleles in loci_dict.items()
        )

    @staticmethod
    def _get_nearest_allele(
        dye_index: int,
        base_pair: float,
        panel: Panel,
    ) -> tuple[str, str]:
        """Find the nearest allele in the panel for a given dye and base-pair.

        Args:
            dye_index: 0-based dye channel index.
            base_pair: Mean base-pair position of the predicted region.
            panel: Reference panel.

        Returns:
            Tuple of (marker_name, allele_name).
        """
        dye_mapping = panel.dye_bp_to_allele_mapping.get(dye_index, {})
        if not dye_mapping:
            return "Unknown", "Unknown"

        nearest_bp = min(dye_mapping.keys(), key=lambda k: abs(k - base_pair))
        return dye_mapping[nearest_bp]
