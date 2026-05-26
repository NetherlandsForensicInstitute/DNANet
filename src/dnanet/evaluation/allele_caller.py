"""Allele calling strategies.

Design pattern: **Strategy**
    :class:`AlleleCaller` defines the interface; implementations provide
    different allele-calling algorithms.

Currently implemented are binary segmentation based algorithms, that translates pixel-level segmentation masks
into allele calls by finding connected components of positive predictions and mapping their positions
to an allele in the reference panel.
Implemented are:
    - :class:`NearestBasePairCaller` — Assigns each predicted region to the
      nearest allele by base-pair distance.
    - :class: `ExactBasePairCaller` - Assigns each predicted region to the
      allele that has not more than 0.3 base pairs distance.

# TODO: implement multi class allele calling.
"""

from __future__ import annotations

import abc
from collections import defaultdict

import numpy as np
from loguru import logger

from dnanet.core import Allele, Marker, Panel


class AlleleCaller(abc.ABC):
    """Abstract base class for allele calling strategies."""

    @abc.abstractmethod
    def call_alleles(
        self,
        **kwargs
    ) -> tuple[Marker, ...]:
        """Translate any input to a sequence of :class:`~dnanet.core.marker.Marker` objects with called alleles."""


class FromBinaryMaskCaller(AlleleCaller):
    """Base class for allele calling strategies from binary segmentation masks.

    Implementations translate a pixel-level segmentation mask into a
    sequence of :class:`~dnanet.core.marker.Marker` objects with called
    alleles.

    Args:
        threshold: Probability threshold for positive predictions.
        exclude_non_autosomal: If True, filter out non-autosomal markers from the results.
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
        """Call alleles from a segmentation prediction.

        Args:
            prediction_image: (C, L) predicted mask (probabilities or binary).
            signal_image: (C, L) raw EPG signal data (for RFU extraction).
            scaler: (L, ) array mapping scan positions to base pairs.
            panel: Reference panel with allele definitions.

        Returns:
            Tuple of Markers with called alleles.
        """
        markers = self._translate_pixels_to_alleles(
            scaler, prediction_image, signal_image, panel,
        )

        if self.exclude_non_autosomal:
            markers = tuple(m for m in markers if m.is_autosomal)
        # TODO: add flag to remove OB peaks?

        return markers

    def _translate_pixels_to_alleles(
        self,
        scaler: np.ndarray,
        prediction_image: np.ndarray,
        signal_image: np.ndarray,
        panel: Panel,
    ) -> tuple[Marker, ...]:
        """Translate pixel-level segmentation masks to alleles.

        For each connected component of positive predictions:
        1. Compute the base pair positions of the prediction (via the scaler).
        2. Translate the base pair position of the maximum rfu value (the peak top) and dye index to
        marker and allele name.
        3. Record the peak height (max RFU in the component).
        """
        loci_dict: dict[tuple[int, str], set[tuple[str, float]]] = defaultdict(set)
        rfus: dict[tuple[str, str, float], int] = defaultdict(int)

        for dye_index, dye_pred in enumerate(prediction_image):
            # Find indices of positive predictions.
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
                bin_basepairs = scaler[prediction_bin]  # Use the scaler to translate pixels to base pairs.
                bin_rfus = signal_image[dye_index, prediction_bin]  # Find the rfu values inside the predicted bin.
                bp_max_rfu = float(bin_basepairs[np.argmax(bin_rfus)])  # Find the base pair of the highest rfu.

                marker_name, allele_name = self.call_allele_from_basepair(
                    dye_index, bp_max_rfu, panel,
                )

                # Store the allele_name and also the mean_bp the allele was found on
                loci_dict[(dye_index, marker_name)].add((allele_name, bp_max_rfu))

                # Track highest RFU for this allele
                max_rfu = int(np.max(bin_rfus))
                rfus[(marker_name, allele_name, bp_max_rfu)] = max(
                    rfus[(marker_name, allele_name, bp_max_rfu)], max_rfu,
                )

        return tuple(
            Marker(
                name=marker_name,
                dye_row=dye_index,
                alleles=frozenset(
                    Allele(name=allele_name, height=rfus[(marker_name, allele_name, bp)], base_pair=bp)
                    for allele_name, bp in sorted(alleles)
                ),
            )
            for (dye_index, marker_name), alleles in loci_dict.items()
        )

    @staticmethod
    def call_allele_from_basepair(dye_index: int, base_pair: float, panel: Panel) -> tuple[str, str]:
        """Translate the dye index and the base pair position on the dye to an allele name using the panel."""
        raise NotImplementedError


class NearestBasePairCaller(FromBinaryMaskCaller):
    """Call alleles by nearest base-pair matching."""

    def __init__(
            self,
            threshold: float = 0.5,
            exclude_non_autosomal: bool = False,
    ) -> None:
        super().__init__(threshold, exclude_non_autosomal)

    @staticmethod
    def call_allele_from_basepair(
        dye_index: int,
        base_pair: float,
        panel: Panel,
    ) -> tuple[str, str]:
        """Find the nearest allele in the panel for a given dye and base-pair and returns its allele name and
        marker name. If no alleles can be found, return 'Unknown' for both names.

        Args:
            dye_index: 0-based dye channel index.
            base_pair: Base-pair position of the predicted region.
            panel: Reference panel.

        Returns:
            Tuple of (marker_name, allele_name).
        """
        dye_mapping = panel.dye_bp_to_allele_mapping.get(dye_index, {})
        if not dye_mapping:
            logger.error("No dye mapping available for dye row {}", dye_index)
            return "Unknown", "Unknown"

        nearest_bp = min(dye_mapping.keys(), key=lambda k: abs(k - base_pair))
        return dye_mapping[nearest_bp]


class ExactBasePairCaller(FromBinaryMaskCaller):
    """Call alleles by comparing predicted base pairs to base pairs of the Panel.

    Consider a maximum allowed distance of 0.3 base pairs between the found base pair and the bin center. Demand
    base pairs to fall in exactly one bin of the panel, otherwise regard the peak 'Out of Bin'.
    """

    def __init__(
            self,
            threshold: float = 0.5,
            exclude_non_autosomal: bool = False,
    ) -> None:
        super().__init__(threshold, exclude_non_autosomal)

    @staticmethod
    def call_allele_from_basepair(
            dye_index: int,
            base_pair: float,
            panel: Panel,
    ) -> tuple[str, str]:
        """Find the allele and marker for a given dye and base-pair by comparing the base pair to the bin base pairs
        stored in the panel. Demand that the base pair must be within 0.3 difference of the bin base pair, otherwise
        regard the peak as 'Out of Bin'.

        Args:
            dye_index: 0-based dye channel index.
            base_pair: Base-pair position of the predicted region.
            panel: Reference panel.

        Returns:
            Tuple of (marker_name, allele_name).
        """
        dye_mapping = panel.dye_bp_to_allele_mapping.get(dye_index, {})
        if not dye_mapping:
            logger.error("No dye mapping available for dye row {}", dye_index)
            return "Unknown", "Unknown"

        candidate_names = set()
        for bin_bp, (marker, allele) in dye_mapping.items():
            if abs(bin_bp - base_pair) < 0.3:
                candidate_names.add((marker, allele))

        if len(candidate_names) == 1:
            # We have one marker/allele matched, return the found candidate.
            return list(candidate_names)[0]

        # Try to see if we can get the marker name, as the peak might not fall in an exact allele bin, but might
        # fall in a marker.
        return panel.get_marker_name_by_dye_and_bp(dye_index, base_pair), "Out of Bin"
