"""Allele calling strategies.

Design pattern: **Strategy**
    :class:`AlleleCaller` defines the interface; implementations provide
    different allele-calling algorithms.

Currently implemented are prediction-image based algorithms that translate
pixel-level model outputs into allele calls by finding connected components
of allele-positive predictions and mapping their positions to an allele in
the reference panel.
Implemented are:
    - :class:`NearestBasePairCaller` — Assigns each predicted region to the
      nearest allele by base-pair distance.
    - :class:`ExactBasePairCaller` - Assigns each predicted region to the
      allele that has not more than 0.3 base pairs distance.
"""

from __future__ import annotations

import abc
from typing import Literal
from collections import defaultdict

import numpy as np
from loguru import logger

from dnanet.core import Panel, Allele, Marker, LabelCategory


class AlleleCaller(abc.ABC):
    """Abstract base class for allele calling strategies."""

    @abc.abstractmethod
    def call_alleles(
        self,
        **kwargs
    ) -> tuple[Marker, ...]:
        """Translate any input to a sequence of :class:`~dnanet.core.marker.Marker` objects with called alleles."""


class FromSegmentationImageCaller(AlleleCaller):
    """Base class for allele calling strategies from 2-D prediction images.

    Implementations translate a pixel-level prediction image into a
    sequence of :class:`~dnanet.core.marker.Marker` objects with called
    alleles. The prediction image can either be:
    - a binary / probabilistic allele mask, or
    - a 2-D class-index map where a single class denotes allele signal.

    Args:
        threshold: Probability threshold for positive predictions when
            ``prediction_mode`` resolves to ``"binary"``.
        exclude_non_autosomal: If True, filter out non-autosomal markers from the results.
        prediction_mode: Representation used by ``prediction_image``.
        allele_class_index: Class index to treat as allele signal when
            ``prediction_mode`` resolves to ``"multiclass_labels"``.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        exclude_non_autosomal: bool = False,
        prediction_mode: Literal["auto", "binary", "multiclass_labels"] = "auto",
    ) -> None:
        if prediction_mode not in {"auto", "binary", "multiclass_labels"}:
            raise ValueError(
                "prediction_mode must be 'auto', 'binary', or 'multiclass_labels', "
                f"got {prediction_mode!r}."
            )
        self.threshold = threshold
        self.exclude_non_autosomal = exclude_non_autosomal
        self.prediction_mode = prediction_mode
        self.allele_class_index = list(LabelCategory).index(LabelCategory.ALLELE)

    def call_alleles(
        self,
        prediction_image: np.ndarray,
        signal_image: np.ndarray,
        scaler: np.ndarray,
        panel: Panel,
    ) -> tuple[Marker, ...]:
        """Call alleles from a prediction image.

        Args:
            prediction_image: ``(C, L)`` predicted image. Either probabilities /
            binary values in ``"binary"`` mode or class indices in
                ``"multiclass_labels"`` mode. In ``"auto"`` mode, float and bool
                arrays are treated as binary while integer arrays with labels
                outside ``{0, 1}`` are treated as multiclass.
            signal_image: (C, L) raw EPG signal data (for RFU extraction).
            scaler: (L, ) array mapping scan positions to base pairs.
            panel: Reference panel with allele definitions.

        Returns:
            Tuple of Markers with called alleles.
        """
        allele_mask = self._prediction_to_allele_mask(prediction_image)
        markers = self._translate_pixels_to_alleles(
            scaler, allele_mask, signal_image, panel,
        )

        if self.exclude_non_autosomal:
            markers = tuple(m for m in markers if m.is_autosomal)
        # TODO: add flag to remove OB peaks?

        return markers

    def _prediction_to_allele_mask(self, prediction_image: np.ndarray) -> np.ndarray:
        """Normalize supported prediction representations to an allele mask."""
        prediction = np.asarray(prediction_image)
        if prediction.ndim != 2:
            raise ValueError(
                "prediction_image must be a 2-D array of shape (num_dyes, scanpoints), "
                f"got shape {prediction.shape}."
            )

        prediction_mode = self._resolve_prediction_mode(prediction)
        if prediction_mode == "binary":
            return prediction >= self.threshold

        return prediction == self.allele_class_index

    def _resolve_prediction_mode(
        self,
        prediction: np.ndarray,
    ) -> Literal["binary", "multiclass_labels"]:
        """Resolve the effective prediction mode for a normalized array."""
        if self.prediction_mode != "auto":
            return self.prediction_mode

        if prediction.dtype == np.bool_ or np.issubdtype(prediction.dtype, np.floating):
            return "binary"

        if np.issubdtype(prediction.dtype, np.integer):
            unique_values = np.unique(prediction)
            if np.all(np.isin(unique_values, (0, 1))):
                raise ValueError(
                    "prediction_image is ambiguous in auto mode: integer arrays containing "
                    "only values {0, 1} could be either binary masks or multiclass labels. "
                    "Set allele caller prediction_mode explicitly."
                )
            return "multiclass_labels"

        raise ValueError(
            "prediction_image has unsupported dtype for auto mode: "
            f"{prediction.dtype}."
        )

    def _translate_pixels_to_alleles(
        self,
        scaler: np.ndarray,
        allele_mask: np.ndarray,
        signal_image: np.ndarray,
        panel: Panel,
    ) -> tuple[Marker, ...]:
        """Translate an allele-positive mask to allele calls.

        For each connected component of positive predictions:
        1. Compute the base pair positions of the prediction (via the scaler).
        2. Translate the base pair position of the maximum rfu value (the peak top) and dye index to
        marker and allele name.
        3. Record the peak height (max RFU in the component).
        """
        loci_dict: dict[tuple[int, str], set[tuple[str, float]]] = defaultdict(set)
        rfus: dict[tuple[str, str, float], int] = defaultdict(int)

        for dye_index, dye_pred in enumerate(allele_mask):
            # Find indices of positive predictions.
            positives = np.where(dye_pred)[0]
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


class NearestBasePairCaller(FromSegmentationImageCaller):
    """Call alleles by nearest base-pair matching."""

    def __init__(
            self,
            threshold: float = 0.5,
            exclude_non_autosomal: bool = False,
            prediction_mode: Literal["auto", "binary", "multiclass_labels"] = "auto",
    ) -> None:
        super().__init__(
            threshold=threshold,
            exclude_non_autosomal=exclude_non_autosomal,
            prediction_mode=prediction_mode,
        )

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


class ExactBasePairCaller(FromSegmentationImageCaller):
    """Call alleles by comparing predicted base pairs to base pairs of the Panel.

    Consider a maximum allowed distance of 0.3 base pairs between the found base pair and the bin center. Demand
    base pairs to fall in exactly one bin of the panel, otherwise regard the peak 'Out of Bin'.
    """

    def __init__(
            self,
            threshold: float = 0.5,
            exclude_non_autosomal: bool = False,
            prediction_mode: Literal["auto", "binary", "multiclass_labels"] = "auto",
    ) -> None:
        super().__init__(
            threshold=threshold,
            exclude_non_autosomal=exclude_non_autosomal,
            prediction_mode=prediction_mode,
        )

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
