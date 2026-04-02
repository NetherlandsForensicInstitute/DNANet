"""PeakWindowDataset — dataset producing extracted peak windows.

Extends :class:`~dnanet.data.hid_dataset.HIDDataset` to extract individual
peak windows from full DNA profiles. Each item is an
:class:`~dnanet.data.extracted_peak.ExtractedPeak` instead of a
:class:`~dnanet.data.image.HIDImage`.

This is the dataset used for training the standalone peak classifier.

Design pattern: **Decorator**
    Wraps an existing ``HIDDataset`` and transforms its items from full
    profiles to extracted peak windows, adding peak-specific preprocessing
    (optional smoothing, log-scale normalization) on top.
"""

from __future__ import annotations

import logging
from typing import Iterator, Sequence

from tqdm import tqdm
from loguru import logger

from dnanet.data.dataset import SimpleDataset, InMemoryDataset
from dnanet.data.hid_dataset import HIDDataset
from dnanet.data.extracted_peak import ExtractedPeak
from dnanet.data.preprocessing.scaling import RFU_MAX_VALUE, scale_rfu_numpy
from dnanet.data.preprocessing.baseline import fft_lowpass_smooth
from dnanet.data.preprocessing.peak_extraction import extract_peak_windows
from dnanet.data.strategies import StrategyRegistry


class PeakWindowDataset(InMemoryDataset):
    """Dataset of extracted peak windows from DNA profiles.

    Takes a base :class:`HIDDataset`, extracts peaks from every loaded
    profile, and presents them as a flat sequence of
    :class:`ExtractedPeak` objects.

    Args:
        base_dataset: Source dataset of full DNA profiles.
        threshold: Minimum RFU height for peak detection.
        window_size: Width of extraction window in scan points.
        labels: Class label names. Index 0 is the first class.
        include_max_pool_dyes: Include max-pooled other-dye channel.
        preprocess: Apply preprocessing (smoothing + scaling) to peaks.
        smooth_keep_factor: FFT smoothing keep fraction (None to skip).
        log_scale: Apply log1p scaling during preprocessing.
        max_rfu_value: Max RFU for normalization (None to skip).
        use_ground_truth: Use annotation image for labeling.
    """

    def __init__(
        self,
        base_dataset: HIDDataset | InMemoryDataset,
        threshold: float = 40,
        window_size: int = 120,
        include_max_pool_dyes: bool = False,
        preprocess: bool = True,
        smooth_keep_factor: float | None = 0.4,
        log_scale: bool = True,
        max_rfu_value: int | None = RFU_MAX_VALUE,
        use_ground_truth: bool = True,
    ) -> None:
        super().__init__(shuffle=False)


        self.threshold = threshold
        self.window_size = window_size
        self.labels = StrategyRegistry.get_dataset_strategy().get_annotation_classes()
        self.label_to_idx = {name: idx for idx, name in enumerate(self.labels)}
        self.idx_to_label = {idx: name for idx, name in enumerate(self.labels)}
        self.include_max_pool_dyes = include_max_pool_dyes
        self.preprocess = preprocess
        self.smooth_keep_factor = smooth_keep_factor
        self.log_scale = log_scale
        self.max_rfu_value = max_rfu_value
        self.use_ground_truth = use_ground_truth

        # Extract peaks from all images
        self._data = list(tqdm(
            self._iterate_peaks(base_dataset),
            desc="Extracting peak windows",
            unit="peaks",
        ))

        logger.info(
            "PeakWindowDataset: extracted {} peaks from {} profiles "
            "(threshold={}, window={})",
            len(self._data), len(base_dataset),
            threshold, window_size,
        )

    def _iterate_peaks(
        self, base_dataset: InMemoryDataset,
    ) -> Iterator[ExtractedPeak]:
        """Extract and optionally preprocess peaks from all images."""
        for image in base_dataset:
            peaks = extract_peak_windows(
                image,
                threshold=self.threshold,
                window_size=self.window_size,
                include_max_pool_dyes=self.include_max_pool_dyes
            )

            for peak in peaks:
                if self.preprocess:
                    self._preprocess_peak(peak)
                yield peak

    def _preprocess_peak(self, peak: ExtractedPeak) -> None:
        """Apply in-place preprocessing to a peak's data.

        Optionally applies FFT smoothing and RFU scaling.
        """
        data = peak.data.astype("float64")

        if self.smooth_keep_factor is not None:
            data = fft_lowpass_smooth(data, self.smooth_keep_factor)

        if self.log_scale or self.max_rfu_value is not None:
            data = scale_rfu_numpy(
                data,
                log_scale=self.log_scale,
                max_rfu=self.max_rfu_value,
            )

        peak._data = data
