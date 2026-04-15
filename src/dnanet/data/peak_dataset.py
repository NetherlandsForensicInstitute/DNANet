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

from typing import TYPE_CHECKING, Any, List, Iterator

from torch.utils.data import IterableDataset

from dnanet.data.dataset import TransformableDataset
from dnanet.data.preprocessing.baseline import fft_lowpass_smooth
from dnanet.data.preprocessing.peak_extraction import extract_peak_windows
from dnanet.data.preprocessing.scaling import RFU_MAX_VALUE, scale_rfu_numpy

if TYPE_CHECKING:
    from dnanet.data.image import HIDImage
    from dnanet.data.hid_dataset import HIDDataset
    from dnanet.data.transformer import TransformDataCallable
    from dnanet.data.extracted_peak import ExtractedPeak
    from dnanet.data.strategies import DatasetStrategy


class PeakWindowDataset(IterableDataset, TransformableDataset):
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
    """

    def __init__(
        self,
        images: List[HIDImage],
        dataset_strategy: DatasetStrategy,
        transform: TransformDataCallable | None = None,
        threshold: float = 40,
        window_size: int = 120,
        include_max_pool_dyes: bool = False,
        preprocess: bool = True,
        smooth_keep_factor: float | None = 0.4,
        log_scale: bool = True,
        max_rfu_value: int | None = RFU_MAX_VALUE,
        # TODO: add load_in_memory option
    ) -> None:
        super().__init__()

        self._images = images
        self._transform = transform
        self._dataset_strategy = dataset_strategy
        self.threshold = threshold
        self.window_size = window_size
        self.labels = self._dataset_strategy.get_annotation_classes()
        self.label_to_idx = {name: idx for idx, name in enumerate(self.labels)}
        self.idx_to_label = {idx: name for idx, name in enumerate(self.labels)}
        self.include_max_pool_dyes = include_max_pool_dyes
        self.preprocess = preprocess
        self.smooth_keep_factor = smooth_keep_factor
        self.log_scale = log_scale
        self.max_rfu_value = max_rfu_value

    @classmethod
    def from_hid_dataset(cls, base_dataset: HIDDataset, **kwargs):
        """Create a PeakWindowDataset based on a HIDDataset's images and transform."""
        return cls.__init__(
            images=base_dataset.images,
            dataset_strategy=base_dataset.dataset_strategy,
            transform=base_dataset.transform,
            **kwargs
        )

    def _iterate_peaks(self) -> Iterator[ExtractedPeak]:
        """Extract and optionally preprocess peaks from all images."""
        for image in self._images:
            peaks = extract_peak_windows(
                image,
                threshold=self.threshold,
                window_size=self.window_size,
                include_max_pool_dyes=self.include_max_pool_dyes,
            )

            for peak in peaks:
                if self.preprocess:
                    self._preprocess_peak(peak)
                yield peak

    def _preprocess_peak(self, peak: ExtractedPeak) -> None:
        """Apply in-place preprocessing to a peak's data.

        Optionally applies FFT smoothing and RFU scaling.
        """
        data = peak.data.astype('float64')

        if self.smooth_keep_factor is not None:
            data = fft_lowpass_smooth(data, self.smooth_keep_factor)

        if self.log_scale or self.max_rfu_value is not None:
            data = scale_rfu_numpy(
                data,
                log_scale=self.log_scale,
                max_rfu=self.max_rfu_value,
            )

        peak._data = data

    @property
    def images(self) -> List[HIDImage]:
        return self._images

    @property
    def transform(self) -> TransformDataCallable | None:
        return self._transform

    @property
    def dataset_strategy(self) -> DatasetStrategy:
        return self._dataset_strategy

    def __iter__(self) -> Iterator[Any]:
        """Create an iterator for the peaks."""
        for peak in self._iterate_peaks():
            if self.transform:
                yield self.transform(peak)
            else:
                yield peak

    def subset(self, indices: List[int]) -> PeakWindowDataset:
        """Create a subset of PeakWindowDataset with only indicated indices."""
        return PeakWindowDataset(
            images=[self._images[idx] for idx in indices],
            dataset_strategy=self._dataset_strategy,
            transform=self._transform,
            threshold=self.threshold,
            window_size=self.window_size,
            include_max_pool_dyes=self.include_max_pool_dyes,
            preprocess=self.preprocess,
            smooth_keep_factor=self.smooth_keep_factor,
            log_scale=self.log_scale,
            max_rfu_value=self.max_rfu_value,
        )
