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

from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info

from dnanet.data.dataset import TransformableDataset
from dnanet.data.preprocessing.baseline import fft_lowpass_smooth
from dnanet.data.preprocessing.peak_extraction import extract_peak_windows
from dnanet.data.preprocessing.scaling import RFU_MAX_VALUE, scale_rfu_numpy
from tqdm import tqdm

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
        load_in_memory: Eagerly extract and cache all peaks at init time.
    """

    def __init__(
        self,
        dataset_strategy: DatasetStrategy,
        images: List[HIDImage] | None = None,
        base_dataset: HIDDataset | None = None,
        image_indices: List[int] | None = None,
        transform: TransformDataCallable | None = None,
        threshold: float = 40,
        window_size: int = 120,
        include_max_pool_dyes: bool = False,
        preprocess: bool = True,
        smooth_keep_factor: float | None = 0.4,
        log_scale: bool = True,
        max_rfu_value: int | None = RFU_MAX_VALUE,
        load_in_memory: bool = False,
        _cached_peak_lists: list[list[ExtractedPeak]] | None = None,
    ) -> None:
        super().__init__()

        if (images is None) == (base_dataset is None):
            raise ValueError('Provide exactly one of `images` or `base_dataset`.')

        self._images = images
        self._base_dataset = base_dataset
        if image_indices is None:
            source_len = len(images) if images is not None else len(base_dataset)  # type: ignore[arg-type]
            image_indices = list(range(source_len))
        self._image_indices = list(image_indices)
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
        self.load_in_memory = load_in_memory
        self._cached_peak_lists = None

        if self.load_in_memory:
            self._cached_peak_lists = (
                _cached_peak_lists if _cached_peak_lists is not None else self._materialize_peaks()
            )

    @classmethod
    def from_hid_dataset(cls, base_dataset: HIDDataset, **kwargs):
        """Create a PeakWindowDataset backed by cached HID rows."""
        return PeakWindowDataset(
            dataset_strategy=base_dataset.dataset_strategy,
            base_dataset=base_dataset,
            transform=base_dataset.transform,
            **kwargs
        )

    def _iterate_peaks(self) -> Iterator[ExtractedPeak]:
        """Extract and optionally preprocess peaks from all images."""
        for image in self._iter_worker_images():
            yield from self._extract_peaks(image)

    def _extract_peaks(self, image: HIDImage) -> Iterator[ExtractedPeak]:
        """Extract peaks from one image and apply optional preprocessing."""
        peaks = extract_peak_windows(
            image,
            threshold=self.threshold,
            window_size=self.window_size,
            include_max_pool_dyes=self.include_max_pool_dyes,
            dataset_strategy=self.dataset_strategy
        )

        for peak in peaks:
            if self.preprocess:
                self._preprocess_peak(peak)
            yield peak

    def _materialize_peaks(self) -> list[list[ExtractedPeak]]:
        """Eagerly extract peaks for each image in this dataset."""
        logger.info("Eagerly extracting peaks for all images...")
        return [
            list(self._extract_peaks(self._materialize_image(index)))
            for index in tqdm(self._image_indices, desc='Extracting peaks', unit='image')
        ]

    def _materialize_image(self, index: int) -> HIDImage:
        if self._base_dataset is not None:
            return self._base_dataset.get_image(index)
        assert self._images is not None
        return self._images[index]

    def _worker_image_indices(self) -> List[int]:
        """Return the image indices assigned to the current dataloader worker."""
        worker_info = get_worker_info()
        if worker_info is None:
            return self._image_indices
        return self._image_indices[worker_info.id::worker_info.num_workers]

    def _iter_worker_images(self) -> Iterator[HIDImage]:
        """Yield only the images needed by the current worker."""
        for index in self._worker_image_indices():
            yield self._materialize_image(index)

    def _worker_images(self) -> List[HIDImage]:
        """Return the worker shard as a list of materialized images."""
        return list(self._iter_worker_images())

    def _iter_worker_peak_lists(self) -> Iterator[list[ExtractedPeak]]:
        """Yield only the cached peak lists needed by the current worker."""
        assert self._cached_peak_lists is not None

        worker_info = get_worker_info()
        if worker_info is None:
            yield from self._cached_peak_lists
            return

        yield from self._cached_peak_lists[worker_info.id::worker_info.num_workers]

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
        if self._base_dataset is not None:
            return [self._base_dataset.get_stub_image(idx) for idx in self._image_indices]
        assert self._images is not None
        return [self._images[idx] for idx in self._image_indices]

    @property
    def transform(self) -> TransformDataCallable | None:
        return self._transform

    @property
    def dataset_strategy(self) -> DatasetStrategy:
        return self._dataset_strategy

    def __iter__(self) -> Iterator[Any]:
        """Create an iterator for the peaks."""
        peaks = (
            (peak for peak_list in self._iter_worker_peak_lists() for peak in peak_list)
            if self._cached_peak_lists is not None
            else self._iterate_peaks()
        )

        for peak in peaks:
            if self.transform:
                yield self.transform(peak)
            else:
                yield peak

    def subset(self, indices: List[int]) -> PeakWindowDataset:
        """Create a subset of PeakWindowDataset with only indicated indices."""
        subset_indices = [self._image_indices[idx] for idx in indices]
        subset_peak_lists = None
        if self._cached_peak_lists is not None:
            subset_peak_lists = [self._cached_peak_lists[idx] for idx in indices]

        return PeakWindowDataset(
            dataset_strategy=self._dataset_strategy,
            images=self._images,
            base_dataset=self._base_dataset,
            image_indices=subset_indices,
            transform=self._transform,
            threshold=self.threshold,
            window_size=self.window_size,
            include_max_pool_dyes=self.include_max_pool_dyes,
            preprocess=self.preprocess,
            smooth_keep_factor=self.smooth_keep_factor,
            log_scale=self.log_scale,
            max_rfu_value=self.max_rfu_value,
            load_in_memory=self.load_in_memory,
            _cached_peak_lists=subset_peak_lists,
        )
