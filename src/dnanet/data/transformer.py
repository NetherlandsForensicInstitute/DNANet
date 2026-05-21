import abc
from abc import abstractmethod
from typing import Any, Tuple, Generic, TypeVar
from dataclasses import dataclass

import torch
from torch.utils.data import default_collate

from dnanet.data.image import HIDImage, TrainableElement
from dnanet.data.strategies import DatasetStrategy
from dnanet.data.extracted_peak import ExtractedPeak
from dnanet.data.strategies.scaling import ScalingStrategy
from dnanet.data.preprocessing.scaling import scale_rfu_torch
from dnanet.data.preprocessing.peak_extraction import extract_peaks_torch


TrainableT = TypeVar('TrainableT', bound=TrainableElement)

def _allele_metadata_from_image(image: HIDImage) -> dict[str, Any]:
    return {
        "allele_annotation": image.allele_annotation,
        "panel": image.adjusted_panel,
        "path": image.path,
        "scaler": image.scaler,
        "signal_image": image.data,
    }


class TransformDataCallable(abc.ABC, Generic[TrainableT]):
    """Base class for transforms."""
    @abstractmethod
    def __call__(self, image: TrainableT) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch: list[Tuple[torch.Tensor | Tuple, torch.Tensor]]) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        return default_collate(batch)


@dataclass(frozen=True)
class SegmentationTransformer(TransformDataCallable[HIDImage]):
    """Transformer used for segmentation tasks.

    Target output here is a scanpoint indexed label.
    """
    def __call__(self, image: HIDImage) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        data = image.data
        x = torch.tensor(data, dtype=torch.float32)

        # Target: segmentation mask with same shape
        if image.annotation is not None:
            y = torch.tensor(
                image.annotation.data,
                dtype=torch.int64,
            )
        else:
            y = torch.zeros_like(x)

        return x, y

@dataclass(frozen=True)
class AlleleMetadataTransformer(TransformDataCallable[HIDImage]):
    """Transformer that adds allele metadata to predictions to enable allele-metrics."""
    transformer: TransformDataCallable[HIDImage]

    def __call__(
        self,
        image: HIDImage,
    ) -> tuple[torch.Tensor | Tuple, torch.Tensor, dict[str, Any]]:
        inputs, target = self.transformer(image)
        return inputs, target, _allele_metadata_from_image(image)

    def collate_fn(
        self,
        batch: list[tuple[torch.Tensor | Tuple, torch.Tensor, dict[str, Any]]],
    ) -> tuple[torch.Tensor | Tuple, torch.Tensor, list[dict[str, Any]]]:
        inputs_and_targets = [(inputs, target) for inputs, target, _metadata in batch]
        metadata = [metadata for _inputs, _target, metadata in batch]
        inputs, targets = self.transformer.collate_fn(inputs_and_targets)

        return inputs, targets, metadata


@dataclass(frozen=True)
class CombinedTransformer(TransformDataCallable[HIDImage]):  # noqa: D101
    threshold: int = 25
    window_size: int = 120
    include_max_pool_dyes: bool = False
    autoencoder_log_scale: bool = True
    autoencoder_max_rfu: int | None = None

    def __call__(self, image: HIDImage) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        data = image.data

        # Full image: (D, L, 1) → (D, L)
        if data.ndim == 3:
            data_2d = data[:, :, 0]
        else:
            data_2d = data

        # Extract peaks as tensors
        peak_windows, marker_idxs, peak_centers = extract_peaks_torch(
            image,
            scaling_strategy=image.scaling_strategy,
            threshold=self.threshold,
            window_size=self.window_size,
            include_max_pool_dyes=self.include_max_pool_dyes,
        )
        n_peaks = peak_windows.shape[0]

        ## preprocess image for autoencoder
        full_image = torch.tensor(data_2d, dtype=torch.float32)
        full_image = scale_rfu_torch(full_image, log_scale=self.autoencoder_log_scale, max_rfu=self.autoencoder_max_rfu)

        # TODO: preprocess peaks

        # Target: per-position annotation (D, L)
        if image.annotation is not None:
            ann = image.annotation.data
            if ann.ndim == 3:
                ann = ann[:, :, 0]
            target = torch.tensor(ann, dtype=torch.long)
        else:
            target = torch.zeros_like(full_image, dtype=torch.long)

        inputs = (full_image, peak_windows, marker_idxs, peak_centers, n_peaks)
        return inputs, target

    @staticmethod
    def collate_fn(batch: list[Tuple[torch.Tensor | Tuple, torch.Tensor]]) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        full_images = []
        targets = []
        peak_counts = []

        all_peak_windows = []
        all_marker_idxs = []
        all_peak_centers = []

        for inputs, target in batch:
            full_image, peak_windows, marker_idxs, peak_centers, n_peaks = inputs
            full_images.append(full_image)
            targets.append(target)
            peak_counts.append(int(n_peaks))

            all_peak_windows.append(peak_windows)
            all_marker_idxs.append(marker_idxs)
            all_peak_centers.append(peak_centers)


        full_images = torch.stack(full_images, dim=0)   # (N, D, L)
        targets = torch.stack(targets, dim=0)           # (N, D, L)
        peak_counts = torch.tensor(peak_counts, dtype=torch.long)  # (N,)

        peak_windows = torch.cat(all_peak_windows, dim=0)  # (P_total, C, W)
        marker_idxs = torch.cat(all_marker_idxs, dim=0)    # (P_total,)
        peak_centers = torch.cat(all_peak_centers, dim=0)

        new_inputs = (full_images, peak_windows, marker_idxs, peak_centers, peak_counts)
        return new_inputs, targets


@dataclass(frozen=True)
class ReconstructionTransformer(TransformDataCallable[HIDImage]):
    """Transformer for reconstruction tasks.
    
    This transformer will set sample and target to the same value to enable reconstruction.
    """
    n_dyes: int = 5
    autoencoder_log_scale: bool = True
    autoencoder_max_rfu: int | None = None

    def __call__(self, image: HIDImage) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        data = image.data

        if data.ndim == 3:
            data_2d = data[:, :, 0]
        else:
            data_2d = data

        # Select dye channels (exclude size standard)
        data_2d = data_2d[: self.n_dyes]

        raw = torch.tensor(data_2d, dtype=torch.float32)
        preprocessed = scale_rfu_torch(raw, self.autoencoder_log_scale, self.autoencoder_max_rfu)

        # Input is preprocessed, target is the raw data (loss in
        # non-preprocessed space; inverse scaling is done in the
        # reconstruction module)
        return preprocessed, raw

@dataclass(frozen=True)
class PeakClassificationTransformer(TransformDataCallable[ExtractedPeak]):
    """Transformer for the PeakNet Classifier."""
    include_marker: bool = True

    def __call__(self, peak: ExtractedPeak) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        data = peak.data

        peak_tensor = torch.tensor(data, dtype=torch.float32)

        annotation_idx = peak.annotation_idx
        target = torch.tensor(annotation_idx, dtype=torch.long)

        if self.include_marker:
            marker_idx = peak.marker_index
            marker_tensor = torch.tensor([marker_idx], dtype=torch.long)
            inputs = (peak_tensor, marker_tensor)
        else:
            inputs = peak_tensor

        return inputs, target
