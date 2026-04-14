import abc
from abc import abstractmethod
from typing import Any, Tuple, Generic, TypeVar
from dataclasses import dataclass

import torch
from torch.utils.data import default_collate

from dnanet.data.image import HIDImage, TrainableElement
from dnanet.data.strategies import StrategyRegistry
from dnanet.data.extracted_peak import ExtractedPeak
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

    @abstractmethod
    def __call__(self, image: TrainableT) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch: list[Tuple[torch.Tensor | Tuple, torch.Tensor]]) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        return default_collate(batch)


@dataclass
class SegmentationTransformer(TransformDataCallable[HIDImage]):
    def __call__(self, image: HIDImage) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        data = image.data
        x = torch.tensor(data, dtype=torch.float32)

        # Target: segmentation mask with same shape
        if image.annotation is not None:
            y = torch.tensor(
                image.annotation.data,
                dtype=torch.float32,
            )
        else:
            y = torch.zeros_like(x)

        return x, y

@dataclass
class SegmentationTransformerMetaData(SegmentationTransformer):
    def __call__(self, image: HIDImage) -> tuple[torch.Tensor | Tuple, torch.Tensor, dict[str, Any]]:
        x, y = super().__call__(image)

        return x, y, _allele_metadata_from_image(image)

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor | Tuple, torch.Tensor, dict[str, Any]]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        xs, ys, metadata = zip(*batch, strict=True)

        return torch.stack(xs), torch.stack(ys), list(metadata)


@dataclass
class CombinedTransformer(TransformDataCallable[HIDImage]):
    threshold: int = 25
    window_size: int = 120
    include_max_pool_dyes: bool = False

    def __call__(self, image: HIDImage) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        data = image.data

        # Full image: (D, L, 1) → (D, L)
        if data.ndim == 3:
            data_2d = data[:, :, 0]
        else:
            data_2d = data

        full_image = torch.tensor(data_2d, dtype=torch.float32)

        # Extract peaks as tensors
        peak_windows, marker_idxs, peak_centers = extract_peaks_torch(
            image,
            threshold=self.threshold,
            window_size=self.window_size,
            include_max_pool_dyes=self.include_max_pool_dyes,
        )
        n_peaks = peak_windows.shape[0]

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

@dataclass
class CombinedTransformerMetaData(CombinedTransformer):
    def __call__(
        self,
        image: HIDImage,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor, dict[str, Any]]:
        inputs, target = super().__call__(image)

        return inputs, target, _allele_metadata_from_image(image)

    @staticmethod
    def collate_fn(
        batch: list[
            tuple[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
                torch.Tensor,
                dict[str, Any],
            ]
        ],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        inputs_and_targets = [(inputs, target) for inputs, target, _metadata in batch]
        metadata = [metadata for _inputs, _target, metadata in batch]
        inputs, targets = CombinedTransformer.collate_fn(inputs_and_targets)

        return inputs, targets, metadata

@dataclass
class ReconstructionTransformer(TransformDataCallable[HIDImage]):
    n_dyes: int = 5
    log_scale: bool = True
    max_rfu: int | None = None

    def __call__(self, image: HIDImage) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        data = image.data

        if data.ndim == 3:
            data_2d = data[:, :, 0]
        else:
            data_2d = data

        # Select dye channels (exclude size standard)
        data_2d = data_2d[: self.n_dyes]

        raw = torch.tensor(data_2d, dtype=torch.float32)
        preprocessed = scale_rfu_torch(raw, self.log_scale, self.max_rfu)

        # Input is preprocessed, target is the raw data (loss in
        # non-preprocessed space; inverse scaling is done in the
        # reconstruction module)
        return preprocessed, raw

@dataclass
class PeakClassificationTransformer(TransformDataCallable[ExtractedPeak]):
    include_marker: bool = True

    def __call__(self, peak: ExtractedPeak) -> Tuple[torch.Tensor | Tuple, torch.Tensor]:
        data = peak.data

        peak_tensor = torch.tensor(data, dtype=torch.float32)

        if self.include_marker:
            marker_idx = StrategyRegistry.get_scaling_strategy().marker_to_idx[peak.marker_name]
            marker_tensor = torch.tensor([marker_idx], dtype=torch.long)
        else:
            marker_tensor = torch.full(size=(1,), fill_value=-1, dtype=torch.long)

        ann_label: str = peak.annotation.data
        dataset_strategy = StrategyRegistry.get_dataset_strategy()
        annotation_to_idx = {
            name: idx for idx, name in enumerate(dataset_strategy.get_annotation_classes())
        }
        annotation_idx = annotation_to_idx[ann_label]
        target = torch.tensor(annotation_idx, dtype=torch.long)

        inputs = (peak_tensor, marker_tensor)
        return inputs, target
