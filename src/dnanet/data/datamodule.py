"""PyTorch Lightning DataModules for DNANet.

Provides DataModules for all training scenarios:

- :class:`DNANetDataModule` — Full-profile segmentation (UNet).
- :class:`PeakDataModule` — Standalone peak classification.
- :class:`PeakNetDataModule` — Combined PeakNet (full images + peak windows).
- :class:`ReconstructionDataModule` — Autoencoder reconstruction with
  RFU preprocessing.

Design pattern: **Adapter**
    Each DataModule adapts DNANet's domain-specific datasets to PyTorch
    Lightning's ``LightningDataModule`` interface.

Design pattern: **Bridge**
    Bridges between two independent hierarchies: DNANet data loading
    and PyTorch training infrastructure.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader

from dnanet.data.image import HIDImage
from dnanet.data.dataset import InMemoryDataset
from dnanet.data.strategies import StrategyRegistry


# ---------------------------------------------------------------------------
# Segmentation (full-profile)
# ---------------------------------------------------------------------------


class HIDTorchDataset(Dataset):
    """Wraps a list of HIDImages as a PyTorch Dataset.

    Each item is returned as a ``(input_tensor, target_tensor)`` tuple
    ready for the segmentation model.

    Design pattern: **Adapter**
        Adapts HIDImage's numpy-based data access to PyTorch's tensor-based
        Dataset protocol.
    """

    def __init__(self, images: Sequence[HIDImage]) -> None:
        self.images = list(images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]

        # Input: (dyes, signal_length, 1) -> (1, dyes, signal_length)
        data = image.data
        x = torch.tensor(np.transpose(data, (2, 0, 1)), dtype=torch.float32)

        # Target: segmentation mask with same shape
        if image.annotation is not None:
            y = torch.tensor(
                np.transpose(image.annotation.data, (2, 0, 1)),
                dtype=torch.float32,
            )
        else:
            y = torch.zeros_like(x)

        return x, y


class DNANetDataModule(L.LightningDataModule):
    """Lightning DataModule for DNA profile segmentation.

    Args:
        dataset: A loaded InMemoryDataset (e.g. HIDDataset).
        batch_size: Batch size for DataLoaders.
        val_fraction: Fraction of data to use for validation.
        num_workers: Number of DataLoader workers.
        seed: Random seed for train/val splitting.
    """

    def __init__(
        self,
        dataset: InMemoryDataset,
        batch_size: int = 16,
        val_fraction: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed

        self._train_dataset: HIDTorchDataset | None = None
        self._val_dataset: HIDTorchDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Split the dataset and wrap in PyTorch Datasets."""
        if self._train_dataset is not None:
            return  # already set up

        train_data, val_data = self._dataset.split(fraction=1.0 - self.val_fraction, seed=self.seed)
        self._train_dataset = HIDTorchDataset(list(train_data))
        self._val_dataset = HIDTorchDataset(list(val_data))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Peak classification (standalone)
# ---------------------------------------------------------------------------


class PeakTorchDataset(Dataset):
    """Wraps a list of ExtractedPeaks as a PyTorch Dataset.

    Each item returns ``(peak_tensor, marker_idx, target)`` for the
    peak classifier.
    """

    def __init__(
        self,
        peaks: Sequence,
        label_to_idx: dict[str, int]
    ) -> None:
        self.peaks = list(peaks)
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.peaks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        peak = self.peaks[idx]

        # Peak data: (C, W) float32
        x = torch.tensor(peak.data, dtype=torch.float32)

        # Marker embedding index
        marker_idx = torch.tensor(peak.marker_index, dtype=torch.long)

        # Target label
        label = peak.label
        target = torch.tensor(self.label_to_idx.get(label, 0), dtype=torch.long)

        return x, marker_idx, target


class PeakDataModule(L.LightningDataModule):
    """Lightning DataModule for standalone peak classification.

    Wraps a :class:`~dnanet.data.peak_dataset.PeakWindowDataset` into
    train/val splits and DataLoaders.

    Args:
        dataset: A loaded PeakWindowDataset.
        batch_size: Batch size.
        val_fraction: Fraction for validation.
        num_workers: DataLoader workers.
        seed: Random seed.
    """

    def __init__(
        self,
        dataset: InMemoryDataset,
        batch_size: int = 16,
        val_fraction: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed


        self._label_to_idx = {
            label: idx for idx, label in enumerate(StrategyRegistry.get_dataset_strategy().get_annotation_classes())
        }

        self._train_dataset: PeakTorchDataset | None = None
        self._val_dataset: PeakTorchDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_dataset is not None:
            return

        train_data, val_data = self._dataset.split(
            fraction=1.0 - self.val_fraction,
            seed=self.seed,
        )
        self._train_dataset = PeakTorchDataset(
            list(train_data),
            self._label_to_idx,
        )
        self._val_dataset = PeakTorchDataset(
            list(val_data),
            self._label_to_idx,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# PeakNet (full images + peak windows)
# ---------------------------------------------------------------------------


class PeakNetTorchDataset(Dataset):
    """Wraps HIDImages for PeakNet training.

    Each item returns a dict with the full image, its extracted peaks,
    and the segmentation target. The custom :func:`peaknet_collate_fn`
    assembles these into the batched tensors PeakNet expects.
    """

    def __init__(
        self,
        images: Sequence[HIDImage],
        threshold: float = 40,
        window_size: int = 120,
        include_max_pool_dyes: bool = False,
    ) -> None:
        self.images = list(images)
        self.threshold = threshold
        self.window_size = window_size
        self.include_max_pool_dyes = include_max_pool_dyes

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from dnanet.data.preprocessing.peak_extraction import extract_peaks_torch

        image = self.images[idx]
        data = image.data

        # Full image: (D, L, 1) → (D, L)
        if data.ndim == 3:
            data_2d = data[:, :, 0]
        else:
            data_2d = data

        full_image = torch.tensor(data_2d, dtype=torch.float32)

        # Extract peaks as tensors (on CPU)
        peak_windows, marker_idxs, peak_centers = extract_peaks_torch(
            image,
            device='cpu',
            threshold=self.threshold,
            window_size=self.window_size,
            include_max_pool_dyes=self.include_max_pool_dyes,
        )

        # Target: per-position annotation (D, L)
        if image.annotation is not None:
            ann = image.annotation.data
            if ann.ndim == 3:
                ann = ann[:, :, 0]
            target = torch.tensor(ann, dtype=torch.long)
        else:
            target = torch.zeros_like(full_image, dtype=torch.long)

        return {
            'full_image': full_image,  # (D, L)
            'peak_windows': peak_windows,  # (P, C, W)
            'marker_idxs': marker_idxs,  # (P,)
            'peak_centers': peak_centers,  # (P, 2)
            'target': target,  # (D, L)
            'n_peaks': peak_windows.shape[0],
        }


def peaknet_collate_fn(
    batch: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function for PeakNet batches.

    Stacks full images and concatenates ragged peak data with a
    ``peak_counts`` tensor to track how many peaks came from each image.

    Returns:
        Tuple of:
        - ``full_images``: ``(N, D, L)``
        - ``peak_windows``: ``(P_total, C, W)`` — all peaks concatenated
        - ``marker_idxs``: ``(P_total,)``
        - ``peak_centers``: ``(P_total, 2)``
        - ``peak_counts``: ``(N,)`` — peaks per image
        - ``targets``: ``(N, D, L)``
    """
    full_images = torch.stack([b['full_image'] for b in batch])
    targets = torch.stack([b['target'] for b in batch])

    peak_counts = torch.tensor([b['n_peaks'] for b in batch], dtype=torch.long)

    # Concatenate all peaks
    peak_windows = torch.cat([b['peak_windows'] for b in batch], dim=0)
    marker_idxs = torch.cat([b['marker_idxs'] for b in batch], dim=0)
    peak_centers = torch.cat([b['peak_centers'] for b in batch], dim=0)

    return full_images, peak_windows, marker_idxs, peak_centers, peak_counts, targets


class PeakNetDataModule(L.LightningDataModule):
    """Lightning DataModule for combined PeakNet training.

    Handles the complexity of PeakNet's dual-input format: full images
    for the autoencoder branch, plus extracted peak windows for the
    classifier branch.

    Args:
        dataset: Source HIDDataset with full profiles.
        batch_size: Batch size.
        val_fraction: Validation fraction.
        num_workers: DataLoader workers.
        seed: Random seed.
        peak_threshold: RFU threshold for peak detection.
        window_size: Peak window width.
        include_max_pool_dyes: Add max-pooled dye channel.
    """

    def __init__(
        self,
        dataset: InMemoryDataset,
        batch_size: int = 16,
        val_fraction: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
        peak_threshold: float = 40,
        window_size: int = 120,
        include_max_pool_dyes: bool = False,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed
        self.peak_threshold = peak_threshold
        self.window_size = window_size
        self.include_max_pool_dyes = include_max_pool_dyes

        self._train_dataset: PeakNetTorchDataset | None = None
        self._val_dataset: PeakNetTorchDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_dataset is not None:
            return

        train_data, val_data = self._dataset.split(
            fraction=1.0 - self.val_fraction,
            seed=self.seed,
        )

        self._train_dataset = PeakNetTorchDataset(
            list(train_data),
            threshold=self.peak_threshold,
            window_size=self.window_size,
            include_max_pool_dyes=self.include_max_pool_dyes,
        )
        self._val_dataset = PeakNetTorchDataset(
            list(val_data),
            threshold=self.peak_threshold,
            window_size=self.window_size,
            include_max_pool_dyes=self.include_max_pool_dyes,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=peaknet_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=peaknet_collate_fn,
        )


# ---------------------------------------------------------------------------
# Reconstruction (autoencoder)
# ---------------------------------------------------------------------------


class ReconstructionTorchDataset(Dataset):
    """Wraps HIDImages for autoencoder training.

    Returns ``(preprocessed_input, original_target)`` pairs where the
    input is optionally log-scaled and normalized, and the target is the
    raw signal (for computing loss in original RFU space) or the same
    preprocessed signal (for computing loss in preprocessed space).
    """

    def __init__(
        self,
        images: Sequence[HIDImage],
        log_scale: bool = True,
        max_rfu: int | None = None,
        n_dyes: int = 5,
    ) -> None:
        self.images = list(images)
        self.log_scale = log_scale
        self.max_rfu = max_rfu
        self.n_dyes = n_dyes

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from dnanet.data.preprocessing.scaling import scale_rfu_torch

        image = self.images[idx]
        data = image.data

        if data.ndim == 3:
            data_2d = data[:, :, 0]
        else:
            data_2d = data

        # Select dye channels (exclude size standard)
        data_2d = data_2d[: self.n_dyes]

        raw = torch.tensor(data_2d, dtype=torch.float32)
        preprocessed = scale_rfu_torch(raw, self.log_scale, self.max_rfu)

        # Input is preprocessed, target is also preprocessed (loss in
        # preprocessed space; inverse scaling is done in the module for
        # RFU-space metrics)
        return preprocessed, preprocessed


class ReconstructionDataModule(L.LightningDataModule):
    """Lightning DataModule for autoencoder reconstruction.

    Args:
        dataset: Source HIDDataset.
        batch_size: Batch size.
        val_fraction: Validation fraction.
        num_workers: DataLoader workers.
        seed: Random seed.
        log_scale: Apply log1p scaling.
        max_rfu: Max RFU for normalization.
        n_dyes: Number of dye channels to use.
    """

    def __init__(
        self,
        dataset: InMemoryDataset,
        batch_size: int = 16,
        val_fraction: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
        log_scale: bool = True,
        max_rfu: int | None = None,
        n_dyes: int = 5,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed
        self.log_scale = log_scale
        self.max_rfu = max_rfu
        self.n_dyes = n_dyes

        self._train_dataset: ReconstructionTorchDataset | None = None
        self._val_dataset: ReconstructionTorchDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_dataset is not None:
            return

        train_data, val_data = self._dataset.split(
            fraction=1.0 - self.val_fraction,
            seed=self.seed,
        )

        kw = dict(
            log_scale=self.log_scale,
            max_rfu=self.max_rfu,
            n_dyes=self.n_dyes,
        )
        self._train_dataset = ReconstructionTorchDataset(list(train_data), **kw)
        self._val_dataset = ReconstructionTorchDataset(list(val_data), **kw)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
