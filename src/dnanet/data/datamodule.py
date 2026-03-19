"""PyTorch Lightning DataModule for DNANet.

Design pattern: **Adapter**
    ``DNANetDataModule`` adapts DNANet's domain-specific ``InMemoryDataset``
    to PyTorch Lightning's ``LightningDataModule`` interface. It wraps the
    in-memory data into standard ``torch.utils.data.Dataset`` objects and
    provides ``DataLoader`` instances with proper collation.

Design pattern: **Bridge**
    The DataModule bridges between two independent hierarchies:
    1. DNANet data loading (HIDImage, HIDDataset, caching, strategies)
    2. PyTorch training infrastructure (DataLoader, LightningModule, Trainer)

    Neither side needs to know about the other — the DataModule is the
    single point of integration.

Usage:
    The DataModule is instantiated by the Hydra task runner and passed to
    ``lightning.Trainer.fit()``. It handles:
    - ``prepare_data()``: Download from HuggingFace if needed
    - ``setup()``: Load images, split into train/val
    - ``train_dataloader()`` / ``val_dataloader()``: Return DataLoaders
"""

from __future__ import annotations

from typing import Sequence

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dnanet.data.dataset import InMemoryDataset
from dnanet.data.image import HIDImage


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
        x = torch.tensor(
            np.transpose(data, (2, 0, 1)), dtype=torch.float32
        )

        # Target: segmentation mask with same shape
        if image.annotation is not None and image.annotation.image is not None:
            y = torch.tensor(
                np.transpose(image.annotation.image, (2, 0, 1)),
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

        train_data, val_data = self._dataset.split(
            fraction=1.0 - self.val_fraction, seed=self.seed
        )
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
