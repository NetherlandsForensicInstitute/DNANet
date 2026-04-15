"""Tests for the generic DNANetDataModule."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from dnanet.core.annotation import ScanpointAnnotation
from dnanet.data.datamodule import DNANetDataModule
from dnanet.data.image import HIDImage
from dnanet.data.transformer import SegmentationTransformer


class SplitPreservingDataset(Dataset):
    """Test-local dataset double that keeps transforms across splits."""

    def __init__(self, data, transform=None) -> None:
        self._data = list(data)
        self.transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int):
        item = self._data[index]
        if self.transform is not None:
            return self.transform(item)
        return item

    def split(
        self,
        fraction: float,
        seed: int | None = None,
    ) -> tuple["SplitPreservingDataset", "SplitPreservingDataset"]:
        shuffled = random.Random(seed).sample(self._data, len(self._data))
        split_idx = int(len(shuffled) * fraction)
        return (
            SplitPreservingDataset(shuffled[:split_idx], transform=self.transform),
            SplitPreservingDataset(shuffled[split_idx:], transform=self.transform),
        )


def _make_fake_image(
    scaling_strategy,
    dataset_strategy,
    name: str = "fake.hid",
    num_dyes: int = 5,
    signal_length: int = 100,
    with_annotation: bool = True,
) -> HIDImage:
    img = HIDImage(
        path=name,
        scaling_strategy=scaling_strategy,
        dataset_strategy=dataset_strategy,
        load_in_memory=True,
    )
    img._data = np.random.rand(num_dyes, signal_length, 1).astype(np.float32)
    if with_annotation:
        mask = np.zeros((num_dyes, signal_length, 1), dtype=np.int8)
        mask[0, 10:20, 0] = 1
        img.annotation = ScanpointAnnotation(data=mask)
    return img





@pytest.fixture
def segmentation_dataset(ppf6c_kit, nfi_rnd_dataset) -> SplitPreservingDataset:
    images = [_make_fake_image(ppf6c_kit, nfi_rnd_dataset, f"fake_{i}.hid") for i in range(10)]
    return SplitPreservingDataset(images, transform=SegmentationTransformer())


@pytest.fixture
def unlabeled_segmentation_dataset(ppf6c_kit, nfi_rnd_dataset) -> SplitPreservingDataset:
    images = [
        _make_fake_image(ppf6c_kit, nfi_rnd_dataset, f"fake_{i}.hid", with_annotation=False)
        for i in range(10)
    ]
    return SplitPreservingDataset(images, transform=SegmentationTransformer())


class TestDNANetDataModule:

    def test_segmentation_transform_batches_tensor_pairs(
        self,
        segmentation_dataset: SplitPreservingDataset,
    ) -> None:
        dm = DNANetDataModule(segmentation_dataset, batch_size=4, val_fraction=0.2, seed=42)
        dm.setup("fit")
        x, y = next(iter(dm.train_dataloader()))
        assert x.shape[0] <= 4
        assert x.shape[1:] == (5, 100, 1)
        assert y.shape == x.shape

    def test_segmentation_transform_zeros_targets_without_annotations(
        self,
        unlabeled_segmentation_dataset: SplitPreservingDataset,
    ) -> None:
        dm = DNANetDataModule(
            unlabeled_segmentation_dataset,
            batch_size=4,
            val_fraction=0.2,
            seed=42,
        )
        dm.setup("fit")
        _, y = next(iter(dm.train_dataloader()))
        assert torch.all(y == 0)
