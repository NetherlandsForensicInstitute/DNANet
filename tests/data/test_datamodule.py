"""Tests for the generic DNANetDataModule."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from dnanet.core.annotation import ScanpointAnnotation
from dnanet.data.datamodule import DNANetDataModule
from dnanet.data.dataset import SimpleDataset
from dnanet.data.image import HIDImage
from dnanet.data.transformer import SegmentationTransformer


class SplitPreservingDataset:
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
    name: str = "fake.hid",
    num_dyes: int = 5,
    signal_length: int = 100,
    with_annotation: bool = True,
) -> HIDImage:
    img = HIDImage(path=name, load_in_memory=True)
    img._data = np.random.rand(num_dyes, signal_length, 1).astype(np.float32)
    if with_annotation:
        mask = np.zeros((num_dyes, signal_length, 1), dtype=np.int8)
        mask[0, 10:20, 0] = 1
        img.annotation = ScanpointAnnotation(data=mask)
    return img


@pytest.fixture
def plain_dataset() -> SimpleDataset:
    tensors = [torch.tensor([float(i)], dtype=torch.float32) for i in range(10)]
    return SimpleDataset(data=tensors)


@pytest.fixture
def segmentation_dataset() -> SplitPreservingDataset:
    images = [_make_fake_image(f"fake_{i}.hid") for i in range(10)]
    return SplitPreservingDataset(images, transform=SegmentationTransformer())


@pytest.fixture
def unlabeled_segmentation_dataset() -> SplitPreservingDataset:
    images = [
        _make_fake_image(f"fake_{i}.hid", with_annotation=False)
        for i in range(10)
    ]
    return SplitPreservingDataset(images, transform=SegmentationTransformer())


class TestDNANetDataModule:
    def test_setup_splits_data(self, plain_dataset: SimpleDataset) -> None:
        dm = DNANetDataModule(plain_dataset, batch_size=2, val_fraction=0.2, seed=42)
        dm.setup("fit")
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        assert len(dm._train_dataset) + len(dm._val_dataset) == len(plain_dataset)

    def test_setup_idempotent(self, plain_dataset: SimpleDataset) -> None:
        dm = DNANetDataModule(plain_dataset, batch_size=2, val_fraction=0.2, seed=42)
        dm.setup("fit")
        train = dm._train_dataset
        dm.setup("fit")
        assert dm._train_dataset is train

    def test_plain_dataset_uses_default_collate(self, plain_dataset: SimpleDataset) -> None:
        dm = DNANetDataModule(plain_dataset, batch_size=4, val_fraction=0.2, seed=42)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, torch.Tensor)
        assert batch.ndim == 2
        assert batch.shape[0] <= 4

    def test_train_and_val_dataloaders_exist(self, plain_dataset: SimpleDataset) -> None:
        dm = DNANetDataModule(plain_dataset, batch_size=2, val_fraction=0.2, seed=42)
        dm.setup("fit")
        assert isinstance(dm.train_dataloader(), torch.utils.data.DataLoader)
        assert isinstance(dm.val_dataloader(), torch.utils.data.DataLoader)

    def test_segmentation_transform_batches_tensor_pairs(
        self,
        segmentation_dataset: SplitPreservingDataset,
    ) -> None:
        dm = DNANetDataModule(segmentation_dataset, batch_size=4, val_fraction=0.2, seed=42)
        dm.setup("fit")
        x, y = next(iter(dm.train_dataloader()))
        assert x.shape[0] <= 4
        assert x.shape[1:] == (1, 5, 100)
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
