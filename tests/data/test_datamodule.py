"""Tests for DNANetDataModule and HIDTorchDataset."""

import numpy as np
import torch
import pytest

from dnanet.data.image import HIDImage
from dnanet.data.dataset import SimpleDataset
from dnanet.core.annotation import Annotation
from dnanet.data.datamodule import HIDTorchDataset, DNANetDataModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_image(
    name: str = "fake.hid",
    num_dyes: int = 5,
    signal_length: int = 100,
    with_annotation: bool = True,
) -> HIDImage:
    """Build an HIDImage with pre-populated data (no I/O)."""
    img = HIDImage(path=name, use_cache=True)
    img._data = np.random.rand(num_dyes, signal_length, 1).astype(np.float32)
    if with_annotation:
        mask = np.zeros((num_dyes, signal_length, 1), dtype=np.int8)
        mask[0, 10:20, 0] = 1  # fake allele region on dye 0
        img._annotation = Annotation(image=mask)
    return img


@pytest.fixture
def fake_images():
    """List of 10 fake HIDImages for testing."""
    return [_make_fake_image(f"fake_{i}.hid") for i in range(10)]


@pytest.fixture
def fake_dataset(fake_images):
    """A SimpleDataset wrapping fake images."""
    return SimpleDataset(data=fake_images)


# ---------------------------------------------------------------------------
# HIDTorchDataset
# ---------------------------------------------------------------------------

class TestHIDTorchDataset:
    def test_len(self, fake_images):
        ds = HIDTorchDataset(fake_images)
        assert len(ds) == 10

    def test_getitem_returns_tensor_tuple(self, fake_images):
        ds = HIDTorchDataset(fake_images)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_input_shape(self, fake_images):
        ds = HIDTorchDataset(fake_images)
        x, _ = ds[0]
        # (dyes, signal_length, 1) transposed to (1, dyes, signal_length)
        assert x.shape == (1, 5, 100)

    def test_target_shape_with_annotation(self, fake_images):
        ds = HIDTorchDataset(fake_images)
        _, y = ds[0]
        assert y.shape == (1, 5, 100)

    def test_target_zeros_without_annotation(self):
        img = _make_fake_image(with_annotation=False)
        ds = HIDTorchDataset([img])
        _, y = ds[0]
        assert y.shape == (1, 5, 100)
        assert torch.all(y == 0)

    def test_dtype_float32(self, fake_images):
        ds = HIDTorchDataset(fake_images)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_annotation_values_preserved(self, fake_images):
        ds = HIDTorchDataset(fake_images)
        _, y = ds[0]
        # The mask has 1s at dye 0, positions 10-19
        assert y[0, 0, 10:20].sum() == 10
        assert y[0, 0, :10].sum() == 0


# ---------------------------------------------------------------------------
# DNANetDataModule
# ---------------------------------------------------------------------------

class TestDNANetDataModule:
    def test_setup_splits_data(self, fake_dataset):
        dm = DNANetDataModule(fake_dataset, batch_size=2, val_fraction=0.2, seed=42)
        dm.setup("fit")
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        assert len(dm._train_dataset) + len(dm._val_dataset) == 10

    def test_setup_idempotent(self, fake_dataset):
        dm = DNANetDataModule(fake_dataset, batch_size=2, val_fraction=0.2, seed=42)
        dm.setup("fit")
        train = dm._train_dataset
        dm.setup("fit")  # second call should be no-op
        assert dm._train_dataset is train

    def test_train_dataloader_returns_loader(self, fake_dataset):
        dm = DNANetDataModule(fake_dataset, batch_size=2, val_fraction=0.2, seed=42)
        dm.setup("fit")
        loader = dm.train_dataloader()
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_val_dataloader_returns_loader(self, fake_dataset):
        dm = DNANetDataModule(fake_dataset, batch_size=2, val_fraction=0.2, seed=42)
        dm.setup("fit")
        loader = dm.val_dataloader()
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_batch_size_respected(self, fake_dataset):
        dm = DNANetDataModule(fake_dataset, batch_size=4, val_fraction=0.2, seed=42)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        x, y = batch
        assert x.shape[0] <= 4

    def test_batch_tensor_shapes(self, fake_dataset):
        dm = DNANetDataModule(fake_dataset, batch_size=4, val_fraction=0.2, seed=42)
        dm.setup("fit")
        x, y = next(iter(dm.train_dataloader()))
        # (batch, 1, dyes, signal_length)
        assert x.ndim == 4
        assert x.shape[1] == 1
        assert x.shape[2] == 5
        assert x.shape[3] == 100
        assert y.shape == x.shape
