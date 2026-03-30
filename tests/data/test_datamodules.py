"""Tests for DataModule variants."""

import numpy as np
import torch
import pytest

from dnanet.data.dataset import SimpleDataset
from dnanet.data.datamodule import (
    PeakDataModule,
    HIDTorchDataset,
    DNANetDataModule,
    PeakTorchDataset,
    ReconstructionDataModule,
    ReconstructionTorchDataset,
    peaknet_collate_fn,
)
from dnanet.data.extracted_peak import ExtractedPeak


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------

class MockAnnotation:
    def __init__(self, image):
        self.image = image


class MockImage:
    """Minimal HIDImage-like."""

    def __init__(self, n_dyes=5, length=4096, with_annotation=True):
        self._data = np.random.rand(n_dyes, length, 1).astype(np.float32) * 100
        if with_annotation:
            self.annotation = MockAnnotation(
                np.random.randint(0, 2, (n_dyes, length, 1)).astype(np.float32)
            )
        else:
            self.annotation = None
        self.path = type("P", (), {"stem": "test_profile"})()
        self._panel = None

    @property
    def data(self):
        return self._data


def _make_mock_dataset(n=5, **kwargs) -> SimpleDataset:
    return SimpleDataset(data=[MockImage(**kwargs) for _ in range(n)])


def _make_mock_peaks(n=20) -> list[ExtractedPeak]:
    return [
        ExtractedPeak(
            data=np.random.rand(1, 120).astype(np.float32),
            dye_index=i % 5,
            peak_center=1000 + i * 50,
            window_size=120,
            peak_height=float(200 + i * 10),
            label="allele" if i % 2 == 0 else "noise",
            marker_index=i % 28,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# HIDTorchDataset + DNANetDataModule (segmentation)
# ---------------------------------------------------------------------------

class TestHIDTorchDataset:

    def test_returns_correct_shapes(self):
        images = [MockImage(n_dyes=5, length=100)]
        ds = HIDTorchDataset(images)
        x, y = ds[0]
        assert x.shape == (1, 5, 100)
        assert y.shape == (1, 5, 100)

    def test_no_annotation_gives_zeros(self):
        images = [MockImage(with_annotation=False, length=100)]
        ds = HIDTorchDataset(images)
        x, y = ds[0]
        assert y.sum() == 0


class TestDNANetDataModule:

    def test_setup_creates_splits(self):
        dataset = _make_mock_dataset(10, length=100)
        dm = DNANetDataModule(dataset, batch_size=2, val_fraction=0.2, seed=42)
        dm.setup()
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        assert len(dm._train_dataset) + len(dm._val_dataset) == 10

    def test_dataloaders_iterate(self):
        dataset = _make_mock_dataset(10, length=100)
        dm = DNANetDataModule(dataset, batch_size=4, val_fraction=0.2, seed=42)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert len(batch) == 2  # (x, y)


# ---------------------------------------------------------------------------
# PeakTorchDataset + PeakDataModule (classification)
# ---------------------------------------------------------------------------

class TestPeakTorchDataset:

    def test_returns_triplet(self):
        peaks = _make_mock_peaks(5)
        ds = PeakTorchDataset(peaks)
        x, marker, target = ds[0]
        assert x.shape == (1, 120)
        assert marker.dim() == 0  # scalar
        assert target.dim() == 0

    def test_label_mapping(self):
        peaks = _make_mock_peaks(4)
        peaks[0].label = "allele"
        peaks[1].label = "noise"
        ds = PeakTorchDataset(peaks, label_to_idx={"noise": 0, "allele": 1})
        _, _, t0 = ds[0]
        _, _, t1 = ds[1]
        assert t0.item() == 1  # allele
        assert t1.item() == 0  # noise


class TestPeakDataModule:

    def test_setup_and_iterate(self):
        peaks = _make_mock_peaks(20)
        ds = SimpleDataset(data=peaks)
        dm = PeakDataModule(ds, batch_size=4, val_fraction=0.2, seed=42)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert len(batch) == 3  # (x, marker, target)


# ---------------------------------------------------------------------------
# PeakNet collate function
# ---------------------------------------------------------------------------

class TestPeakNetCollation:

    def test_collate_fn_shapes(self):
        batch = [
            {
                "full_image": torch.randn(5, 100),
                "peak_windows": torch.randn(3, 1, 120),
                "marker_idxs": torch.zeros(3, dtype=torch.long),
                "peak_centers": torch.zeros(3, 2, dtype=torch.long),
                "target": torch.zeros(5, 100, dtype=torch.long),
                "n_peaks": 3,
            },
            {
                "full_image": torch.randn(5, 100),
                "peak_windows": torch.randn(5, 1, 120),
                "marker_idxs": torch.zeros(5, dtype=torch.long),
                "peak_centers": torch.zeros(5, 2, dtype=torch.long),
                "target": torch.zeros(5, 100, dtype=torch.long),
                "n_peaks": 5,
            },
        ]

        full, windows, markers, centers, counts, targets = peaknet_collate_fn(batch)

        assert full.shape == (2, 5, 100)
        assert windows.shape == (8, 1, 120)  # 3 + 5
        assert markers.shape == (8,)
        assert centers.shape == (8, 2)
        assert counts.tolist() == [3, 5]
        assert targets.shape == (2, 5, 100)


# ---------------------------------------------------------------------------
# Reconstruction DataModule
# ---------------------------------------------------------------------------

class TestReconstructionDataModule:

    def test_returns_preprocessed_pairs(self):
        images = [MockImage(n_dyes=5, length=100)]
        ds = ReconstructionTorchDataset(images, log_scale=True, n_dyes=5)
        x, target = ds[0]
        assert x.shape == (5, 100)
        assert target.shape == (5, 100)
        assert (x >= 0).all()

    def test_setup_and_iterate(self):
        dataset = _make_mock_dataset(10, length=100)
        dm = ReconstructionDataModule(
            dataset, batch_size=4, val_fraction=0.2, seed=42, n_dyes=5,
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert len(batch) == 2
