"""Tests for current DNANetDataModule transformer integrations."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dnanet.data.transformer as transformer_module
from dnanet.core.annotation import ScanpointAnnotation
from dnanet.data.datamodule import DNANetDataModule
from dnanet.data.extracted_peak import ExtractedPeak
from dnanet.data.image import HIDImage
from dnanet.data.strategies import PowerPlexFusion6CStrategy
from dnanet.data.strategies.registry import StrategyRegistry
from dnanet.data.transformer import (
    CombinedTransformer,
    PeakClassificationTransformer,
    ReconstructionTransformer,
)


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
    peak_count: int = 3,
) -> HIDImage:
    img = HIDImage(
        path=name,
        scaling_strategy=PowerPlexFusion6CStrategy(),
        load_in_memory=True,
        meta={"peak_count": peak_count},
    )
    img._data = (np.random.rand(num_dyes, signal_length, 1) * 100).astype(np.float32)
    if with_annotation:
        mask = np.zeros((num_dyes, signal_length, 1), dtype=np.int8)
        mask[0, 5:15, 0] = 1
        img.annotation = ScanpointAnnotation(data=mask)
    return img


def _make_mock_peaks(n: int = 8) -> list[ExtractedPeak]:
    return [
        ExtractedPeak(
            data=np.random.rand(1, 120).astype(np.float32),
            dye_index=i % 5,
            peak_center=1000 + i * 25,
            window_size=120,
            peak_height=float(200 + i * 10),
            label="allele" if i % 2 == 0 else "noise",
            marker_name="D3S1358",
            marker_index=i % 28,
        )
        for i in range(n)
    ]

@pytest.fixture
def ppf6c() -> PowerPlexFusion6CStrategy:
    return PowerPlexFusion6CStrategy()

@pytest.fixture
def configured_peak_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        StrategyRegistry,
        "_dataset_strategy",
        SimpleNamespace(get_annotation_classes=lambda: ["noise", "allele"]),
    )


@pytest.fixture
def fake_peak_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    def _extract_peaks_torch(
        image,
        scaling_strategy,
        threshold,
        window_size,
        include_max_pool_dyes,
    ):
        del scaling_strategy, threshold
        peak_count = image.meta["peak_count"]
        channels = 2 if include_max_pool_dyes else 1
        peak_windows = torch.randn(peak_count, channels, window_size)
        marker_idxs = torch.arange(peak_count, dtype=torch.long)
        peak_centers = torch.stack([
            torch.zeros(peak_count, dtype=torch.long),
            torch.arange(peak_count, dtype=torch.long),
        ], dim=1)
        return peak_windows, marker_idxs, peak_centers

    monkeypatch.setattr(transformer_module, "extract_peaks_torch", _extract_peaks_torch)


class TestPeakClassificationTransformer:
    @pytest.mark.usefixtures("configured_peak_registry")
    def test_datamodule_batches_marker_and_target_tensors(
        self,
        ppf6c,
    ) -> None:
        peaks = _make_mock_peaks(10)
        dataset = SplitPreservingDataset(
            peaks,
            transform=PeakClassificationTransformer(
                scaling_strategy=ppf6c,
                include_marker=True,
            ),
        )
        dm = DNANetDataModule(dataset, batch_size=4, val_fraction=0.2, seed=42)
        dm.setup("fit")

        inputs, targets = next(iter(dm.train_dataloader()))
        peak_data, marker_idx = inputs

        assert peak_data.shape[0] <= 4
        assert peak_data.shape[1:] == (1, 120)
        assert marker_idx.shape[0] == peak_data.shape[0]
        assert targets.shape[0] == peak_data.shape[0]
        assert targets.dtype == torch.long

    @pytest.mark.usefixtures("configured_peak_registry")
    def test_datamodule_batches_negative_marker_when_disabled(
        self,
        ppf6c,
    ) -> None:
        peaks = _make_mock_peaks(10)
        dataset = SplitPreservingDataset(
            peaks,
            transform=PeakClassificationTransformer(
                scaling_strategy=ppf6c,
                include_marker=False,
            ),
        )
        dm = DNANetDataModule(dataset, batch_size=4, val_fraction=0.2, seed=42)
        dm.setup("fit")

        inputs, _targets = next(iter(dm.train_dataloader()))
        _peak_data, marker_idx = inputs
        assert torch.all(marker_idx == -1)


class TestReconstructionTransformer:
    def test_datamodule_returns_preprocessed_pairs(self) -> None:
        images = [_make_fake_image(f"reconstruction_{i}.hid") for i in range(10)]
        dataset = SplitPreservingDataset(
            images,
            transform=ReconstructionTransformer(log_scale=True, n_dyes=5),
        )
        dm = DNANetDataModule(dataset, batch_size=4, val_fraction=0.2, seed=42)
        dm.setup("fit")

        x, target = next(iter(dm.train_dataloader()))
        assert x.shape[0] <= 4
        assert x.shape[1:] == (5, 100)
        assert target.shape == x.shape
        assert torch.all(x >= 0)


class TestCombinedTransformer:
    @pytest.mark.usefixtures("fake_peak_extractor")
    def test_datamodule_returns_nested_peaknet_batch(
        self,
    ) -> None:
        images = [
            _make_fake_image(f"combined_{i}.hid", peak_count=3)
            for i in range(10)
        ]
        dataset = SplitPreservingDataset(
            images,
            transform=CombinedTransformer(window_size=120),
        )
        dm = DNANetDataModule(dataset, batch_size=2, val_fraction=0.2, seed=42)
        dm.setup("fit")

        inputs, targets = next(iter(dm.train_dataloader()))
        full_images, peak_windows, marker_idxs, peak_centers, peak_counts = inputs

        assert full_images.shape == (2, 5, 100)
        assert peak_windows.shape == (6, 1, 120)
        assert marker_idxs.shape == (6,)
        assert peak_centers.shape == (6, 2)
        assert peak_counts.tolist() == [3, 3]
        assert targets.shape == (2, 5, 100)
