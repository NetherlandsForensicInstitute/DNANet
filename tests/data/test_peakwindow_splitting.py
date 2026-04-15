# 8 replicas × 3 samples each, 24 samples total, 2 NoC values (2 and 5).
# Using 2 classes lets StratifiedGroupKFold/StratifiedKFold succeed even with
# small val folds (e.g. fraction=0.75 → n_splits=4 → 2 replicas in val ≥ 2 classes).
from unittest.mock import MagicMock

import pytest

from dnanet.data.hid_dataset import HIDDataset
from dnanet.data.peak_dataset import PeakWindowDataset
from dnanet.data.strategies.datasets.nfi_rnd import NFIRnDStrategy


STEMS = [
    'Mixture dataset 1/Injx/1A2_A01_01',
    'Mixture dataset 1/Injx/1A2_A01_02',
    'Mixture dataset 1/Injx/1A2_A02_01',  # replica 1A2, NoC=2
    'Mixture dataset 1/Injx/2A2_A01_01',
    'Mixture dataset 2/Injx/2A2_A01_02',
    'Mixture dataset 2/Injx/2A2_A02_01',  # replica 2A2, NoC=2
    'Mixture dataset 2/Injx/3A2_A01_01',
    'Mixture dataset 2/Injx/3A2_A01_02',
    'Mixture dataset 3/Injx/3A2_A02_01',  # replica 3A2, NoC=2
    'Mixture dataset 3/Injx/4A2_A01_01',
    'Mixture dataset 3/Injx/4A2_A01_02',
    'Mixture dataset 3/Injx/4A2_A02_01',  # replica 4A2, NoC=2
    'Mixture dataset 4/Injx/1B5_A01_01',
    'Mixture dataset 4/Injx/1B5_A01_02',
    'Mixture dataset 4/Injx/1B5_A02_01',  # replica 1B5, NoC=5
    'Mixture dataset 4/Injx/2B5_A01_01',
    'Mixture dataset 5/Injx/2B5_A01_02',
    'Mixture dataset 5/Injx/2B5_A02_01',  # replica 2B5, NoC=5
    'Mixture dataset 5/Injx/3B5_A01_01',
    'Mixture dataset 5/Injx/3B5_A01_02',
    'Mixture dataset 6/Injx/3B5_A02_01',  # replica 3B5, NoC=5
    'Mixture dataset 6/Injx/4B5_A01_01',
    'Mixture dataset 6/Injx/4B5_A01_02',
    'Mixture dataset 6/Injx/4B5_A02_01',  # replica 4B5, NoC=5
]


def _fake_img(stem: str):
    img = MagicMock()
    img.path.stem = stem.split('/')[-1]
    img.path.absolute = lambda: stem
    return img


def _fake_dataset(stems: list[str]):
    ds = MagicMock()
    ds.images = [_fake_img(s) for s in stems]
    ds.__len__ = MagicMock(return_value=len(stems))
    return PeakWindowDataset(images=ds.images, dataset_strategy=ds.dataset_strategy)


class TestPeakWindowSplit:

    def test_fractional_split(self):
        ds = _fake_dataset(STEMS)
        splits: tuple[PeakWindowDataset] = NFIRnDStrategy.split(ds, fraction=0.84)

        assert len(splits[0].images) == 20
        assert len(splits[1].images) == 4

    def test_kfold_split(self):
        ds = _fake_dataset(STEMS)
        folds: list[tuple[PeakWindowDataset, ...]] = NFIRnDStrategy.split(ds, k_folds=3)

        assert len(folds) == 3
        assert len(folds[0][0].images) == 16
        assert len(folds[0][1].images) == 6
