"""PyTorch Lightning DataModules for DNANet.

"""

from __future__ import annotations

import lightning as L
from torch.utils.data import Dataset, DataLoader, default_collate

from dnanet.data.dataset import TransformableDataset

from dnanet.data.strategies import StrategyRegistry


class DNANetDataModule(L.LightningDataModule):
    """Lightning DataModule for DNA profiles.

    Args:
        dataset: A loaded dataset (e.g. HIDDataset).
        batch_size: Batch size for DataLoaders.
        val_fraction: Fraction of data to use for validation.
        num_workers: Number of DataLoader workers.
        seed: Random seed for train/val splitting.
    """

    def __init__(
        self,
        dataset: TransformableDataset,
        batch_size: int = 16,
        val_fraction: float = 0.8,
        num_workers: int = 0,
        seed: int = 42,
        stratify_noc: bool = False,
        group_by_replica: bool = False,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed
        self.stratify_noc = stratify_noc
        self.group_by_replica = group_by_replica

        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None
        self._collate_fn = default_collate

    def setup(self, stage: str | None = None) -> None:
        """Split the dataset and wrap in PyTorch Datasets."""
        if self._train_dataset is not None:
            return  # already set up

        dataset_strategy = StrategyRegistry.get_dataset_strategy()
        train_data, val_data = dataset_strategy.split(self._dataset, self.val_fraction, self.seed, stratify_noc=self.stratify_noc, group_by_replica=self.group_by_replica)

        self._train_dataset = train_data
        self._val_dataset = val_data

        if hasattr(self._dataset, 'transform') and self._dataset.transform is not None:
            collate_fn = self._dataset.transform.collate_fn
            self._collate_fn = collate_fn



    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
