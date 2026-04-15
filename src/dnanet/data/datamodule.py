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
        val_fraction: Fraction of total data to use for validation.
        test_fraction: Fraction of total data to hold out as a test set.
            When 0.0 (default), no test split is created.
        num_workers: Number of DataLoader workers.
        seed: Random seed for splitting.
    """

    def __init__(
        self,
        dataset: TransformableDataset,
        batch_size: int = 16,
        val_fraction: float = 0.2,
        test_fraction: float = 0.0,
        num_workers: int = 0,
        seed: int | None = None,
        stratify_noc: bool = False,
        group_by_replica: bool = False,
    ) -> None:
        super().__init__()
        if val_fraction + test_fraction >= 1.0:
            raise ValueError(
                f'val_fraction ({val_fraction}) + test_fraction ({test_fraction}) must be < 1.0'
            )
        if val_fraction <= 0.0:
            raise ValueError(f'val_fraction must be > 0, got {val_fraction}')
        if test_fraction < 0.0:
            raise ValueError(f'test_fraction must be >= 0, got {test_fraction}')

        self._dataset = dataset
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.num_workers = num_workers
        self.seed = seed
        self.stratify_noc = stratify_noc
        self.group_by_replica = group_by_replica

        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None
        self._test_dataset: Dataset | None = None
        self._collate_fn = default_collate

    def setup(self, stage: str | None = None) -> None:
        """Split the dataset and wrap in PyTorch Datasets."""
        if self._train_dataset is not None:
            return  # already set up

        from dnanet.data.strategies.registry import StrategyRegistry

        strategy = StrategyRegistry.get_dataset_strategy()
        train_fraction = 1.0 - self.val_fraction - self.test_fraction

        if self.test_fraction > 0.0:
            train_data, val_data, test_data = strategy.split(
                self._dataset,
                fraction=train_fraction,
                test_fraction=self.test_fraction,
                seed=self.seed,
            )
            self._test_dataset = test_data
        elif self.val_fraction > 0.0:
            train_data, val_data = strategy.split(
                self._dataset,
                fraction=train_fraction,
                seed=self.seed,
            )
        else:
            train_data = self._dataset
            val_data = None

        self._train_dataset = train_data
        self._val_dataset = val_data

        if hasattr(self._dataset, 'transform') and self._dataset.transform is not None:
            self._collate_fn = self._dataset.transform.collate_fn

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

    def test_dataloader(self) -> DataLoader:
        if self._test_dataset is None:
            raise RuntimeError(
                'test_dataloader() called but no test split was created. '
                'Pass test_fraction > 0 to DNANetDataModule.'
            )
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
