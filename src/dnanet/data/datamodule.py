"""PyTorch Lightning DataModules for DNANet.

"""

from __future__ import annotations

from typing import Tuple, Optional

import lightning as L
from torch.utils.data import ConcatDataset, Dataset, DataLoader, default_collate, Subset

from dnanet.data.dataset import TransformableDataset
from dnanet.data.strategies.datasets.dataset import DatasetStrategy





class DNANetDataModule(L.LightningDataModule):
    """Lightning DataModule for DNA profiles.

    Args:
        dataset: A loaded dataset (e.g. HIDDataset or torch ConcatDataset).
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
        num_workers: int = 0,
        dataset_strategy: DatasetStrategy | None = None
    ) -> None:
        super().__init__()

        self._dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_strategy = dataset_strategy if dataset_strategy else self._dataset.dataset_strategy

        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None
        self._test_dataset: Dataset | None = None
        self._collate_fn = default_collate

    def setup(self, stage: str | None = None, **split_kwargs) -> None:
        """Split the dataset and wrap in PyTorch Datasets."""
        
        # Setup dataset transform as a collate function
        if hasattr(self._dataset, 'transform') and self._dataset.transform is not None:
            self._collate_fn = self._dataset.transform.collate_fn
        
        # Get splitting kwargs
        val_fraction = split_kwargs.get("val_fraction")
        test_fraction = split_kwargs.get("test_fraction")
        seed = split_kwargs.get("seed")
        k_folds = split_kwargs.get("k_folds")
        
        # If no splitting logic is provided, we only set the train loader
        if val_fraction is None and test_fraction is None and k_folds is None:
            self._train_dataset = self._dataset
            return 
        
        if val_fraction + test_fraction >= 1.0:
            raise ValueError(
                f'val_fraction ({val_fraction}) + test_fraction ({test_fraction}) must be < 1.0'
            )
        if val_fraction <= 0.0:
            raise ValueError(f'val_fraction must be > 0, got {val_fraction}')
        if test_fraction < 0.0:
            raise ValueError(f'test_fraction must be >= 0, got {test_fraction}')

        if self._train_dataset is not None:
            return  # already set up

        if isinstance(self._dataset, ConcatDataset):
            # Demand uniform collate_fn among all datasets of the ConcatDataset.
            collate_fns = [getattr(ds, 'transform', None) for ds in self._dataset.datasets]
            if len(set(collate_fns)) != 1:
                raise ValueError(f"Found multiple collate functions for ConcatDataset ({collate_fns}), but expected same function for all datasets.")
            if collate_fns[0] is not None:
                self._collate_fn = collate_fns[0].collate_fn

        train_fraction = 1.0 - val_fraction - test_fraction

        if test_fraction > 0.0:
            train_data, val_data, test_data = self.dataset_strategy.split(
                self._dataset,
                fraction=train_fraction,
                **split_kwargs
            )
            self._test_dataset = test_data
        elif val_fraction > 0.0:
            train_data, val_data = self.dataset_strategy.split(
                self._dataset,
                fraction=train_fraction,
                **split_kwargs
            )
        else:
            if transform := getattr(self._dataset, 'transform', None):
                self._collate_fn = transform.collate_fn

            train_data, val_data, test_data = self.apply_single_dataset_splitting(self._dataset)

        self._train_dataset = train_data
        self._val_dataset = val_data

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
