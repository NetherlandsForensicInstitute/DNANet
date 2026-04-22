"""PyTorch Lightning DataModules for DNANet.

"""

from __future__ import annotations

from typing import Tuple, Optional

import lightning as L
from torch.utils.data import ConcatDataset, Dataset, DataLoader, default_collate, Subset

from dnanet.data.dataset import TransformableDataset


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
        """Split the dataset and wrap in PyTorch Datasets, and set collate function if available."""
        # TODO: implement setups for different stages
        if self._train_dataset is not None:
            return  # already set up

        if isinstance(self._dataset, ConcatDataset):
            # Demand uniform collate_fn among all datasets of the ConcatDataset.
            collate_fns = [getattr(ds, 'transform', None) for ds in self._dataset.datasets]
            if len(set(collate_fns)) != 1:
                raise ValueError(f"Found multiple collate functions for ConcatDataset ({collate_fns}), but expected same function for all datasets.")
            if collate_fns[0] is not None:
                self._collate_fn = collate_fns[0].collate_fn

            train_data, val_data, test_data = self.apply_concatenated_dataset_splitting(self._dataset)
        else:
            if transform := getattr(self._dataset, 'transform', None):
                self._collate_fn = transform.collate_fn

            train_data, val_data, test_data = self.apply_single_dataset_splitting(self._dataset)

        self._train_dataset = train_data
        self._val_dataset = val_data
        self._test_dataset = test_data

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def apply_single_dataset_splitting(self, dataset: Dataset | TransformableDataset) -> Tuple[Subset, Optional[Subset], Optional[Subset]]:
        """Apply splitting for a single dataset using its strategy."""
        train_fraction = 1.0 - self.val_fraction - self.test_fraction

        val_data, test_data = None, None
        if self.test_fraction > 0.0:
            # TODO: provide other splitting kwargs, such as stratify_noc, group_by_replicas
            train_data, val_data, test_data = dataset.dataset_strategy.split(
                dataset,
                fraction=train_fraction,
                test_fraction=self.test_fraction,
                seed=self.seed,
            )
        elif self.val_fraction > 0.0:
            train_data, val_data = dataset.dataset_strategy.split(
                dataset,
                fraction=train_fraction,
                seed=self.seed,
            )
        else:
            train_data = dataset
        return train_data, val_data, test_data

    def apply_concatenated_dataset_splitting(self, dataset: ConcatDataset) -> Tuple[ConcatDataset, Optional[ConcatDataset], Optional[ConcatDataset]]:
        """Apply splitting for every dataset in the concatenated dataset separately and combine the splits."""
        subsets_train, subsets_val, subsets_test = [], [], []
        for ds in dataset.datasets:
            train_data, val_data, test_data = self.apply_single_dataset_splitting(ds)
            subsets_train.append(train_data)
            if val_data:
                subsets_val.append(val_data)
            if test_data:
                subsets_test.append(test_data)
        train_data = ConcatDataset(subsets_train)
        val_data = ConcatDataset(subsets_val) if subsets_val else None
        test_data = ConcatDataset(subsets_test) if subsets_test else None
        return train_data, val_data, test_data

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
