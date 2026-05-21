"""PyTorch Lightning DataModules for DNANet."""

from __future__ import annotations

import lightning as L
from torch.utils.data import Dataset, DataLoader, ConcatDataset, default_collate

from dnanet.data.dataset import TransformableDataset
from dnanet.data.splitting import dataset_splitter


class DNANetDataModule(L.LightningDataModule):
    """Lightning DataModule for DNA profiles.

    Args:
        dataset: A loaded dataset (e.g. HIDDataset or torch ConcatDataset).
        batch_size: Batch size for DataLoaders.
        num_workers: Number of DataLoader workers.
        dataset_strategy: Optional override for the dataset's splitting strategy.
        **split_kwargs: Passed to dataset_splitter. Supports val_fraction,
            test_fraction, seed, and any strategy-specific kwargs (e.g.
            genotype_aware, stratify_noc). K-fold is not supported here.
    """

    def __init__(
        self,
        dataset: TransformableDataset,
        batch_size: int = 16,
        num_workers: int = 0,
        shuffle_train: bool = True,
        **split_kwargs,
    ) -> None:
        super().__init__()

        self._dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._split_kwargs = split_kwargs
        self.shuffle_train = shuffle_train

        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None
        self._test_dataset: Dataset | None = None
        self._collate_fn = default_collate

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Split the dataset into train/val/test subsets."""
        if self._train_dataset is not None:
            return

        if self._split_kwargs.get('k_folds') is not None:
            raise ValueError(
                'K-fold splitting is not supported in DNANetDataModule. '
                'Use dataset_splitter() directly and iterate over folds.'
            )

        if isinstance(self._dataset, ConcatDataset):
            collate_fns = [getattr(ds, 'transform', None) for ds in self._dataset.datasets]
            if len(set(collate_fns)) != 1:
                raise ValueError(
                    f"Found multiple collate functions for ConcatDataset ({collate_fns}), "
                    "but expected same function for all datasets."
                )
            if collate_fns[0] is not None:
                self._collate_fn = collate_fns[0].collate_fn
        elif hasattr(self._dataset, 'transform') and self._dataset.transform is not None:
            self._collate_fn = self._dataset.transform.collate_fn

        train, val, test = dataset_splitter(self._dataset, **self._split_kwargs)
        if stage == "test" and self._split_kwargs == {}:
            # If no splitting arguments are provided and splitting is prepared for testing, we use
            # the entire dataset that is stored in `train` for testing.
            test, train, val = train, None, None
        self._train_dataset = train  # type: ignore[assignment]
        self._val_dataset = val  # type: ignore[assignment]
        self._test_dataset = test  # type: ignore[assignment]

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None, "call setup() before train_dataloader()"
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, "call setup() with val_fraction > 0 before val_dataloader()"
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            persistent_workers=self.num_workers > 0,
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
