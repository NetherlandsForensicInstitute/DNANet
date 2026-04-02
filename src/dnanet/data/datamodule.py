"""PyTorch Lightning DataModules for DNANet.

"""

from __future__ import annotations

import lightning as L
from torch.utils.data import Dataset, DataLoader, default_collate


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
        dataset: Dataset,
        batch_size: int = 16,
        val_fraction: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed

        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None
        self._collate_fn = default_collate

    def setup(self, stage: str | None = None) -> None:
        """Split the dataset and wrap in PyTorch Datasets."""
        if self._train_dataset is not None:
            return  # already set up

        train_data, val_data = self._dataset.split(fraction=1.0 - self.val_fraction, seed=self.seed) #Fixme: splitting

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
