"""Dataset containers for collections of HIDImages.

Design pattern: **Iterator + Sequence**
    ``InMemoryDataset`` implements Python's ``Sequence`` protocol (``__len__``,
    ``__getitem__``, ``__iter__``), making it usable with for-loops, list
    comprehensions, and PyTorch DataLoaders.

    ``SimpleDataset`` is a lightweight wrapper for any list of images — used
    as the return type of splits and folds so you don't carry the heavy
    loading logic of ``HIDDataset`` into train/test subsets.

Design pattern: **Template Method** (for splitting)
    ``InMemoryDataset.split()`` and ``split_k_fold()`` define the default
    splitting algorithm. ``HIDDataset`` overrides these to add replica-aware
    and NoC-balanced splitting — the rest of the pipeline doesn't need to know.
"""

from __future__ import annotations

import random
from itertools import chain
from typing import Iterator, Sequence

from dnanet.data.splitting import split_data_in_k_folds


class InMemoryDataset(Sequence):
    """Base class for in-memory datasets.

    Holds a list of items (typically ``HIDImage`` instances) and provides
    splitting, iteration, and random-access indexing.

    Args:
        shuffle: If True, iteration order is randomized each time.
    """

    def __init__(self, shuffle: bool = False) -> None:
        self.shuffle = shuffle
        self._data: list = []

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int):
        return self._data[index]

    def __iter__(self) -> Iterator:
        if self.shuffle:
            yield from random.sample(self._data, len(self._data))
        else:
            yield from self._data

    def split(
        self, fraction: float, seed: float | None = None
    ) -> tuple[SimpleDataset, SimpleDataset]:
        """Randomly split into two datasets.

        Args:
            fraction: Fraction of items in the first dataset.
            seed: Random seed for reproducibility.

        Returns:
            Two ``SimpleDataset`` instances.
        """
        if not 0 < fraction < 1:
            raise ValueError(f"Fraction must be between 0 and 1, got {fraction}")

        random.seed(seed)
        shuffled = random.sample(self._data, len(self._data))
        split_idx = int(len(shuffled) * fraction)

        return (
            SimpleDataset(data=shuffled[:split_idx], shuffle=self.shuffle),
            SimpleDataset(data=shuffled[split_idx:], shuffle=self.shuffle),
        )

    def split_k_fold(
        self, n_folds: int, seed: float | None = None
    ) -> list[tuple[SimpleDataset, SimpleDataset]]:
        """Split into k train/test folds.

        Args:
            n_folds: Number of folds.
            seed: Random seed for reproducibility.

        Returns:
            List of ``(train_dataset, test_dataset)`` tuples.
        """
        folds = split_data_in_k_folds(self._data, n_folds, seed)

        result = []
        for i in range(len(folds)):
            test = folds[i]
            train = list(chain.from_iterable(folds[:i] + folds[i + 1 :]))
            random.shuffle(train)
            result.append((
                SimpleDataset(data=train, shuffle=self.shuffle),
                SimpleDataset(data=test, shuffle=self.shuffle),
            ))
        return result


class SimpleDataset(InMemoryDataset):
    """A lightweight dataset wrapping an existing list of items.

    Used as the return type of ``split()`` and ``split_k_fold()`` to avoid
    re-running heavy data loading logic.
    """

    def __init__(self, data: Sequence, shuffle: bool = False) -> None:
        super().__init__(shuffle)
        self._data = list(data)
