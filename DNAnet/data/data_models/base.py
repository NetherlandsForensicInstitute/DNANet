import random
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from DNAnet.data.data_models import Annotation
from DNAnet.data.split import split_data_in_k_folds


class Image(ABC):
    """
    Abstract base class for an image, that holds data, an annotation and meta information.
    """

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """
        The raw data content of the image.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def annotation(self) -> Annotation:
        """
        A ground truth annotation, holding for instance a class label or segmentation image.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def meta(self) -> Dict[str, Any]:
        """
        Meta information about the image.
        """
        raise NotImplementedError


class InMemoryDataset(Sequence[Image]):
    """
    A base class for a dataset holding images. The dataset should at least be
    iterable and splittable.
    """
    def __init__(self, shuffle: Optional[bool] = False):
        self.shuffle = shuffle
        self._data = []

    def __len__(self) -> int:
        """
        The number of images that the dataset holds.
        """
        return len(self._data)

    def __iter__(self) -> Iterator[Image]:
        """
        An iterator method to allow iteration over the dataset and apply shuffling if desired.
        """
        if self.shuffle:
            yield from random.Random().sample(self._data, len(self._data))
        else:
            yield from self._data

    def __getitem__(self, index: int) -> Image:
        return self._data[index]

    def split(self, fraction: float, seed: Optional[float] = None) \
            -> Tuple['SimpleDataset', 'SimpleDataset']:
        """
        Split the data of the dataset randomly into two new datasets.
        :param fraction: The fraction of the dataset that should be included in the first dataset.
        :param seed: An optional seed to make the split deterministic.
        :return: Two datasets each holding a random subset of the original data.
        """
        if not 0 < fraction < 1:
            raise ValueError(f"Fraction should be between 0 and 1, got {fraction}.")

        random.seed(seed)
        shuffled_data = random.sample(self._data, len(self._data))
        split_idx = int(len(shuffled_data) * fraction)

        dataset1 = SimpleDataset(data=shuffled_data[:split_idx], shuffle=self.shuffle)
        dataset2 = SimpleDataset(data=shuffled_data[split_idx:], shuffle=self.shuffle)
        return dataset1, dataset2

    def split_k_fold(self, n_folds: int, seed: Optional[float] = None) -> \
            List[Tuple['SimpleDataset', 'SimpleDataset']]:
        """
        Split the dataset images into 'n_folds' train/test folds.
        """
        images_per_fold = split_data_in_k_folds(self._data, n_folds, seed)
        train_test_folds = self._make_train_test_splits(images_per_fold)
        return train_test_folds

    def _make_train_test_splits(self, folds: List[List[Image]]) \
            -> List[Tuple['SimpleDataset', 'SimpleDataset']]:
        """
        Make train and test sets from each separate fold.
        """
        for images_per_fold in folds:
            random.shuffle(images_per_fold)

        train_test_folds = []
        for i in range(len(folds)):
            test_set = folds[i]
            train_set = chain.from_iterable(folds[:i] + folds[i + 1:])
            train_test_folds.append((SimpleDataset(data=list(train_set), shuffle=self.shuffle),
                                     SimpleDataset(data=test_set, shuffle=self.shuffle)))
        return train_test_folds


class SimpleDataset(InMemoryDataset):
    """
    A simple dataset class that can handle a sequence of images directly, without relying on
    internal parsing first.
    """
    def __init__(self, data: Sequence[Image], shuffle: Optional[bool] = False):
        super().__init__(shuffle)
        self._data = data
