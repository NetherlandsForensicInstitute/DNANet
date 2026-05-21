import abc
from typing import List

from dnanet.data.image import HIDImage
from dnanet.data.strategies import DatasetStrategy
from dnanet.data.transformer import TransformDataCallable


class TransformableDataset(abc.ABC):  # noqa: D101
    @property
    @abc.abstractmethod
    def images(self) -> List[HIDImage]:
        raise NotImplementedError
    @property
    @abc.abstractmethod
    def transform(self) -> TransformDataCallable | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dataset_strategy(self) -> DatasetStrategy:
        raise NotImplementedError
