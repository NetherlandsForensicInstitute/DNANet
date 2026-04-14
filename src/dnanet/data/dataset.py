import abc
from typing import List

from dnanet.data.image import HIDImage
from dnanet.data.transformer import TransformDataCallable


class TransformableDataset(abc.ABC):
    @property
    @abc.abstractmethod
    def images(self) -> List[HIDImage]:
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def transform(self) -> TransformDataCallable | None:
        raise NotImplementedError
