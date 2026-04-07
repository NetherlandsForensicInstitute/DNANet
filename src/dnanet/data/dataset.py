import abc

from dnanet.data.transformer import TransformDataCallable


class TransformableDataset(abc.ABC):

    @property
    @abc.abstractmethod
    def transform(self) -> TransformDataCallable | None:
        raise NotImplementedError
