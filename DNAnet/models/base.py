from abc import ABC, abstractmethod
from typing import Sequence

import torch

from DNAnet.data.data_models.base import Image, InMemoryDataset
from DNAnet.models.prediction import Prediction
from DNAnet.typing import PathLike


TORCH_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Model(ABC):

    @abstractmethod
    def predict(self, image: Image) -> Prediction:
        """
        Make a model prediction for a single image.
        """
        raise NotImplementedError

    def predict_batch(self, batch: Sequence[Image]) -> Sequence[Prediction]:
        """
        Make model predictions for multiple images.
        """
        return list(map(self.predict, batch))

    @abstractmethod
    def save(self, model_dir: PathLike):
        """
        Load any file(s) from the specified `model_dir`.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, model_dir: PathLike):
        """
        Save any file(s) to the specified `model_dir`.
        """
        raise NotImplementedError


class TrainableModel(Model, ABC):

    @abstractmethod
    def fit(self, dataset: InMemoryDataset[Image], **kwargs):
        """
        Fit the model on the dataset.
        """
        raise NotImplementedError
