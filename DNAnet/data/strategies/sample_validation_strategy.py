from typing import Protocol

from DNAnet.data.data_models.base import Image


class SampleValidationStrategy(Protocol):
    def __call__(self, image: Image) -> bool : ...


class NFIValidationStrategy(SampleValidationStrategy):
    def __call__(self, image) -> bool:
        # check if the image is a valid sample
        return image.data is not None