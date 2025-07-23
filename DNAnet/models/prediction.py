from operator import itemgetter
from typing import Mapping, Sequence, Optional

import numpy as np

from DNAnet.data.data_models import Marker
from DNAnet.typing import PathLike


class Prediction:
    """
    This class represents a single model prediction.

    :param classification: A mapping of textual labels to their corresponding
        confidence scores.
    :param image: A matrix representing an image. To be used e.g. in
        segmentation tasks, where predictions are made at the pixel level
    :param original_image_path: The path to the original image file
    :param called_alleles: A list of called alleles, which are instances of
        the Marker class.
    """

    def __init__(self,
                 classification: Mapping[str, float] = None,
                 image: np.ndarray = None,
                 original_image_path: PathLike = None,
                 called_alleles: Optional[Sequence[Marker]] = None):

        self.classification = classification
        self.image = image
        self.original_image_path = original_image_path
        self.called_alleles = called_alleles

    @property
    def label(self) -> str:
        """
        Returns the single label with the highest classification confidence.
        """
        if not self.classification:
            raise ValueError("Can't determine label without classification")
        return max(self.classification.items(), key=itemgetter(1))[0]

    @property
    def confidence(self) -> float:
        """
        Returns the confidence score of the label with the
        highest classification confidence.
        """
        if not self.classification:
            raise ValueError("Can't return a confidence score without classification")
        return self.classification[self.label]

    def to_dict(self):
        return {
            "classification": self.classification,
            "image": self.image.tolist() if (self.image is not None) else None,
            "original_image_path": str(self.original_image_path),
            "called_alleles": [allele.to_dict() for allele in self.called_alleles] if self.called_alleles else None,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            classification=data['classification'],
            image=np.array(data['image']),
            original_image_path=data['original_image_path'],
            called_alleles=[Marker.from_dict(allele) for allele in data['called_alleles']] if data.get('called_alleles') else None,
        )

    def __hash__(self):
        return hash((
            tuple(self.classification.items()) if self.classification else (),
            self.image.tobytes() if self.image is not None else None,
        ))

    def __str__(self) -> str:
        return f"Prediction(" \
            f"classification={self.classification}, " \
            f"image={self.image}, " \
            f"original_image_path={self.original_image_path}, " \
            f"called_alleles={self.called_alleles if self.called_alleles else None}" \
            f")"

    def __repr__(self) -> str:
        return str(self)
