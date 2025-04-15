from operator import itemgetter
from typing import Any, Mapping, MutableMapping

import numpy as np


class Prediction:
    """
    This class represents a single model prediction.

    :param classification: A mapping of textual labels to their corresponding
        confidence scores.
    :param image: A matrix representing an image. To be used e.g. in
        segmentation tasks, where predictions are made at the pixel level
    :param meta: A mapping where additional metadata about this prediction
        can be stored.
    """

    def __init__(self,
                 classification: Mapping[str, float] = None,
                 image: np.ndarray = None,
                 meta: MutableMapping[str, Any] = None):

        self.classification = classification
        self.image = image
        self.meta = meta or dict()

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
            "meta": self.meta
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            classification=data['classification'],
            image=np.array(data['image']),
            meta=data['meta']
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
               f"meta={self.meta}" \
               f")"

    def __repr__(self) -> str:
        return str(self)
