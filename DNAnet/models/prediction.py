import logging
from operator import itemgetter
import itertools
from pathlib import Path
from typing import Mapping, Sequence, Optional

import numpy as np

from DNAnet.data.data_models import Marker
from DNAnet.typing import PathLike

LOGGER = logging.getLogger('dnanet')

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
            "called_alleles": [marker.to_dict() for marker in self.called_alleles] if self.called_alleles else None,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            classification=data['classification'],
            image=np.array(data['image']),
            original_image_path=data['original_image_path'],
            called_alleles=[Marker.from_dict(marker) for marker in data['called_alleles']] if data.get('called_alleles') else None,
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


def write_predictions_to_txt(predictions: Sequence[Prediction], folder_name: PathLike):
    """
    Write the called alleles of a list of predictions to .txt files, located in the `folder_name`.
    """
    folder_name = Path(folder_name) if isinstance(folder_name, str) else folder_name
    folder_name.mkdir(parents=True, exist_ok=True)
    for pred in predictions:
        if not (sample_name := pred.original_image_path.stem):
            raise ValueError(f"Need a `sample_name` to write prediction to file, got {sample_name}")
        if not (called_alleles := pred.called_alleles):
            LOGGER.info(f"No called alleles found for {sample_name}, skipping")
            continue
        # create the header of the txt file based on the max nr of alleles on a marker
        max_n_alleles = max([len(m.alleles) for m in called_alleles])
        header_allele_height = itertools.chain(
            *[[f"Allele#{i}", f"Height#{i}"] for i in range(1, max_n_alleles + 1)]
        )
        header = ["Sample Name", "Marker"] + list(header_allele_height)
        lines = ["\t".join(header) + "\n"]
        # per marker, gather the allele names and heights
        for marker in called_alleles:
            line = [sample_name, marker.name]
            for allele in sorted(marker.alleles, key=lambda x: x.name):
                line.extend([allele.name, str(allele.height)])
            line.extend([''] * (max_n_alleles - len(line)))
            lines.append("\t".join(line) + "\n")

        with open(folder_name / f"{sample_name}.txt", "w") as f:
            f.writelines(lines)

    LOGGER.info(f"Successfully wrote predictions to {folder_name}")
