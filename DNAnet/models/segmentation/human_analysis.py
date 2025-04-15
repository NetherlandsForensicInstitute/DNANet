import logging

from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.models.base import Model
from DNAnet.models.prediction import Prediction
from DNAnet.typing import PathLike


LOGGER = logging.getLogger('dnanet')

NON_AUTOSOMAL_MARKERS = ['AMEL', 'DYS391', 'DYS576', 'DYS570']

class HumanAnalysis(Model):
    def __init__(self):
        """
        Initializes a model where the human analyst annotations become the predicted called alleles
        (those annotations should be stored in 'called_alleles_manual' of the meta attribute
        of the HIDImage).
        """

    def predict(self, image: HIDImage) -> Prediction:
        if 'called_alleles_manual' not in image.meta:
            raise ValueError("No `called_alleles_manual` found when running HumanAnalysis model, "
                             "set `ground_truth_as_annotations=True` in the data config file to "
                             "load `called_alleles_manual`.")
        # remove DYS and AMEL as these are not in the ground truth annotations
        called_alleles = [m for m in image.meta['called_alleles_manual']
                          if m.name not in NON_AUTOSOMAL_MARKERS]
        return Prediction(image=None, meta={'called_alleles': called_alleles,
                                            'original_image_path': image.path})

    def load(self, model_dir: PathLike):
        pass

    def save(self, model_dir: PathLike):
        pass
