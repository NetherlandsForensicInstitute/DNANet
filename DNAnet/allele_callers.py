import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from DNAnet.data.data_models import Allele, Marker, Panel
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.models.prediction import Prediction


LOGGER = logging.getLogger('dnanet')

NON_AUTOSOMAL_MARKERS = ['AMEL', 'DYS391', 'DYS576', 'DYS570']


class AlleleCaller(ABC):
    """
    Base class for an object that can call alleles from the predicted segmentation image.
    """

    @abstractmethod
    def call_alleles(self,
                     image: HIDImage,
                     prediction: Prediction) -> Prediction:
        raise NotImplementedError

    @abstractmethod
    def translate_pixels_to_alleles(self,
                                    scaler: np.ndarray,
                                    prediction: np.ndarray,
                                    image: np.ndarray,
                                    panel: Panel) -> List[Marker]:
        raise NotImplementedError


class NearestBasePairCaller(AlleleCaller):
    """
    Object that calls alleles from predicted segmentation image, by comparing the mean base
    pair location of a predicted bin with the mean base pair of an allele from the panel.
    """

    def call_alleles(self,
                     image: HIDImage,
                     prediction: Prediction) -> Prediction:
        called_alleles = self.translate_pixels_to_alleles(
            image._scaler[np.newaxis, :],
            prediction.image,
            image.data,
            image._panel
        )

        if 'called_alleles_manual' in image.meta:
            # in this case we are working on ground truth annotations, remove DYS and AMEL
            # from the prediction as these markers are not present in the annotations
            prediction.meta["called_alleles"] = \
                [m for m in called_alleles if m.name not in NON_AUTOSOMAL_MARKERS]
        else:
            prediction.meta["called_alleles"] = called_alleles
        return prediction

    def translate_pixels_to_alleles(
            self,
            scaler: np.ndarray,
            prediction_image: np.ndarray,
            image: np.ndarray,
            panel: Panel
    ) -> List[Marker]:
        """
        Translate the predictions in the `prediction_image` to actual marker
        and allele names. First search for the mean base pair of a prediction
        group using the `scaler`. Then find the allele name corresponding to
        the base pair that is closest to the mean predicted base pair via the panel.
        """
        loci_dict = defaultdict(set)
        rfus = defaultdict(int)
        for dye_index, dye in enumerate(prediction_image):
            # find indices of groups of positive predictions (where logits are greater than .5)
            positives, _ = np.where(dye >= 0.5)
            if positives.size == 0:  # no predictions present in this dye
                LOGGER.warning(f"No predictions present in dye row {dye_index}")
                continue

            # split the positives in separate arrays by splitting on where the
            # indices are not consecutive
            predicted_bins = np.split(positives, np.where(np.diff(positives) != 1)[0] + 1)
            for prediction_bin in predicted_bins:
                # get the mean basepair of the bin via its pixel values and the scaler
                mean_bp = np.mean(scaler[:, prediction_bin])
                marker_name, allele_name = self.get_allele_by_nearest_bp(dye_index, mean_bp, panel)
                loci_dict[(dye_index, marker_name)].add(allele_name)
                # save highest rfu found for this allele (alleles may be found several times)
                max_rfu = max(image[dye_index, prediction_bin])
                rfus[(marker_name, allele_name)] = int(max(
                    rfus[(marker_name, allele_name)],
                    max_rfu
                ))

        return [
            Marker(
                dye_index,
                marker_name,
                alleles=[
                    Allele(name=allele_name, height=rfus[(marker_name, allele_name)])
                    for allele_name in alleles
                ],
            )
            for (dye_index, marker_name), alleles in loci_dict.items()
        ]

    @staticmethod
    def get_allele_by_nearest_bp(
            dye_index: int,
            base_pair: float,
            panel: Panel
    ) -> Tuple[str, str]:
        """
        Retrieve the marker and allele name that is, according to the panel,
        closest to the provided `base_pair` located on the dye with index `dye_index`.
        """
        all_basepairs_on_dye = panel.dye_bp_to_allele_mapping[dye_index].keys()
        nearest_bp = min(all_basepairs_on_dye, key=lambda k: abs(k - base_pair))
        marker_name, allele_name = panel.get_allele_name_by_dye_and_bp(dye_index, nearest_bp)
        return marker_name, allele_name
