import logging
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from typing import List, Tuple

import numpy as np

from DNAnet.data.data_models import Allele, Marker, Panel
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.models.prediction import Prediction

LOGGER = logging.getLogger('dnanet')

NON_AUTOSOMAL_MARKERS = ['AMEL', 'DYS391', 'DYS576', 'DYS570']
scan_point_annotation = namedtuple('ScanPointAnnotation',
                                   ['dye_index', 'start', 'end', 'label'])


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
            prediction.called_alleles = \
                [m for m in called_alleles if m.name not in NON_AUTOSOMAL_MARKERS]
        else:
            prediction.called_alleles = called_alleles
        return prediction

    def call_alleles_from_scan_points_annotations(self,
                                                  image: HIDImage,
                                                  annotations: List[scan_point_annotation],
                                                  aggregate: bool=True,
                                                  warn_for_multiple_peaks=False) -> List[
        Marker]:
        """
        Calls alleles from annotations as provided by the annotation tool.
        If aggregate, only provides every allele once, with rfu = max rfu.
        Else, can provide the same allele multiple times, e.g. from different annotators.

        stores start and end of annotation in left and right bin of the allele
        gives a warning if warn_for_multiple_peaks and multiple local maxima are found in an
        annotated region.
        """
        scaler = image._scaler[np.newaxis, :]
        loci_dict = defaultdict(list)
        rfus = defaultdict(int)
        left = defaultdict(int)
        right = defaultdict(int)
        for annotation in annotations:
            rfu_of_annotation = image.data[annotation.dye_index, max(annotation.start,0):annotation.end+1]
            top_bp = np.argmax(rfu_of_annotation) + annotation.start
            max_rfu = int(max(rfu_of_annotation))

            # find the bin closest to the top rfu in the annotation
            # not to the whole annotation range, as the long slopes of a peak can also be annotated
            allele_name, marker_name = self.get_marker_and_allele_from_bin(annotation.dye_index,
                                                                           image._panel,
                                                                           top_bp,
                                                                           scaler)

            if warn_for_multiple_peaks:
                from scipy.signal import find_peaks

                # find all peaks of at least certain RFU, whose distance to the next bottom is
                # at least prominence, and who are distance apart
                peaks, _ = find_peaks(rfu_of_annotation.squeeze(),
                                      height=50,
                                      prominence=20,
                                      distance=5,
                                      width=5)
                if len(peaks)>1:
                    print('found multiple peaks for ', image.path.stem, marker_name, allele_name, rfu_of_annotation.squeeze()[peaks], _)

            if aggregate:
                loci_dict[(annotation.dye_index, marker_name)].append(allele_name)
                # save highest rfu found for this allele (alleles may be found several times)
                rfus[(marker_name, allele_name)] = int(max(
                    rfus[(marker_name, allele_name)],
                    max_rfu
                ))
                left[(marker_name, allele_name)] = float(min(
                    left[(marker_name, allele_name)],
                    annotation.start
                ))
                right[(marker_name, allele_name)] = float(max(
                    right[(marker_name, allele_name)],
                    annotation.end
                ))
            if not aggregate:
                allele = Allele(name=allele_name, height=max_rfu,
                                left_bin=float(scaler[:, annotation.start]),
                                right_bin=float(scaler[:, annotation.end]))
                loci_dict[(annotation.dye_index, marker_name)].append(allele)


        if aggregate:
            return [
                Marker(
                    dye_index,
                    marker_name,
                    alleles=[
                        Allele(name=allele_name,
                               height=rfus[(marker_name, allele_name)],
                               left_bin=scaler[:, left[(marker_name, allele_name)]],
                               right_bin=scaler[:, right[(marker_name, allele_name)]],
                               )
                        for allele_name in set(alleles)
                    ],
                )
                for (dye_index, marker_name), alleles in loci_dict.items()
            ]
        else:
            return [
                Marker(
                    dye_index,
                    marker_name,
                    alleles=alleles,
                )
                for (dye_index, marker_name), alleles in loci_dict.items()
            ]

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
                allele_name, marker_name = self.get_marker_and_allele_from_bin(dye_index, panel,
                                                                               prediction_bin,
                                                                               scaler)
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

    def get_marker_and_allele_from_bin(self, dye_index, panel, prediction_bin, scaler):
        mean_bp = np.mean(scaler[:, prediction_bin])
        marker_name, allele_name = self.get_allele_by_nearest_bp(dye_index, mean_bp, panel)
        return allele_name, marker_name

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
