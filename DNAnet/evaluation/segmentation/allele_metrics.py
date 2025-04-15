from typing import Optional, Sequence

from DNAnet.data.data_models.base import Image
from DNAnet.evaluation.utils import flatten_marker_list_to_locusallelename_list
from DNAnet.models.prediction import Prediction


def allele_precision(
    images: Sequence[Image],
    predictions: Sequence[Prediction],
    locus_name: Optional[str] = None,
    low_or_high_threshold: Optional[str] = None,
    min_rfu: Optional[int] = None,
) -> float:
    true_positives, false_positives = 0, 0
    if 'called_alleles_manual' in images[0].meta:
        # in this case, we are loading donor alleles, that do not have peak heights
        min_rfu_annotation = None
        low_or_high_threshold_annotation = None
    else:
        min_rfu_annotation = min_rfu
        low_or_high_threshold_annotation = low_or_high_threshold

    for image, prediction in zip(images, predictions):
        # convert the list of markers to list of locus_allele name strings
        predicted_alleles = flatten_marker_list_to_locusallelename_list(
            prediction.meta["called_alleles"], locus=locus_name, min_rfu=min_rfu,
            low_or_high=low_or_high_threshold
        )

        annotated_alleles = flatten_marker_list_to_locusallelename_list(
            image.meta["called_alleles"], locus=locus_name, min_rfu=min_rfu_annotation,
            low_or_high=low_or_high_threshold_annotation
        )
        total_positive = len(predicted_alleles)
        true = len(set(annotated_alleles).intersection(set(predicted_alleles)))
        true_positives += true
        false_positives += total_positive - true
    return true_positives / (true_positives + false_positives) \
        if true_positives + false_positives != 0 else 0


def allele_recall(
    images: Sequence[Image],
    predictions: Sequence[Prediction],
    locus_name: Optional[str] = None,
    low_or_high_threshold: Optional[str] = None,
    min_rfu: Optional[int] = None,
) -> float:
    true_positives, false_negatives = 0, 0
    if 'called_alleles_manual' in images[0].meta:
        # in this case, we are loading donor alleles, that do not have peak heights
        min_rfu_annotation = None
        low_or_high_threshold_annotation = None
    else:
        min_rfu_annotation = min_rfu
        low_or_high_threshold_annotation = low_or_high_threshold

    for image, prediction in zip(images, predictions):
        # convert the list of markers to list of locus_allele name strings
        predicted_alleles = flatten_marker_list_to_locusallelename_list(
            prediction.meta["called_alleles"], locus=locus_name, min_rfu=min_rfu,
            low_or_high=low_or_high_threshold
        )
        annotated_alleles = flatten_marker_list_to_locusallelename_list(
            image.meta["called_alleles"], locus=locus_name, min_rfu=min_rfu_annotation,
            low_or_high=low_or_high_threshold_annotation
        )
        total = len(annotated_alleles)
        true = len(set(annotated_alleles).intersection(set(predicted_alleles)))
        true_positives += true
        false_negatives += total - true
    return true_positives / (true_positives + false_negatives) \
        if true_positives + false_negatives != 0 else 0


def allele_f1_score(
        images: Sequence[Image], predictions: Sequence[Prediction]
) -> float:
    p = allele_precision(images, predictions)
    r = allele_recall(images, predictions)
    if p == 0 and r == 0:
        return 0
    return 2 * p * r / (p + r) if p + r != 0 else 0
