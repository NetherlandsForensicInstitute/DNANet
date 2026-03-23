"""Evaluation module for DNA profile analysis.

Provides:
    - **Pixel metrics**: precision, recall, F1, IoU at scan-point level.
    - **Allele metrics**: precision, recall, F1 at allele-call level.
    - **Allele calling**: Strategy pattern for translating pixel masks to allele calls.
    - **Visualization**: Matplotlib-based EPG profile plotting.
"""

from dnanet.evaluation.allele_caller import AlleleCaller, NearestBasePairCaller
from dnanet.evaluation.metrics import (
    allele_f1_score,
    allele_precision,
    allele_recall,
    average_binary_iou,
    pixel_f1_score,
    pixel_precision,
    pixel_recall,
)

__all__ = [
    "AlleleCaller",
    "NearestBasePairCaller",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1_score",
    "average_binary_iou",
    "allele_precision",
    "allele_recall",
    "allele_f1_score",
]
