"""Evaluation module for DNA profile analysis.

Provides:
    - **Allele metrics**: precision, recall, F1 at allele-call level.
    - **Allele calling**: Strategy pattern for translating pixel masks to allele calls.
    - **Visualization**: Matplotlib-based EPG profile plotting.
"""

from dnanet.evaluation.metrics import (
    AlleleMetric,
    AlleleRecall,
    AlleleF1Score,
    AllelePrecision,
    PerRFUOutcomeMetric,
)
from dnanet.evaluation.allele_caller import (
    AlleleCaller,
    ExactBasePairCaller,
    NearestBasePairCaller,
    FromSegmentationImageCaller,
)


__all__ = [
    'AlleleCaller',
    'FromSegmentationImageCaller',
    'NearestBasePairCaller',
    'ExactBasePairCaller',
    'AlleleMetric',
    'AllelePrecision',
    'AlleleRecall',
    'AlleleF1Score',
    'PerRFUOutcomeMetric',
]
