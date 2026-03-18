"""Core domain models for forensic DNA profile analysis.

This package contains the fundamental data structures that model the forensic
DNA domain: alleles, markers, panels, annotations, and predictions. These are
pure data containers with no dependencies on ML frameworks.
"""

from dnanet.core.allele import Allele
from dnanet.core.annotation import Annotation
from dnanet.core.constants import DyeIndex, LabelCategory
from dnanet.core.marker import Marker
from dnanet.core.panel import Panel
from dnanet.core.prediction import Prediction


__all__ = [
    "Allele",
    "Annotation",
    "DyeIndex",
    "LabelCategory",
    "Marker",
    "Panel",
    "Prediction",
]
