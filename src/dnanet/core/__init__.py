"""Core domain models for forensic DNA profile analysis.

This package contains the fundamental data structures that model the forensic
DNA domain: alleles, markers, panels, annotations, and predictions. These are
pure data containers with no dependencies on ML frameworks.
"""

from dnanet.core.panel import Panel
from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.constants import LabelCategory
from dnanet.core.annotation import (
    Annotation,
    ClassAnnotation,
    AlleleAnnotation,
    ScanpointAnnotation,
)


__all__ = [
    "Allele",
    "AlleleAnnotation",
    "Annotation",
    "ClassAnnotation",
    "LabelCategory",
    "Marker",
    "Panel",
    "ScanpointAnnotation",
]
