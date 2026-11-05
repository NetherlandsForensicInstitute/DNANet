"""Interactive label tool for EPG scan-point annotation.

Provides:
    - :class:`LabelTool` — Matplotlib-based interactive annotation UI.
    - :class:`AnnotationStore` — CSV-based annotation persistence (Repository pattern).
    - :func:`main` — CLI entry point for ``dnanet-label``.
"""

from dnanet.tools.dnannotator.cli import main
from dnanet.tools.dnannotator.tool import LabelTool
from dnanet.tools.dnannotator.annotations import AnnotationStore


__all__ = ['LabelTool', 'AnnotationStore', 'main']
