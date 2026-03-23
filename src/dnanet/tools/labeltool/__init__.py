"""Interactive label tool for EPG scan-point annotation.

Provides:
    - :class:`LabelTool` — Matplotlib-based interactive annotation UI.
    - :class:`AnnotationStore` — CSV-based annotation persistence (Repository pattern).
    - :func:`main` — CLI entry point for ``dnanet-label``.
"""

from dnanet.tools.labeltool.annotations import AnnotationStore
from dnanet.tools.labeltool.cli import main
from dnanet.tools.labeltool.tool import LabelTool

__all__ = ["LabelTool", "AnnotationStore", "main"]
