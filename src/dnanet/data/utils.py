"""Generic data utilities.

Functions here are small, stateless helpers that are dataset-agnostic and
kit-agnostic. Dataset-specific helpers (R&D filename parsing, ProvedIt
naming conventions, etc.) live in their respective ``DatasetStrategy``
implementations under ``dnanet.data.strategies.datasets``.

If this file grows beyond ~100 lines, it's time to split it.
"""

from __future__ import annotations

from pathlib import Path

import coolname


def generate_random_name() -> str:
    """Generate a random human-readable experiment name (e.g. 'BrilliantFalcon')."""
    return ''.join(word.capitalize() for word in coolname.generate())


def find_files_by_suffix(root: str | Path, suffix: str) -> list[Path]:
    """Recursively find all files with a given suffix."""
    return list(Path(root).rglob(f'*{suffix}'))
