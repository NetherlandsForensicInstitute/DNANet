"""Shared type aliases used across the project.

Kept minimal: only aliases that appear in 3+ modules belong here.
"""

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]
"""A file-system path, either as a string or a `pathlib.Path`."""
