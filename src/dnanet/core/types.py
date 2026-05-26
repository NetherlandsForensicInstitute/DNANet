"""Shared type aliases used across the project.

Kept minimal: only aliases that appear in 3+ modules belong here.
"""

from typing import Union
from pathlib import Path


PathLike = Union[str, Path]
"""A file-system path, either as a string or a `pathlib.Path`."""
