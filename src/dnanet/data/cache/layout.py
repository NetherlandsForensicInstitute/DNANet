"""Filesystem layout for the memmap cache."""

from __future__ import annotations

from pathlib import Path


CACHE_VERSION = 3  # bump when the on-disk layout changes

_COMPLETE_MARKER = '_COMPLETE'

DATA_BIN = 'data.bin'
ANNOTATION_BIN = 'annotation.bin'
SCALER_BIN = 'scaler.bin'
INDEX_PARQUET = 'index.parquet'
SHAPES_JSON = 'shapes.json'
FINGERPRINT_JSON = 'fingerprint.json'
MANIFEST_JSONL = 'manifest.jsonl'


def cache_key_dir(cache_root: Path, key: str) -> Path:
    return Path(cache_root) / key


def is_complete(cache_dir: Path) -> bool:
    return (Path(cache_dir) / _COMPLETE_MARKER).exists()


def mark_complete(cache_dir: Path) -> None:
    (Path(cache_dir) / _COMPLETE_MARKER).touch()


def clear_complete(cache_dir: Path) -> None:
    p = Path(cache_dir) / _COMPLETE_MARKER
    if p.exists():
        p.unlink()
