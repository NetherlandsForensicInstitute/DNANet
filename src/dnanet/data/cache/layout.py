"""Filesystem layout for the memmap cache."""

from __future__ import annotations

from pathlib import Path


CACHE_VERSION = 5  # bump when the on-disk layout changes

_COMPLETE_MARKER = '_COMPLETE'

DATA_BIN = 'data.bin'
ANNOTATION_BIN = 'annotation.bin'
SCALER_BIN = 'scaler.bin'
INDEX_PARQUET = 'index.parquet'
SHAPES_JSON = 'shapes.json'
FINGERPRINT_JSON = 'fingerprint.json'
MANIFEST_JSONL = 'manifest.jsonl'
# Deduped sidecars — panels and allele annotations are heavily repeated across
# rows (all replicas of the same sample share one annotation; many samples
# share one ladder-adjusted panel), so we intern them once and reference by key.
PANELS_JSON = 'panels.json'
ALLELES_JSON = 'alleles.json'


def cache_key_dir(cache_root: Path, key: str) -> Path:  # noqa: D103
    return Path(cache_root) / key


def is_complete(cache_dir: Path) -> bool:  # noqa: D103
    return (Path(cache_dir) / _COMPLETE_MARKER).exists()


def mark_complete(cache_dir: Path) -> None:  # noqa: D103
    (Path(cache_dir) / _COMPLETE_MARKER).touch()


def clear_complete(cache_dir: Path) -> None:  # noqa: D103
    p = Path(cache_dir) / _COMPLETE_MARKER
    if p.exists():
        p.unlink()
