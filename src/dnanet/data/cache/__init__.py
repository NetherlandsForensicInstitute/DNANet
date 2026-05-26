"""Lazy, memmap-backed disk cache for HIDDataset.

The cache for a given key lives under:

    <cache_dir>/<key>/
        data.bin           # (N, D, L)       int16    — fluorescence signal
        annotation.bin     # (N, D, L)       int8     — scanpoint mask
        scaler.bin         # (N, L)          float32  — base-pair scaler
        index.parquet      # N rows of path + small keys into sidecars + meta_json
        panels.json        # {panel_key: adjusted_panel_json}  — deduped
        alleles.json       # {allele_key: allele_annotation_json}  — deduped
        shapes.json        # row counts, per-array shapes and dtypes
        fingerprint.json   # hash of source files + config; checked on load
        manifest.jsonl     # append-only writer log (removed at finalize)
        _COMPLETE          # sentinel, only touched after a successful finalize

At runtime ``HIDDataset.__init__`` reads only ``index.parquet`` + ``shapes.json``
(no pixel data). ``__getitem__`` opens the three ``.bin`` files via
``numpy.memmap`` on first access in each worker and slices the requested row.
"""

from dnanet.data.cache.layout import is_complete, cache_key_dir
from dnanet.data.cache.reader import IndexEntry, MemmapCacheReader
from dnanet.data.cache.writer import MemmapCacheWriter
from dnanet.data.cache.fingerprint import (
    compute_key,
    read_fingerprint,
    compute_fingerprint,
    validate_fingerprint,
)


__all__ = [
    'MemmapCacheReader',
    'MemmapCacheWriter',
    'IndexEntry',
    'compute_key',
    'compute_fingerprint',
    'read_fingerprint',
    'validate_fingerprint',
    'is_complete',
    'cache_key_dir',
]
