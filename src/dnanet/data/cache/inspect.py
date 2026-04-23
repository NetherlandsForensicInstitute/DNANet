"""Cache introspection: report on-disk and in-RAM cost of a cache directory.

Run as::

    python -m dnanet.data.cache.inspect <cache_dir>

The cache_dir may be a concrete ``<cache_root>/<key>/`` or the parent
``<cache_root>`` — in the latter case every key directory is inspected.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pyarrow.parquet as pq

from dnanet.data.cache.layout import (
    DATA_BIN,
    SCALER_BIN,
    PANELS_JSON,
    SHAPES_JSON,
    ALLELES_JSON,
    INDEX_PARQUET,
    ANNOTATION_BIN,
    FINGERPRINT_JSON,
    is_complete,
)


def _mb(n: int) -> str:
    return f'{n / 1e6:>8.2f} MB'


def _gb(n: int) -> str:
    return f'{n / 1e9:>6.2f} GB'


def _dir_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _inspect_one(cache_dir: Path) -> None:
    print(f'\n=== {cache_dir} ===')
    if not cache_dir.is_dir():
        print('  (not a directory)')
        return
    print(f'  complete marker:  {is_complete(cache_dir)}')

    shapes_path = cache_dir / SHAPES_JSON
    if shapes_path.exists():
        shapes = json.loads(shapes_path.read_text())
        n = shapes.get('n', 0)
        print(f'  rows (n):         {n}')

    # -- On-disk sizes ----------------------------------------------------
    sizes = {
        DATA_BIN: _dir_size(cache_dir / DATA_BIN),
        ANNOTATION_BIN: _dir_size(cache_dir / ANNOTATION_BIN),
        SCALER_BIN: _dir_size(cache_dir / SCALER_BIN),
        INDEX_PARQUET: _dir_size(cache_dir / INDEX_PARQUET),
        PANELS_JSON: _dir_size(cache_dir / PANELS_JSON),
        ALLELES_JSON: _dir_size(cache_dir / ALLELES_JSON),
        SHAPES_JSON: _dir_size(cache_dir / SHAPES_JSON),
        FINGERPRINT_JSON: _dir_size(cache_dir / FINGERPRINT_JSON),
    }
    print('  on-disk:')
    for name, sz in sizes.items():
        print(f'    {name:<20}  {_mb(sz)}')
    print(f'    {"TOTAL":<20}  {_mb(sum(sizes.values()))}')

    # -- In-RAM cost of the Python-side index ----------------------------
    idx_path = cache_dir / INDEX_PARQUET
    if not idx_path.exists():
        return
    table = pq.read_table(idx_path)
    n = table.num_rows
    cols = table.schema.names
    print(f'  index columns:    {cols}')

    # String column byte totals (what ends up pinned in self._index)
    print(f'  index in-RAM (string columns, approx):')
    rows = table.to_pylist()
    string_total = 0
    for col in cols:
        col_bytes = sum(len(r[col]) for r in rows if isinstance(r.get(col), str) and r.get(col))
        if col_bytes:
            print(f'    {col:<20}  {_mb(col_bytes)}')
            string_total += col_bytes
    print(f'    {"STRING TOTAL":<20}  {_mb(string_total)}')

    # -- Sidecars --------------------------------------------------------
    panels_path = cache_dir / PANELS_JSON
    if panels_path.exists():
        panels = json.loads(panels_path.read_text())
        sizes_p = [len(v) for v in panels.values()]
        if sizes_p:
            print(
                f'  panels.json:      {len(panels)} unique '
                f'(avg={sum(sizes_p) // max(len(sizes_p), 1)} B, '
                f'max={max(sizes_p)} B)'
            )

    alleles_path = cache_dir / ALLELES_JSON
    if alleles_path.exists():
        alleles = json.loads(alleles_path.read_text())
        sizes_a = [len(v) for v in alleles.values()]
        if sizes_a:
            print(
                f'  alleles.json:     {len(alleles)} unique '
                f'(avg={sum(sizes_a) // max(len(sizes_a), 1)} B, '
                f'max={max(sizes_a)} B)'
            )

    # -- Projection -------------------------------------------------------
    memmap_shapes = json.loads((cache_dir / SHAPES_JSON).read_text())
    bytes_per_sample = (
        _row_bytes(memmap_shapes['data'])
        + _row_bytes(memmap_shapes['annotation'])
        + _row_bytes(memmap_shapes['scaler'])
    )
    print(
        f'  memmap realize:   {_gb(bytes_per_sample * n)} if load_in_memory=True '
        f'({bytes_per_sample} B/sample)'
    )


def _row_bytes(spec: dict) -> int:
    import numpy as np

    return int(np.prod(spec['shape'])) * int(np.dtype(spec['dtype']).itemsize)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f'Usage: python -m dnanet.data.cache.inspect <cache_dir>', file=sys.stderr)
        return 2

    root = Path(argv[1])
    if not root.exists():
        print(f'{root} does not exist', file=sys.stderr)
        return 1

    # If it looks like a single cache directory (has shapes.json or manifest),
    # inspect it directly; else treat as a parent of key-directories.
    direct_markers = [SHAPES_JSON, INDEX_PARQUET, FINGERPRINT_JSON]
    if any((root / m).exists() for m in direct_markers):
        _inspect_one(root)
    else:
        subdirs = sorted(p for p in root.iterdir() if p.is_dir())
        if not subdirs:
            _inspect_one(root)
        else:
            for d in subdirs:
                _inspect_one(d)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
