"""Lazy, memmap-backed cache reader.

Designed so ``HIDDataset.__init__`` can build an index in memory without
touching any pixel data, and ``__getitem__`` reads a single row from three
memmaps on first access per worker.
"""

from __future__ import annotations

import os
import json
from typing import Any, Tuple
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq

from dnanet.data.cache.layout import (
    DATA_BIN,
    SCALER_BIN,
    SHAPES_JSON,
    INDEX_PARQUET,
    ANNOTATION_BIN,
)


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One row of the cache index — path + serialized metadata blobs.

    Heavy fields (``data``, ``annotation``, ``scaler``) are NOT stored here;
    they live in the memmap files and are fetched by row index.
    """

    row: int
    path: str
    allele_annotation_json: str | None
    adjusted_panel_json: str | None
    meta_json: str | None


class MemmapCacheReader:
    """Thin, worker-safe reader over the three memmap files + index parquet.

    The reader holds no numpy memmap handles until ``get_row`` is called.
    Memmaps are opened the first time a row is requested *in the current
    process*, which is exactly what we want under DataLoader's fork-based
    workers (each worker opens its own memmap independently).
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = Path(cache_dir)

        shapes = json.loads((self._dir / SHAPES_JSON).read_text())
        self._n: int = int(shapes['n'])
        self._data_shape: Tuple[int, ...] = tuple(shapes['data']['shape'])
        self._ann_shape: Tuple[int, ...] = tuple(shapes['annotation']['shape'])
        self._scaler_shape: Tuple[int, ...] = tuple(shapes['scaler']['shape'])
        self._data_dtype = np.dtype(shapes['data']['dtype'])
        self._ann_dtype = np.dtype(shapes['annotation']['dtype'])
        self._scaler_dtype = np.dtype(shapes['scaler']['dtype'])

        # Memmaps are opened lazily and keyed by pid so DataLoader workers
        # each get their own handles (fork-safety).
        self._data_mm: np.memmap | None = None
        self._ann_mm: np.memmap | None = None
        self._scaler_mm: np.memmap | None = None
        self._mm_pid: int | None = None

    # -- Index ------------------------------------------------------------- #

    def load_index(self) -> list[IndexEntry]:
        """Read index.parquet into memory — no heavy data touched."""
        table = pq.read_table(self._dir / INDEX_PARQUET)
        rows = table.to_pylist()
        return [
            IndexEntry(
                row=int(r['row']),
                path=str(r['path']),
                allele_annotation_json=r.get('allele_annotation_json'),
                adjusted_panel_json=r.get('adjusted_panel_json'),
                meta_json=r.get('meta_json'),
            )
            for r in rows
        ]

    # -- Row access -------------------------------------------------------- #

    def __len__(self) -> int:
        return self._n

    def _ensure_mm(self) -> None:
        pid = os.getpid()
        if self._mm_pid == pid and self._data_mm is not None:
            return

        self._data_mm = np.memmap(
            self._dir / DATA_BIN,
            dtype=self._data_dtype,
            mode='r',
            shape=(self._n,) + self._data_shape,
        )
        self._ann_mm = np.memmap(
            self._dir / ANNOTATION_BIN,
            dtype=self._ann_dtype,
            mode='r',
            shape=(self._n,) + self._ann_shape,
        )
        self._scaler_mm = np.memmap(
            self._dir / SCALER_BIN,
            dtype=self._scaler_dtype,
            mode='r',
            shape=(self._n,) + self._scaler_shape,
        )
        self._mm_pid = pid

    def get_row(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(data, annotation, scaler)`` copies for row ``idx``.

        Slices are copied off the memmap so the returned arrays are
        detached, writable, and the tensor conversion downstream is safe.
        """
        if idx < 0:
            idx += self._n
        if not 0 <= idx < self._n:
            raise IndexError(f'row {idx} out of range for cache with {self._n} rows')

        self._ensure_mm()
        assert self._data_mm is not None
        assert self._ann_mm is not None
        assert self._scaler_mm is not None

        data = np.asarray(self._data_mm[idx]).copy()
        annotation = np.asarray(self._ann_mm[idx]).copy()
        scaler = np.asarray(self._scaler_mm[idx]).copy()
        return data, annotation, scaler

    # -- Cleanup on pickling / worker teardown ----------------------------- #

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Drop memmap handles before pickling across process boundaries.
        state['_data_mm'] = None
        state['_ann_mm'] = None
        state['_scaler_mm'] = None
        state['_mm_pid'] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
