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
    PANELS_JSON,
    SHAPES_JSON,
    ALLELES_JSON,
    INDEX_PARQUET,
    ANNOTATION_BIN,
)


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One row of the cache index — path + small keys into the deduped sidecars.

    Heavy fields (``data``, ``annotation``, ``scaler``) live in the memmap
    files and are fetched by row index. Large JSON blobs
    (``adjusted_panel_json``, ``allele_annotation_json``) are stored once per
    unique value in ``panels.json`` / ``alleles.json`` and referenced here by
    ``panel_key`` / ``allele_key`` — keeping the in-RAM index compact.

    ``has_annotation`` is ``False`` when the source HID had no annotation; the
    annotation memmap still holds a zero row for shape uniformity, and the
    reader materializes ``image.annotation=None`` for these rows.
    """

    row: int
    path: str
    has_annotation: bool
    panel_key: int | None
    allele_key: int | None
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

        # Deduped JSON sidecars. Loaded lazily on first lookup.
        self._panels: dict[int, str] | None = None
        self._alleles: dict[int, str] | None = None

    # -- Index ------------------------------------------------------------- #

    def load_index(self) -> list[IndexEntry]:
        """Read index.parquet into memory — no heavy data touched.

        Only compact fields (keys + path + bool + small meta_json) are
        materialized; heavy panel/annotation JSON stays in sidecars and is
        fetched on demand via :meth:`panel_json` / :meth:`allele_json`.
        """
        table = pq.read_table(self._dir / INDEX_PARQUET)
        rows = table.to_pylist()
        return [
            IndexEntry(
                row=int(r['row']),
                path=str(r['path']),
                has_annotation=bool(r.get('has_annotation', True)),
                panel_key=r.get('panel_key'),
                allele_key=r.get('allele_key'),
                meta_json=r.get('meta_json'),
            )
            for r in rows
        ]

    # -- Deduped sidecar lookup ------------------------------------------- #

    def panel_json(self, key: int | None) -> str | None:
        """Resolve a panel_key to its JSON string (loads sidecar on first call)."""
        if key is None:
            return None
        if self._panels is None:
            raw = json.loads((self._dir / PANELS_JSON).read_text())
            self._panels = {int(k): v for k, v in raw.items()}
        return self._panels.get(int(key))

    def allele_json(self, key: int | None) -> str | None:
        """Resolve an allele_key to its JSON string (loads sidecar on first call)."""
        if key is None:
            return None
        if self._alleles is None:
            raw = json.loads((self._dir / ALLELES_JSON).read_text())
            self._alleles = {int(k): v for k, v in raw.items()}
        return self._alleles.get(int(key))

    # -- Size / materialization ------------------------------------------- #

    def memmap_bytes(self) -> int:
        """Total bytes the three memmap files occupy if fully realized in RAM."""

        def _rowbytes(shape: Tuple[int, ...], dtype: np.dtype) -> int:
            return int(np.prod(shape)) * int(dtype.itemsize)

        return (
            self._n * _rowbytes(self._data_shape, self._data_dtype)
            + self._n * _rowbytes(self._ann_shape, self._ann_dtype)
            + self._n * _rowbytes(self._scaler_shape, self._scaler_dtype)
        )

    def materialize(self) -> None:
        """Realize the three memmaps into RAM-resident ``numpy.ndarray`` copies.

        After this call, ``get_row`` slices a RAM array instead of a disk-backed
        memmap. No other behaviour changes. Safe to call more than once.
        """
        self._ensure_mm()
        assert self._data_mm is not None
        assert self._ann_mm is not None
        assert self._scaler_mm is not None
        # np.array() copies the memmap into a regular ndarray.
        self._data_mm = np.array(self._data_mm)
        self._ann_mm = np.array(self._ann_mm)
        self._scaler_mm = np.array(self._scaler_mm)

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
        # Drop sidecar caches so each worker re-loads lazily on first access
        # (avoids a large pickle payload when spawn-style workers transfer
        # dataset state).
        state['_panels'] = None
        state['_alleles'] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
