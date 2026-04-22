"""Append-only memmap cache writer.

Usage::

    with MemmapCacheWriter(cache_dir) as writer:
        already_done = writer.resume_paths()
        for image in stream_of_hid_images():
            if str(image.path) in already_done:
                continue
            writer.write(image)
        writer.finalize(config_payload, source_paths)

Write path:
    1. ``write`` appends one row to each ``.bin`` (data/annotation/scaler) and
       writes one JSON line to ``manifest.jsonl``. The manifest line is written
       only after the binary appends flush, so a crash leaves three ``.bin``
       files and a manifest with matched lengths — or at worst a trailing
       partial ``.bin`` write, which we truncate on resume.
    2. ``finalize`` reads ``manifest.jsonl`` to produce ``index.parquet``,
       writes ``shapes.json`` and ``fingerprint.json``, deletes the manifest,
       and touches ``_COMPLETE``.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any, Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from dnanet.data.cache.layout import (
    DATA_BIN,
    SCALER_BIN,
    PANELS_JSON,
    SHAPES_JSON,
    ALLELES_JSON,
    INDEX_PARQUET,
    ANNOTATION_BIN,
    MANIFEST_JSONL,
    mark_complete,
    clear_complete,
)
from dnanet.data.cache.fingerprint import write_fingerprint, compute_fingerprint


if TYPE_CHECKING:
    from dnanet.data.image import HIDImage


_DATA_DTYPE = np.int16
_ANN_DTYPE = np.int8
_SCALER_DTYPE = np.float32


class _ShapeMismatchError(RuntimeError):
    pass


class MemmapCacheWriter:
    """Stream ``HIDImage`` rows into append-only binary files + a JSONL manifest."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        # Invalidate the complete marker while we're writing.
        clear_complete(self._dir)

        self._data_shape: tuple[int, ...] | None = None
        self._ann_shape: tuple[int, ...] | None = None
        self._scaler_shape: tuple[int, ...] | None = None

        self._data_f: io.BufferedWriter | None = None
        self._ann_f: io.BufferedWriter | None = None
        self._scaler_f: io.BufferedWriter | None = None
        self._manifest_f: io.TextIOWrapper | None = None

        self._row_count = 0
        self._resume_paths: set[str] = set()

        self._recover_from_manifest()

    # -- Resume ------------------------------------------------------------ #

    def _recover_from_manifest(self) -> None:
        """Reopen the cache in append mode, validating prior row counts.

        If the three ``.bin`` files disagree in length (e.g. crash mid-write),
        truncate all three back to the manifest's row count. That guarantees
        matched row lengths on resume.
        """
        manifest_path = self._dir / MANIFEST_JSONL
        if not manifest_path.exists():
            return

        rows: list[dict[str, Any]] = []
        with manifest_path.open('r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Partial trailing line; ignore it and truncate later.
                    break

        if not rows:
            return

        # Shapes come from the first row; later rows are validated on write.
        first = rows[0]
        self._data_shape = tuple(first['data_shape'])
        self._ann_shape = tuple(first['ann_shape'])
        self._scaler_shape = tuple(first['scaler_shape'])

        self._row_count = len(rows)
        self._resume_paths = {r['path'] for r in rows}

        # Truncate each .bin to match the manifest row count (belt-and-suspenders
        # for a partial trailing append).
        self._truncate_to_row_count(DATA_BIN, self._data_shape, _DATA_DTYPE)
        self._truncate_to_row_count(ANNOTATION_BIN, self._ann_shape, _ANN_DTYPE)
        self._truncate_to_row_count(SCALER_BIN, self._scaler_shape, _SCALER_DTYPE)

        # Rewrite the manifest in case we dropped a partial trailing line.
        with manifest_path.open('w') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')

        logger.info('Cache resume: {} rows already written in {}', self._row_count, self._dir)

    def _truncate_to_row_count(
        self, fname: str, row_shape: tuple[int, ...], dtype: np.dtype
    ) -> None:
        path = self._dir / fname
        if not path.exists():
            return
        bytes_per_row = int(np.prod(row_shape)) * np.dtype(dtype).itemsize
        expected_size = self._row_count * bytes_per_row
        if path.stat().st_size != expected_size:
            with path.open('r+b') as f:
                f.truncate(expected_size)

    # -- Public API -------------------------------------------------------- #

    def resume_paths(self) -> set[str]:
        """Source paths already written to the cache."""
        return set(self._resume_paths)

    def write(self, image: HIDImage) -> None:
        """Append one sample to the cache.

        ``image.data`` and ``image.scaler`` are required. ``image.annotation``
        may be ``None`` — a zero-filled placeholder of matching shape is
        written to ``annotation.bin`` and ``has_annotation=False`` is recorded
        in the manifest for that row.
        """
        data = image.data
        scaler = image.scaler
        ann = image.annotation.data if image.annotation is not None else None

        if data is None:
            raise ValueError(f'Refusing to cache {image.path}: data is None')
        if scaler is None:
            raise ValueError(f'Refusing to cache {image.path}: scaler is None')

        data_arr = np.ascontiguousarray(data, dtype=_DATA_DTYPE)
        scaler_arr = np.ascontiguousarray(scaler, dtype=_SCALER_DTYPE)

        has_annotation = ann is not None
        if has_annotation:
            ann_arr = np.ascontiguousarray(ann, dtype=_ANN_DTYPE)
        else:
            # Annotation arrays always match the data shape (same num_dyes×L).
            # Write zeros so the memmap stays uniform across rows.
            ann_arr = np.zeros(data_arr.shape, dtype=_ANN_DTYPE)

        # Lock shapes on first write; validate on every subsequent write.
        if self._data_shape is None:
            self._data_shape = data_arr.shape
            self._ann_shape = ann_arr.shape
            self._scaler_shape = scaler_arr.shape
        else:
            if data_arr.shape != self._data_shape:
                raise _ShapeMismatchError(
                    f'data shape {data_arr.shape} != expected {self._data_shape} for {image.path}'
                )
            if ann_arr.shape != self._ann_shape:
                raise _ShapeMismatchError(
                    f'annotation shape {ann_arr.shape} != expected {self._ann_shape} '
                    f'for {image.path}'
                )
            if scaler_arr.shape != self._scaler_shape:
                raise _ShapeMismatchError(
                    f'scaler shape {scaler_arr.shape} != expected {self._scaler_shape} '
                    f'for {image.path}'
                )

        self._ensure_open()
        assert self._data_f is not None  # for type checker
        assert self._ann_f is not None
        assert self._scaler_f is not None
        assert self._manifest_f is not None

        self._data_f.write(data_arr.tobytes())
        self._ann_f.write(ann_arr.tobytes())
        self._scaler_f.write(scaler_arr.tobytes())
        # Flush to ensure binary bytes are on disk before we commit the manifest line.
        self._data_f.flush()
        self._ann_f.flush()
        self._scaler_f.flush()

        allele_json = (
            json.dumps([m.to_dict() for m in image.allele_annotation.data])
            if image.allele_annotation is not None
            else None
        )
        panel_json = (
            json.dumps([m.to_dict() for m in image.adjusted_panel.markers])
            if image.adjusted_panel is not None
            else None
        )
        meta_json = json.dumps(image.meta, default=str)

        manifest_row = {
            'row': self._row_count,
            'path': str(image.path),
            'data_shape': list(self._data_shape),
            'ann_shape': list(self._ann_shape),
            'scaler_shape': list(self._scaler_shape),
            'has_annotation': has_annotation,
            'allele_annotation_json': allele_json,
            'adjusted_panel_json': panel_json,
            'meta_json': meta_json,
        }
        self._manifest_f.write(json.dumps(manifest_row) + '\n')
        self._manifest_f.flush()

        self._row_count += 1

    def _ensure_open(self) -> None:
        if self._data_f is None:
            self._data_f = open(self._dir / DATA_BIN, 'ab')
        if self._ann_f is None:
            self._ann_f = open(self._dir / ANNOTATION_BIN, 'ab')
        if self._scaler_f is None:
            self._scaler_f = open(self._dir / SCALER_BIN, 'ab')
        if self._manifest_f is None:
            self._manifest_f = open(self._dir / MANIFEST_JSONL, 'a')

    def finalize(
        self,
        config_payload: dict[str, Any],
        source_paths: Iterable[Path],
    ) -> None:
        """Convert the manifest into index.parquet + sidecars, write shapes+fingerprint, mark complete.

        The manifest stores allele_annotation_json and adjusted_panel_json
        inline per row for simple crash-recovery. At finalize time we
        stream-dedupe those strings into ``panels.json`` and ``alleles.json``
        sidecars and the parquet only keeps small integer keys — reducing
        index-in-RAM from ``rows * panel_size`` to ``unique_panels * panel_size``
        (typically 2–3 orders of magnitude smaller).
        """
        self._close()

        if self._row_count == 0:
            raise ValueError(f'Cannot finalize empty cache at {self._dir}')

        self._build_index_and_sidecars()
        self._write_shapes_json()
        write_fingerprint(self._dir, compute_fingerprint(config_payload, source_paths))

        # Manifest is write-log only; index.parquet is the durable source of truth.
        (self._dir / MANIFEST_JSONL).unlink(missing_ok=True)

        mark_complete(self._dir)
        logger.info('Cache finalized: {} rows at {}', self._row_count, self._dir)

    def _build_index_and_sidecars(self) -> None:
        """Stream the manifest → dedupe JSON → write sidecars + index.parquet.

        Peak RAM is bounded by the unique-value set sizes, not row count.
        """
        panels: dict[str, int] = {}
        alleles: dict[str, int] = {}

        # Row buffers for parquet. These hold only small scalars.
        rows_col: list[int] = []
        paths_col: list[str] = []
        has_ann_col: list[bool] = []
        panel_key_col: list[int | None] = []
        allele_key_col: list[int | None] = []
        meta_col: list[str | None] = []

        def _intern(s: str | None, table: dict[str, int]) -> int | None:
            if s is None:
                return None
            k = table.get(s)
            if k is None:
                k = len(table)
                table[s] = k
            return k

        with (self._dir / MANIFEST_JSONL).open('r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows_col.append(int(r['row']))
                paths_col.append(str(r['path']))
                has_ann_col.append(bool(r.get('has_annotation', True)))
                panel_key_col.append(_intern(r.get('adjusted_panel_json'), panels))
                allele_key_col.append(_intern(r.get('allele_annotation_json'), alleles))
                meta_col.append(r.get('meta_json'))

        # Write sidecars first so the parquet commit implies they exist.
        self._write_sidecar(PANELS_JSON, panels)
        self._write_sidecar(ALLELES_JSON, alleles)

        schema = pa.schema(
            [
                pa.field('row', pa.int64()),
                pa.field('path', pa.string()),
                pa.field('has_annotation', pa.bool_()),
                pa.field('panel_key', pa.int32()),
                pa.field('allele_key', pa.int32()),
                pa.field('meta_json', pa.string()),
            ]
        )
        table = pa.table(
            {
                'row': rows_col,
                'path': paths_col,
                'has_annotation': has_ann_col,
                'panel_key': panel_key_col,
                'allele_key': allele_key_col,
                'meta_json': meta_col,
            },
            schema=schema,
        )
        pq.write_table(table, self._dir / INDEX_PARQUET, compression='snappy')

        logger.info(
            'Deduped sidecars: {} unique panels, {} unique allele annotations (from {} rows)',
            len(panels),
            len(alleles),
            self._row_count,
        )

    def _write_sidecar(self, name: str, table: dict[str, int]) -> None:
        """Write ``{key: json_string}`` sidecar.

        Keys are stored as strings in the JSON object (JSON's only allowed
        key type); the reader parses them back to ``int``.
        """
        inv = {str(k): s for s, k in table.items()}
        (self._dir / name).write_text(json.dumps(inv))

    def _write_shapes_json(self) -> None:
        assert self._data_shape is not None
        assert self._ann_shape is not None
        assert self._scaler_shape is not None
        payload = {
            'n': self._row_count,
            'data': {'shape': list(self._data_shape), 'dtype': np.dtype(_DATA_DTYPE).str},
            'annotation': {'shape': list(self._ann_shape), 'dtype': np.dtype(_ANN_DTYPE).str},
            'scaler': {'shape': list(self._scaler_shape), 'dtype': np.dtype(_SCALER_DTYPE).str},
        }
        (self._dir / SHAPES_JSON).write_text(json.dumps(payload, indent=2))

    def _close(self) -> None:
        for f in (self._data_f, self._ann_f, self._scaler_f, self._manifest_f):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        self._data_f = self._ann_f = self._scaler_f = None
        self._manifest_f = None

    # -- Context manager --------------------------------------------------- #

    def __enter__(self) -> 'MemmapCacheWriter':
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._close()
        return False
