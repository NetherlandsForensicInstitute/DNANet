"""Chunked, resumable parquet cache for HIDDataset.

The cache for a given key is a *directory* of parquet chunk files:

    cache_dir/
        <key>/
            chunk_000000.parquet
            chunk_000001.parquet
            ...
            _COMPLETE          ← sentinel; only present after a full successful build

Each chunk is a self-contained parquet file written atomically (``.tmp`` →
``os.replace``). A crash partway through the build leaves the already-written
chunks intact, and ``CacheWriter.cached_paths()`` lets the next run skip the
source files whose chunks are already on disk.

Two cache modes:
- 'shell': stores scaler, annotations, adjusted panel, meta; ``HIDImage._load()``
  still runs on first data access.
- 'full': also stores the rescaled ``_data`` array and the scanpoint
  annotation; ``_load()`` is never called at runtime.
"""

from __future__ import annotations

import os
import json
import hashlib
from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from dnanet.core.panel import Panel
from dnanet.data.image import HIDImage
from dnanet.core.marker import Marker
from dnanet.core.annotation import AlleleAnnotation, ScanpointAnnotation


if TYPE_CHECKING:
    from dnanet.data.strategies.scaling.scaling import ScalingStrategy
    from dnanet.data.strategies.datasets.dataset import DatasetStrategy


CACHE_VERSION = 2  # bumped when switching to the chunked-directory layout
DEFAULT_CHUNK_SIZE = 1024

CacheMode = Literal['shell', 'full']

_COMPLETE_MARKER = '_COMPLETE'


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def compute_key(
    root: Path,
    scaling_strategy: ScalingStrategy,
    dataset_strategy: DatasetStrategy,
    data_loading_strategy: str,
    include_size_standard: bool,
    adjustment_of_annotations: str | None,
    skip_if_invalid_ladder: bool,
) -> str:
    """Return a 16-char hex cache key from the parameters that affect loaded data."""
    payload = {
        'cache_version': CACHE_VERSION,
        'root': str(root.resolve()),
        'scaling': scaling_strategy.cache_signature(),
        'dataset': dataset_strategy.cache_signature(),
        'data_loading_strategy': data_loading_strategy,
        'include_size_standard': include_size_standard,
        'adjustment_of_annotations': adjustment_of_annotations,
        'skip_if_invalid_ladder': skip_if_invalid_ladder,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def is_complete(cache_dir: Path) -> bool:
    """Return True iff the cache directory contains a ``_COMPLETE`` sentinel."""
    return (cache_dir / _COMPLETE_MARKER).exists()


# ---------------------------------------------------------------------------
# Schema and per-row serialization
# ---------------------------------------------------------------------------


def _build_schema(mode: CacheMode) -> pa.Schema:
    fields = [
        pa.field('path', pa.string()),
        pa.field('scaler', pa.binary()),
        pa.field('scaler_shape', pa.list_(pa.int32())),
        pa.field('allele_annotation_json', pa.string()),
        pa.field('adjusted_panel_json', pa.string()),
        pa.field('meta_json', pa.string()),
    ]
    if mode == 'full':
        fields += [
            pa.field('data', pa.binary()),
            pa.field('data_shape', pa.list_(pa.int32())),
            pa.field('annotation', pa.binary()),
            pa.field('annotation_shape', pa.list_(pa.int32())),
        ]
    return pa.schema(fields)


def _image_to_row(img: HIDImage, mode: CacheMode) -> dict:
    row: dict = {
        'path': str(img.path),
        'scaler': None,
        'scaler_shape': None,
        'allele_annotation_json': None,
        'adjusted_panel_json': None,
        'meta_json': json.dumps(img._meta, default=str),
    }

    if img._scaler is not None:
        arr = np.asarray(img._scaler, dtype=np.float32)
        row['scaler'] = arr.tobytes()
        row['scaler_shape'] = list(arr.shape)

    if img._allele_annotation is not None:
        row['allele_annotation_json'] = json.dumps(
            [m.to_dict() for m in img._allele_annotation.data]
        )

    if img._adjusted_panel is not None:
        row['adjusted_panel_json'] = json.dumps([m.to_dict() for m in img._adjusted_panel.markers])

    if mode == 'full':
        row['data'] = None
        row['data_shape'] = None
        row['annotation'] = None
        row['annotation_shape'] = None

        if img._data is not None:
            arr = np.asarray(img._data, dtype=np.int16)
            row['data'] = arr.tobytes()
            row['data_shape'] = list(arr.shape)

        if img._annotation is not None:
            arr = np.asarray(img._annotation.data, dtype=np.int8)
            row['annotation'] = arr.tobytes()
            row['annotation_shape'] = list(arr.shape)

    return row


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class CacheWriter:
    """Streams ``HIDImage`` rows into chunked parquet files.

    Usage::

        with CacheWriter(cache_dir, mode='full') as writer:
            resume = writer.cached_paths()
            for img in producer():
                if str(img.path) in resume:
                    continue
                writer.add(img)
            writer.finalize()

    Each call to ``add`` appends one row to an in-memory buffer; when the
    buffer reaches ``chunk_size`` it is flushed to a new parquet file.
    ``finalize`` flushes any remaining buffered rows and writes the
    ``_COMPLETE`` sentinel. If the writer exits without a successful
    ``finalize`` (e.g. the process crashes), the chunks already on disk are
    still valid and will be picked up on resume.
    """

    def __init__(
        self,
        cache_dir: Path,
        mode: CacheMode,
        chunk_size: int | None = None,
    ) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mode = mode
        self._chunk_size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
        self._schema = _build_schema(mode)
        self._buffer: list[dict] = []
        self._next_chunk_idx = self._scan_existing_chunks()

    def _scan_existing_chunks(self) -> int:
        """Resume-safe: clean up stale tmp files, return the next chunk index."""
        for tmp in self._dir.glob('chunk_*.parquet.tmp'):
            tmp.unlink(missing_ok=True)

        max_idx = -1
        for p in self._dir.glob('chunk_*.parquet'):
            try:
                idx = int(p.stem.split('_', 1)[1])
            except ValueError:
                continue
            if idx > max_idx:
                max_idx = idx
        return max_idx + 1

    def cached_paths(self) -> set[str]:
        """Return the set of source ``path`` values already written to any chunk."""
        paths: set[str] = set()
        for p in sorted(self._dir.glob('chunk_*.parquet')):
            try:
                tbl = pq.read_table(p, columns=['path'])
            except Exception as exc:
                logger.warning('Unreadable cache chunk {} ({}); removing', p, exc)
                p.unlink(missing_ok=True)
                continue
            paths.update(tbl.column('path').to_pylist())
        return paths

    def add(self, image: HIDImage) -> None:
        self._buffer.append(_image_to_row(image, self._mode))
        if len(self._buffer) >= self._chunk_size:
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        if not self._buffer:
            return

        cols = {name: [row.get(name) for row in self._buffer] for name in self._schema.names}
        table = pa.table(cols, schema=self._schema)

        idx = self._next_chunk_idx
        final = self._dir / f'chunk_{idx:06d}.parquet'
        tmp = final.with_suffix('.parquet.tmp')
        pq.write_table(table, tmp, compression='snappy')
        os.replace(tmp, final)

        logger.debug('Cache chunk written: {} ({} rows)', final.name, len(self._buffer))
        self._buffer.clear()
        self._next_chunk_idx += 1

    def finalize(self) -> None:
        """Flush any pending buffer and mark the cache complete."""
        self._flush_chunk()
        (self._dir / _COMPLETE_MARKER).touch()
        logger.info('Cache finalized at {}', self._dir)

    def __enter__(self) -> 'CacheWriter':
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Always try to flush buffered rows so a crash mid-iteration still saves
        # everything that was serialized. `finalize()` will have emptied the
        # buffer already on the success path.
        try:
            self._flush_chunk()
        except Exception as flush_exc:
            logger.warning('Failed to flush final cache chunk: {}', flush_exc)
        return False


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_cache(
    cache_dir: Path,
    scaling_strategy: ScalingStrategy,
    mode: CacheMode,
    include_size_standard: bool = False,
    load_in_memory: bool = True,
) -> list[HIDImage]:
    """Reconstruct ``HIDImage`` instances from all chunk files under ``cache_dir``."""
    chunk_paths = sorted(cache_dir.glob('chunk_*.parquet'))
    if not chunk_paths:
        raise FileNotFoundError(f'No cache chunks found under {cache_dir}')

    images: list[HIDImage] = []
    for chunk_path in chunk_paths:
        table = pq.read_table(chunk_path)
        columns = table.schema.names
        for i in range(len(table)):
            row = {col: table.column(col)[i].as_py() for col in columns}
            images.append(
                _reconstruct(row, scaling_strategy, mode, include_size_standard, load_in_memory)
            )

    logger.info(
        'Cache loaded: {} ({} images, {} chunks, mode={})',
        cache_dir,
        len(images),
        len(chunk_paths),
        mode,
    )
    return images


def _reconstruct(
    row: dict,
    scaling_strategy: ScalingStrategy,
    mode: CacheMode,
    include_size_standard: bool,
    load_in_memory: bool,
) -> HIDImage:
    scaler: np.ndarray | None = None
    if row['scaler'] is not None:
        scaler = np.frombuffer(row['scaler'], dtype=np.float32).reshape(row['scaler_shape']).copy()

    allele_annotation: AlleleAnnotation | None = None
    if row.get('allele_annotation_json'):
        allele_annotation = AlleleAnnotation(
            data=[Marker.from_dict(d) for d in json.loads(row['allele_annotation_json'])]
        )

    adjusted_panel: Panel | None = None
    if row.get('adjusted_panel_json'):
        adjusted_panel = Panel(
            markers=tuple(Marker.from_dict(d) for d in json.loads(row['adjusted_panel_json']))
        )

    meta: dict = json.loads(row['meta_json']) if row.get('meta_json') else {}

    img = HIDImage(
        path=row['path'],
        scaling_strategy=scaling_strategy,
        adjusted_panel=adjusted_panel,
        include_size_standard=include_size_standard,
        allele_annotation=allele_annotation,
        load_in_memory=load_in_memory,
        meta=meta,
    )
    img._scaler = scaler

    if mode == 'full':
        if 'data' not in row or 'annotation' not in row:
            raise ValueError(
                'Loading cache as `full` but chunk is missing `data`/`annotation` columns. '
                'The cache was likely written in `shell` mode.'
            )

        if row.get('data') is not None:
            img._data = np.frombuffer(row['data'], dtype=np.int16).reshape(row['data_shape']).copy()
        if row.get('annotation') is not None:
            ann_data = (
                np.frombuffer(row['annotation'], dtype=np.int8)
                .reshape(row['annotation_shape'])
                .copy()
            )
            img.annotation = ScanpointAnnotation(data=ann_data)

    return img
