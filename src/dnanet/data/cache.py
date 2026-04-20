"""Parquet-based cache for HIDDataset.

Stores pre-processed HIDImage fields (scaler, annotations, adjusted panel,
optionally the rescaled data array) to skip the expensive loading pipeline
on subsequent runs.

Two cache modes:
- 'shell': stores everything except _data; HIDImage._load() still runs on
  first data access, but annotation/panel/ladder work is skipped entirely.
- 'full': also stores the rescaled _data array; _load() is never called.
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


CACHE_VERSION = 1

CacheMode = Literal['shell', 'full']


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


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_cache(path: Path, images: list[HIDImage], mode: CacheMode) -> None:
    """Serialize images to a parquet file at path (atomic write)."""
    rows: dict[str, list] = {
        'path': [],
        'scaler': [],
        'scaler_shape': [],
        'allele_annotation_json': [],
        'adjusted_panel_json': [],
        'meta_json': [],
    }
    if mode == 'full':
        rows['data'] = []
        rows['data_shape'] = []
        rows['annotation'] = []
        rows['annotation_shape'] = []

    for img in images:
        rows['path'].append(str(img.path))

        if img._scaler is not None:
            arr = np.asarray(img._scaler, dtype=np.float64)
            rows['scaler'].append(arr.tobytes())
            rows['scaler_shape'].append(list(arr.shape))
        else:
            rows['scaler'].append(None)
            rows['scaler_shape'].append(None)

        if img._allele_annotation is not None:
            rows['allele_annotation_json'].append(
                json.dumps([m.to_dict() for m in img._allele_annotation.data])
            )
        else:
            rows['allele_annotation_json'].append(None)

        if img._adjusted_panel is not None:
            rows['adjusted_panel_json'].append(
                json.dumps([m.to_dict() for m in img._adjusted_panel.markers])
            )
        else:
            rows['adjusted_panel_json'].append(None)

        rows['meta_json'].append(json.dumps(img._meta, default=str))

        if mode == 'full':
            if img._data is not None:
                arr = np.asarray(img._data, dtype=np.int16)
                rows['data'].append(arr.tobytes())
                rows['data_shape'].append(list(arr.shape))
            else:
                rows['data'].append(None)
                rows['data_shape'].append(None)
            
            if img._annotation is not None:
                arr = np.asarray(img._annotation.data, dtype=np.int8)
                rows['annotation'].append(arr.tobytes())
                rows['annotation_shape'].append(list(arr.shape))
            else:
                rows['annotation'].append(None)
                rows['annotation_shape'].append(None)


    schema_fields = [
        pa.field('path', pa.string()),
        pa.field('scaler', pa.binary()),
        pa.field('scaler_shape', pa.list_(pa.int32())),
        pa.field('allele_annotation_json', pa.string()),
        pa.field('adjusted_panel_json', pa.string()),
        pa.field('meta_json', pa.string()),
    ]
    if mode == 'full':
        schema_fields += [
            pa.field('data', pa.binary()),
            pa.field('data_shape', pa.list_(pa.int32())),
            pa.field('annotation', pa.binary()),
            pa.field('annotation_shape', pa.list_(pa.int32())),
        ]

    table = pa.table(rows, schema=pa.schema(schema_fields))
    tmp = path.with_suffix('.parquet.tmp')
    pq.write_table(table, tmp, compression='snappy')
    os.replace(tmp, path)
    logger.info('Cache written: {} ({} images, mode={})', path, len(images), mode)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_cache(
    path: Path,
    scaling_strategy: ScalingStrategy,
    mode: CacheMode,
    include_size_standard: bool = False,
    load_in_memory: bool = True,
) -> list[HIDImage]:
    """Reconstruct HIDImages from a parquet cache file."""
    table = pq.read_table(path)
    images = [
        _reconstruct(
            {col: table.column(col)[i].as_py() for col in table.schema.names},
            scaling_strategy,
            mode,
            include_size_standard,
            load_in_memory,
        )
        for i in range(len(table))
    ]
    logger.info('Cache loaded: {} ({} images, mode={})', path, len(images), mode)
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
        shape = row['scaler_shape']
        scaler = np.frombuffer(row['scaler'], dtype=np.float64).reshape(shape).copy()

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
        # Check if the cache was saved with 'full' mode
        if 'data' not in row.keys():
            raise ValueError('Loading cache as `full` but no `data` is present')
        if 'annotation' not in row.keys():
            raise ValueError('Loading cache as `full` but no `annotation` is present')
    
        if row.get('data') is not None:
            shape = row['data_shape']
            img._data = np.frombuffer(row['data'], dtype=np.int16).reshape(shape).copy()
        if row.get('annotation') is not None:
            annotation_shape = row['annotation_shape']
            ann_data = np.frombuffer(row['annotation'], dtype=np.int16).reshape(annotation_shape).copy()
            img.annotation = ScanpointAnnotation(data=ann_data)
    return img
