"""Cache-key and source-fingerprint computation.

Two hashes are used:

- **key**: 16-char hex derived from config parameters only. Drives the cache
  directory name, so changing any config field yields a fresh cache directory.
- **fingerprint**: full-SHA hex derived from (config + every source file's
  ``(path, mtime, size)``). Stored inside the cache dir and validated on load;
  a mismatch means sources changed on disk and the cache must be rebuilt.
"""

from __future__ import annotations

import json
import hashlib
from typing import TYPE_CHECKING, Any, Iterable
from pathlib import Path

from dnanet.data.cache.layout import CACHE_VERSION, FINGERPRINT_JSON


if TYPE_CHECKING:
    from dnanet.data.strategies.scaling.scaling import ScalingStrategy
    from dnanet.data.strategies.datasets.dataset import DatasetStrategy


def _config_payload(
    root: Path,
    scaling_strategy: ScalingStrategy,
    dataset_strategy: DatasetStrategy,
    data_loading_strategy: str,
    include_size_standard: bool,
    adjustment_of_annotations: str | None,
    skip_if_invalid_ladder: bool,
) -> dict[str, Any]:
    return {
        'cache_version': CACHE_VERSION,
        'root': str(Path(root).resolve()),
        'scaling': scaling_strategy.cache_signature(),
        'dataset': dataset_strategy.cache_signature(),
        'data_loading_strategy': data_loading_strategy,
        'include_size_standard': include_size_standard,
        'adjustment_of_annotations': adjustment_of_annotations,
        'skip_if_invalid_ladder': skip_if_invalid_ladder,
    }


def compute_key(
    root: Path,
    scaling_strategy: ScalingStrategy,
    dataset_strategy: DatasetStrategy,
    data_loading_strategy: str,
    include_size_standard: bool,
    adjustment_of_annotations: str | None,
    skip_if_invalid_ladder: bool,
) -> str:
    """16-char hex cache key derived from config only."""
    payload = _config_payload(
        root=root,
        scaling_strategy=scaling_strategy,
        dataset_strategy=dataset_strategy,
        data_loading_strategy=data_loading_strategy,
        include_size_standard=include_size_standard,
        adjustment_of_annotations=adjustment_of_annotations,
        skip_if_invalid_ladder=skip_if_invalid_ladder,
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _source_stamps(source_paths: Iterable[Path]) -> list[dict[str, Any]]:
    stamps = []
    for p in source_paths:
        try:
            st = Path(p).stat()
            stamps.append({'path': str(p), 'mtime_ns': st.st_mtime_ns, 'size': st.st_size})
        except FileNotFoundError:
            stamps.append({'path': str(p), 'mtime_ns': None, 'size': None})
    stamps.sort(key=lambda s: s['path'])
    return stamps


def compute_fingerprint(
    config_payload: dict[str, Any],
    source_paths: Iterable[Path],
) -> dict[str, Any]:
    """Build the fingerprint dict (config + source stamps + combined hash)."""
    stamps = _source_stamps(source_paths)
    digest_input = json.dumps({'config': config_payload, 'sources': stamps}, sort_keys=True).encode()
    return {
        'cache_version': CACHE_VERSION,
        'hash': hashlib.sha256(digest_input).hexdigest(),
        'config': config_payload,
        'sources': stamps,
    }


def write_fingerprint(cache_dir: Path, fingerprint: dict[str, Any]) -> None:
    (Path(cache_dir) / FINGERPRINT_JSON).write_text(json.dumps(fingerprint, indent=2))


def read_fingerprint(cache_dir: Path) -> dict[str, Any] | None:
    p = Path(cache_dir) / FINGERPRINT_JSON
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def validate_fingerprint(
    cache_dir: Path,
    config_payload: dict[str, Any],
    source_paths: Iterable[Path],
) -> bool:
    """Return True if the on-disk fingerprint matches the current config+sources."""
    existing = read_fingerprint(cache_dir)
    if existing is None:
        return False
    expected = compute_fingerprint(config_payload, source_paths)
    return existing.get('hash') == expected['hash']


def build_config_payload(
    root: Path,
    scaling_strategy: ScalingStrategy,
    dataset_strategy: DatasetStrategy,
    data_loading_strategy: str,
    include_size_standard: bool,
    adjustment_of_annotations: str | None,
    skip_if_invalid_ladder: bool,
) -> dict[str, Any]:
    """Public helper so HIDDataset can compute the payload once."""
    return _config_payload(
        root=root,
        scaling_strategy=scaling_strategy,
        dataset_strategy=dataset_strategy,
        data_loading_strategy=data_loading_strategy,
        include_size_standard=include_size_standard,
        adjustment_of_annotations=adjustment_of_annotations,
        skip_if_invalid_ladder=skip_if_invalid_ladder,
    )
