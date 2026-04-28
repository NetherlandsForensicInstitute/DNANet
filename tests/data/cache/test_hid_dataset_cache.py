"""Integration tests for ``HIDDataset`` cache lifecycle.

These tests exercise the path from ``HIDDataset.__init__`` through to
``__getitem__`` against the small RD test resources, focusing on cache
behaviour rather than parsing correctness (covered by
``tests/data/test_hid_dataset.py``):

- first-run build creates a key dir
- second-run hits the warm cache (no rebuild)
- touching a source HID invalidates the fingerprint
- a config flip routes to a fresh key
- ``allow_missing_annotations=True`` survives ``None`` annotations
- ``load_in_memory=True`` materializes; the RAM guard refuses oversized caches
"""

from __future__ import annotations

import os
import time

import pytest

from tests.conftest import RD_DIR
from dnanet.data.strategies import NFIRnDStrategy, PowerPlexFusion6CStrategy
from dnanet.data.hid_dataset import HIDDataset
from dnanet.data.cache.layout import is_complete


def _make(cache_dir, **kwargs) -> HIDDataset:
    return HIDDataset(
        root=RD_DIR,
        cache_dir=str(cache_dir),
        scaling_strategy=PowerPlexFusion6CStrategy(),
        dataset_strategy=NFIRnDStrategy('DTH'),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Build → reload
# ---------------------------------------------------------------------------


class TestCacheBuildAndReload:
    def test_first_run_creates_complete_cache(self, tmp_path):
        ds = _make(tmp_path)

        cache_dir = ds._cache_dir  # type: ignore[attr-defined]
        assert is_complete(cache_dir)
        assert (cache_dir / 'data.bin').exists()
        assert (cache_dir / 'index.parquet').exists()
        assert (cache_dir / 'panels.json').exists()

    def test_second_run_hits_warm_cache(self, tmp_path):
        ds1 = _make(tmp_path)
        cache_dir = ds1._cache_dir  # type: ignore[attr-defined]
        complete_mtime = (cache_dir / '_COMPLETE').stat().st_mtime

        # Reload — must NOT rewrite _COMPLETE (warm hit).
        time.sleep(0.01)  # ensure mtime delta would be visible
        ds2 = _make(tmp_path)
        assert ds2._cache_dir == cache_dir  # type: ignore[attr-defined]
        assert (cache_dir / '_COMPLETE').stat().st_mtime == complete_mtime

    def test_getitem_returns_fully_materialized_image(self, tmp_path):
        ds = _make(tmp_path)
        img = ds[0]
        assert img.data is not None
        assert img.data.shape == (5, 4096)


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    def test_touching_source_invalidates_fingerprint(self, tmp_path):
        ds1 = _make(tmp_path)
        cache_dir = ds1._cache_dir  # type: ignore[attr-defined]
        original_complete = (cache_dir / '_COMPLETE').stat().st_mtime

        # Bump every source HID's mtime to simulate an edit.
        time.sleep(0.01)
        for hid in RD_DIR.rglob('*.hid'):
            now_ns = time.time_ns()
            os.utime(hid, ns=(now_ns, now_ns))

        ds2 = _make(tmp_path)
        # Same key directory (config unchanged) but the cache must have been
        # rebuilt — _COMPLETE was rewritten.
        assert ds2._cache_dir == cache_dir  # type: ignore[attr-defined]
        assert (cache_dir / '_COMPLETE').stat().st_mtime > original_complete

    def test_config_flip_routes_to_new_key(self, tmp_path):
        ds1 = _make(tmp_path, include_size_standard=False)
        ds2 = _make(tmp_path, include_size_standard=True)
        assert ds1._cache_dir != ds2._cache_dir  # type: ignore[attr-defined]
        # Both directories live under the same root.
        assert ds1._cache_dir.parent == ds2._cache_dir.parent  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# allow_missing_annotations
# ---------------------------------------------------------------------------


class TestAllowMissingAnnotations:
    def test_flag_changes_cache_key(self, tmp_path):
        """The flag must be part of the config hash so that an annotation-aware
        build can't be served from an annotation-less cache (or vice versa).
        """
        a = _make(tmp_path, allow_missing_annotations=False)
        b = _make(tmp_path, allow_missing_annotations=True)
        assert a._cache_dir != b._cache_dir  # type: ignore[attr-defined]

    def test_flag_recorded_on_dataset(self, tmp_path):
        ds = _make(tmp_path, allow_missing_annotations=True)
        assert ds.allow_missing_annotations is True

    # The full ``annotation=None`` round-trip — writer placeholder, parquet
    # has_annotation=False, reader/HIDDataset surfacing None — is covered at
    # writer/reader level in ``test_writer_reader.py::TestNullableAnnotations``.
    # We don't have an HID source fixture without a matching annotation entry,
    # so an end-to-end HIDDataset assertion here would be brittle.


# ---------------------------------------------------------------------------
# load_in_memory + RAM guard
# ---------------------------------------------------------------------------


class TestRamGuard:
    def test_load_in_memory_small_cache_succeeds(self, tmp_path):
        ds = _make(tmp_path, load_in_memory=True)
        # Reader's data store should now be a regular ndarray, not a memmap.
        import numpy as np

        # Force the memmap-or-ndarray attr to exist (lazy on first get_row).
        ds[0]
        assert not isinstance(ds._reader._data_mm, np.memmap)  # type: ignore[attr-defined]

    def test_ram_guard_refuses_oversized_cache(self, tmp_path, monkeypatch):
        """Stub a tiny total-RAM so the budget falls below ``memmap_bytes()``
        and the guard fires.

        ``budget = int(total_ram * 0.5)`` — we need a value high enough that
        the integer truncation leaves a non-zero budget (otherwise the guard
        treats a zero budget as "RAM size unknown" and skips), but lower than
        the cache size.
        """
        # Build with disk-only first so there's a cache to materialize.
        _make(tmp_path)

        monkeypatch.setattr(HIDDataset, '_total_ram_bytes', staticmethod(lambda: 1000))
        with pytest.raises(RuntimeError, match='load_in_memory=True refused'):
            _make(tmp_path, load_in_memory=True)
