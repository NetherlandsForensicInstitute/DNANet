"""End-to-end tests for ``MemmapCacheWriter`` + ``MemmapCacheReader``.

Each test focuses on one writer/reader contract:
- happy-path round-trip (data integrity)
- nullable annotations (zero-row placeholder + ``has_annotation`` flag)
- sidecar dedup (``panels.json`` + ``alleles.json``)
- shape validation across rows
- resume after a partial write (manifest truncation)
- finalize idempotence guard (no rows → error)
- in-memory materialization
- pickle safety (DataLoader-worker fork analogue)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from dnanet.data.cache import MemmapCacheReader, MemmapCacheWriter
from dnanet.data.cache.layout import (
    DATA_BIN,
    SCALER_BIN,
    PANELS_JSON,
    SHAPES_JSON,
    ALLELES_JSON,
    INDEX_PARQUET,
    ANNOTATION_BIN,
    MANIFEST_JSONL,
    is_complete,
)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_data_arrays_survive_round_trip(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            for i in range(3):
                w.write(make_image(f'r{i}.hid', fill=i + 1))
            w.finalize({'cache_version': 5}, [Path(f'r{i}.hid') for i in range(3)])

        r = MemmapCacheReader(cd)
        idx = r.load_index()
        assert len(idx) == 3
        for i, e in enumerate(idx):
            data, ann, scaler = r.get_row(i)
            assert np.all(data == i + 1)
            # Annotation is fill % 2 by stub convention.
            assert np.all(ann == (i + 1) % 2)
            assert scaler.shape == (16,)
            assert e.path == f'r{i}.hid'
            assert e.has_annotation is True

    def test_finalize_writes_all_artifacts(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            w.write(make_image('a.hid'))
            w.finalize({}, [Path('a.hid')])

        for name in (
            DATA_BIN,
            ANNOTATION_BIN,
            SCALER_BIN,
            INDEX_PARQUET,
            SHAPES_JSON,
            PANELS_JSON,
            ALLELES_JSON,
        ):
            assert (cd / name).exists(), f'{name} should be written'
        assert is_complete(cd)
        # Manifest is the write log; it should be removed at finalize.
        assert not (cd / MANIFEST_JSONL).exists()

    def test_negative_index_wraps(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            w.write(make_image('a.hid', fill=1))
            w.write(make_image('b.hid', fill=2))
            w.finalize({}, [])

        r = MemmapCacheReader(cd)
        d, *_ = r.get_row(-1)
        assert np.all(d == 2)

    def test_out_of_range_raises(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            w.write(make_image('a.hid'))
            w.finalize({}, [])

        r = MemmapCacheReader(cd)
        with pytest.raises(IndexError):
            r.get_row(99)


# ---------------------------------------------------------------------------
# Nullable annotations
# ---------------------------------------------------------------------------


class TestNullableAnnotations:
    def test_writes_zero_placeholder_and_flag(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            w.write(make_image('annotated.hid', annotated=True, fill=1))
            w.write(make_image('unannotated.hid', annotated=False, fill=3))
            w.finalize({}, [])

        r = MemmapCacheReader(cd)
        idx = r.load_index()
        assert idx[0].has_annotation is True
        assert idx[1].has_annotation is False

        # Unannotated row's annotation memmap slot must be zero-filled —
        # this is what HIDDataset relies on to safely return None.
        _, ann_unann, _ = r.get_row(1)
        assert np.all(ann_unann == 0)


# ---------------------------------------------------------------------------
# Sidecar dedup
# ---------------------------------------------------------------------------


class TestSidecarDedup:
    def test_repeated_panel_collapses_to_one_entry(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            for i in range(5):
                w.write(make_image(f'r{i}.hid', panel_id='SHARED', allele_id='SHARED'))
            w.finalize({}, [])

        panels = json.loads((cd / PANELS_JSON).read_text())
        alleles = json.loads((cd / ALLELES_JSON).read_text())
        assert len(panels) == 1
        assert len(alleles) == 1

        r = MemmapCacheReader(cd)
        idx = r.load_index()
        # All five rows point at the same panel/allele entry.
        assert len({e.panel_key for e in idx}) == 1
        assert len({e.allele_key for e in idx}) == 1

    def test_distinct_panels_get_distinct_keys(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            w.write(make_image('a.hid', panel_id='A', allele_id='X'))
            w.write(make_image('b.hid', panel_id='B', allele_id='Y'))
            w.write(make_image('c.hid', panel_id='A', allele_id='Y'))  # mixed reuse
            w.finalize({}, [])

        panels = json.loads((cd / PANELS_JSON).read_text())
        alleles = json.loads((cd / ALLELES_JSON).read_text())
        assert len(panels) == 2
        assert len(alleles) == 2

        idx = MemmapCacheReader(cd).load_index()
        assert idx[0].panel_key == idx[2].panel_key
        assert idx[0].panel_key != idx[1].panel_key
        assert idx[1].allele_key == idx[2].allele_key

    def test_none_panel_yields_null_key(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            w.write(make_image('with.hid', panel_id='A', allele_id='X'))
            w.write(make_image('without.hid', panel_id=None, allele_id=None))
            w.finalize({}, [])

        idx = MemmapCacheReader(cd).load_index()
        assert idx[0].panel_key is not None
        assert idx[1].panel_key is None
        assert idx[1].allele_key is None

    def test_reader_resolves_keys_to_json(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            w.write(make_image('a.hid', panel_id='UNIQUE', allele_id='ALLELE_X'))
            w.finalize({}, [])

        r = MemmapCacheReader(cd)
        e = r.load_index()[0]
        panel_js = r.panel_json(e.panel_key)
        allele_js = r.allele_json(e.allele_key)
        assert 'panel-UNIQUE' in panel_js
        assert 'allele-ALLELE_X' in allele_js
        assert r.panel_json(None) is None
        assert r.allele_json(None) is None


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


class TestShapeValidation:
    def test_shape_mismatch_raises(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            w.write(make_image('a.hid', shape=(6, 16)))
            with pytest.raises(RuntimeError, match='shape'):
                w.write(make_image('b.hid', shape=(6, 32)))


# ---------------------------------------------------------------------------
# Empty finalize
# ---------------------------------------------------------------------------


class TestEmptyFinalize:
    def test_finalize_without_rows_raises(self, tmp_path):
        cd = tmp_path / 'cache'
        with pytest.raises(ValueError, match='empty cache'):
            with MemmapCacheWriter(cd) as w:
                w.finalize({}, [])


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_paths_reports_committed_rows(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        # First "session" — write two rows but don't finalize.
        w1 = MemmapCacheWriter(cd)
        w1.write(make_image('a.hid'))
        w1.write(make_image('b.hid'))
        w1._close()  # type: ignore[attr-defined] — simulate process exit.

        # Second "session" — manifest has 2 rows, .bin files match.
        w2 = MemmapCacheWriter(cd)
        assert w2.resume_paths() == {'a.hid', 'b.hid'}
        # Caller skips the first two and writes a third.
        w2.write(make_image('c.hid'))
        w2.finalize({}, [Path(s) for s in ('a.hid', 'b.hid', 'c.hid')])

        r = MemmapCacheReader(cd)
        idx = r.load_index()
        assert [e.path for e in idx] == ['a.hid', 'b.hid', 'c.hid']

    def test_partial_trailing_bin_is_truncated(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        w1 = MemmapCacheWriter(cd)
        w1.write(make_image('a.hid'))
        w1._close()  # type: ignore[attr-defined]

        # Simulate a crash mid-write of row 2: extend data.bin past the
        # manifest commit (manifest still says n=1) — recovery must trim.
        with (cd / DATA_BIN).open('ab') as f:
            f.write(b'\x00' * 64)

        w2 = MemmapCacheWriter(cd)
        # The recovery truncates data.bin back to the manifest's row count.
        expected = 1 * 6 * 16 * np.dtype(np.int16).itemsize
        assert (cd / DATA_BIN).stat().st_size == expected
        # The cache should still finalize cleanly with the surviving row.
        w2.finalize({}, [Path('a.hid')])
        assert is_complete(cd)


# ---------------------------------------------------------------------------
# In-memory materialize + RAM accounting
# ---------------------------------------------------------------------------


class TestMaterialize:
    def test_memmap_bytes_matches_layout(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            for i in range(4):
                w.write(make_image(f'r{i}.hid'))
            w.finalize({}, [])

        r = MemmapCacheReader(cd)
        # 4 rows × (6×16 int16 + 6×16 int8 + 16 float32)
        per_row = (6 * 16 * 2) + (6 * 16 * 1) + (16 * 4)
        assert r.memmap_bytes() == 4 * per_row

    def test_materialize_swaps_memmap_for_ndarray(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            w.write(make_image('a.hid'))
            w.finalize({}, [])

        r = MemmapCacheReader(cd)
        d_before, _, _ = r.get_row(0)  # opens memmaps
        assert isinstance(r._data_mm, np.memmap)  # type: ignore[attr-defined]

        r.materialize()
        # After materialize, the underlying object is a regular ndarray.
        assert not isinstance(r._data_mm, np.memmap)  # type: ignore[attr-defined]
        assert isinstance(r._data_mm, np.ndarray)  # type: ignore[attr-defined]

        d_after, _, _ = r.get_row(0)
        assert np.array_equal(d_before, d_after)


# ---------------------------------------------------------------------------
# Pickle safety (DataLoader-worker fork analogue)
# ---------------------------------------------------------------------------


class TestPickleSafety:
    def test_pickle_drops_handles_and_caches(self, tmp_path, make_image):
        cd = tmp_path / 'cache'
        with MemmapCacheWriter(cd) as w:
            w.write(make_image('a.hid'))
            w.finalize({}, [])

        r = MemmapCacheReader(cd)
        # Warm everything: memmaps + sidecars.
        r.get_row(0)
        r.panel_json(r.load_index()[0].panel_key)

        blob = pickle.dumps(r)
        restored: MemmapCacheReader = pickle.loads(blob)

        assert restored._data_mm is None  # type: ignore[attr-defined]
        assert restored._panels is None  # type: ignore[attr-defined]
        # The restored reader still works — memmaps reopen on demand.
        d, _, _ = restored.get_row(0)
        assert d.shape == (6, 16)
