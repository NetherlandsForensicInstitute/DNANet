"""Smoke tests for the ``cache-inspect`` CLI tool.

The inspector is intentionally lightweight (it's a debugging aid, not a
runtime dependency) so we only verify it (1) walks both single-key and
parent layouts, (2) reports on the artifacts that exist, and (3) exits
cleanly even when given a half-finalized cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnanet.data.cache import MemmapCacheWriter
from dnanet.data.cache.inspect import main


def _build_finalized_cache(cd: Path, make_image, n: int = 3) -> None:
    with MemmapCacheWriter(cd) as w:
        for i in range(n):
            # Two unique panels across the rows so dedup output is non-trivial.
            w.write(make_image(f'r{i}.hid', panel_id=str(i % 2)))
        w.finalize({}, [Path(f'r{i}.hid') for i in range(n)])


class TestInspect:
    def test_usage_when_no_arg(self, capsys):
        rc = main(['inspect'])
        assert rc == 2
        out = capsys.readouterr()
        assert 'Usage' in out.err

    def test_missing_path(self, tmp_path, capsys):
        rc = main(['inspect', str(tmp_path / 'does-not-exist')])
        assert rc == 1
        assert 'does not exist' in capsys.readouterr().err

    def test_inspects_single_key_dir(self, tmp_path, make_image, capsys):
        cd = tmp_path / 'key'
        _build_finalized_cache(cd, make_image, n=3)

        rc = main(['inspect', str(cd)])
        assert rc == 0
        out = capsys.readouterr().out
        assert 'rows (n):         3' in out
        assert 'data.bin' in out
        assert 'panels.json' in out
        assert 'alleles.json' in out
        # Two unique panels in our setup.
        assert '2 unique' in out

    def test_walks_parent_dir(self, tmp_path, make_image, capsys):
        # Two key directories under one cache root.
        for key in ('aaaa', 'bbbb'):
            _build_finalized_cache(tmp_path / key, make_image, n=2)

        rc = main(['inspect', str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert 'aaaa' in out and 'bbbb' in out

    def test_handles_half_finalized_cache(self, tmp_path, make_image, capsys):
        """An aborted build (manifest present, no _COMPLETE) must not crash."""
        cd = tmp_path / 'partial'
        # Drop a writer mid-build: rows on disk, no finalize.
        w = MemmapCacheWriter(cd)
        w.write(make_image('a.hid'))
        w._close()  # type: ignore[attr-defined]

        # Should print what's there without raising. It still hits the early
        # return when index.parquet is missing.
        rc = main(['inspect', str(cd)])
        assert rc == 0
        assert 'complete marker:  False' in capsys.readouterr().out
