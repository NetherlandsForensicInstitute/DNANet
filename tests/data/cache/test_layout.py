"""Unit tests for ``dnanet.data.cache.layout`` — sentinel + key-dir helpers."""

from __future__ import annotations

from dnanet.data.cache.layout import (
    CACHE_VERSION,
    is_complete,
    cache_key_dir,
    mark_complete,
    clear_complete,
)


class TestCompletionMarker:
    def test_unmarked_is_incomplete(self, tmp_path):
        assert not is_complete(tmp_path)

    def test_mark_then_complete(self, tmp_path):
        mark_complete(tmp_path)
        assert is_complete(tmp_path)

    def test_clear_resets(self, tmp_path):
        mark_complete(tmp_path)
        clear_complete(tmp_path)
        assert not is_complete(tmp_path)

    def test_clear_on_unmarked_is_noop(self, tmp_path):
        # Must not raise even if the marker doesn't exist.
        clear_complete(tmp_path)
        assert not is_complete(tmp_path)


class TestCacheKeyDir:
    def test_joins_root_and_key(self, tmp_path):
        out = cache_key_dir(tmp_path, 'abc123')
        assert out == tmp_path / 'abc123'


class TestCacheVersion:
    def test_is_int(self):
        assert isinstance(CACHE_VERSION, int)
        # Bumping the version is a deliberate act; assert it never goes
        # backwards. This guard catches accidental decrements during merges.
        assert CACHE_VERSION >= 5
