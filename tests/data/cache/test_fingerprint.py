"""Unit tests for ``dnanet.data.cache.fingerprint``.

Two distinct concerns are exercised:

* ``compute_key`` is a pure function of *config* — every config field must
  influence the resulting 16-char hash, so identical config produces an
  identical key and any flag flip routes to a fresh cache directory.
* ``compute_fingerprint`` and ``validate_fingerprint`` extend the hash with
  source-file content hashes and act as the cache-staleness oracle on load.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dnanet.data.cache.fingerprint import (
    compute_key,
    read_fingerprint,
    write_fingerprint,
    compute_fingerprint,
    build_config_payload,
    validate_fingerprint,
)


class _StubStrategy:
    """Stand-in for ScalingStrategy / DatasetStrategy with a stable signature."""

    def __init__(self, sig: dict):
        self._sig = sig

    def cache_signature(self) -> dict:
        return self._sig


@pytest.fixture
def base_kwargs(tmp_path):
    return dict(
        root=tmp_path,
        scaling_strategy=_StubStrategy({'kit': 'PPF6C'}),
        dataset_strategy=_StubStrategy({'class': 'NFIRnD'}),
        data_loading_strategy='superior',
        include_size_standard=True,
        adjustment_of_annotations='top',
        skip_if_invalid_ladder=True,
        allow_missing_annotations=False,
    )


# ---------------------------------------------------------------------------
# compute_key — pure function of config
# ---------------------------------------------------------------------------


class TestComputeKey:
    def test_returns_16char_hex(self, base_kwargs):
        key = compute_key(**base_kwargs)
        assert len(key) == 16
        int(key, 16)  # must be valid hex

    def test_identical_config_yields_identical_key(self, base_kwargs):
        assert compute_key(**base_kwargs) == compute_key(**base_kwargs)

    @pytest.mark.parametrize(
        'override',
        [
            {'data_loading_strategy': 'raw'},
            {'include_size_standard': False},
            {'adjustment_of_annotations': 'complete'},
            {'adjustment_of_annotations': None},
            {'skip_if_invalid_ladder': False},
            {'allow_missing_annotations': True},
        ],
    )
    def test_each_flag_changes_key(self, base_kwargs, override):
        a = compute_key(**base_kwargs)
        b = compute_key(**{**base_kwargs, **override})
        assert a != b, f'flipping {override} did not change the cache key'

    def test_strategy_signature_change_changes_key(self, base_kwargs):
        a = compute_key(**base_kwargs)
        b = compute_key(**{**base_kwargs, 'scaling_strategy': _StubStrategy({'kit': 'GlobalFiler'})})
        assert a != b


# ---------------------------------------------------------------------------
# fingerprint round-trip + invalidation
# ---------------------------------------------------------------------------


class TestFingerprint:
    def _payload(self, base_kwargs):
        return build_config_payload(**base_kwargs)

    def test_round_trip_validates(self, tmp_path, base_kwargs):
        payload = self._payload(base_kwargs)
        src = tmp_path / 'a.hid'
        src.write_bytes(b'hello')
        write_fingerprint(tmp_path, compute_fingerprint(payload, [src]))

        assert validate_fingerprint(tmp_path, payload, [src])
        stored = read_fingerprint(tmp_path)
        assert stored is not None
        assert stored['hash'] == compute_fingerprint(payload, [src])['hash']

    def test_no_fingerprint_file_invalidates(self, tmp_path, base_kwargs):
        payload = self._payload(base_kwargs)
        assert not validate_fingerprint(tmp_path, payload, [])

    def test_corrupt_fingerprint_invalidates(self, tmp_path, base_kwargs):
        payload = self._payload(base_kwargs)
        (tmp_path / 'fingerprint.json').write_text('not json')
        assert not validate_fingerprint(tmp_path, payload, [])

    def test_changed_file_invalidates(self, tmp_path, base_kwargs):
        payload = self._payload(base_kwargs)
        src = tmp_path / 'a.hid'
        src.write_bytes(b'v1')
        write_fingerprint(tmp_path, compute_fingerprint(payload, [src]))
        assert validate_fingerprint(tmp_path, payload, [src])

        # Change file content to simulate an edited source.
        src.write_bytes(b'v1-extended')
        assert not validate_fingerprint(tmp_path, payload, [src])

    def test_missing_source_file_does_not_crash(self, tmp_path, base_kwargs):
        payload = self._payload(base_kwargs)
        ghost = tmp_path / 'ghost.hid'
        # File never written; compute_fingerprint must tolerate FileNotFoundError.
        fp = compute_fingerprint(payload, [ghost])
        assert fp['hash']
        # Validation against the same missing-file set should still match.
        write_fingerprint(tmp_path, fp)
        assert validate_fingerprint(tmp_path, payload, [ghost])

    def test_config_change_invalidates(self, tmp_path, base_kwargs):
        payload = self._payload(base_kwargs)
        write_fingerprint(tmp_path, compute_fingerprint(payload, []))

        # Different config → different fingerprint.
        other_payload = build_config_payload(**{**base_kwargs, 'data_loading_strategy': 'raw'})
        assert not validate_fingerprint(tmp_path, other_payload, [])
