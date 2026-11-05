"""Tests for the Hugging Face dataset download helper."""

import sys
from pathlib import Path

import pytest

from dnanet.data.download import ensure_dataset


def _fake_snapshot(created: list[Path]):
    def snapshot_download(**kwargs):
        target = Path(kwargs['local_dir']) / 'dataset'
        target.mkdir(parents=True, exist_ok=True)
        (target / 'file.txt').write_text('x')
        created.append(target)
        return str(kwargs['local_dir'])

    return snapshot_download


class TestEnsureDataset:
    def test_existing_dataset_skips_download(self, tmp_path, monkeypatch):
        root = tmp_path / 'dataset'
        root.mkdir()
        (root / 'file.txt').write_text('x')
        monkeypatch.setitem(sys.modules, 'huggingface_hub', None)

        assert ensure_dataset(root, 'org/name') == root

    def test_missing_dataset_is_downloaded(self, tmp_path, monkeypatch):
        created: list[Path] = []
        fake = type(sys)('huggingface_hub')
        fake.snapshot_download = _fake_snapshot(created)
        monkeypatch.setitem(sys.modules, 'huggingface_hub', fake)

        root = tmp_path / 'data' / 'dataset'
        assert ensure_dataset(root, 'org/name') == root
        assert created == [root]

    def test_empty_directory_is_downloaded(self, tmp_path, monkeypatch):
        created: list[Path] = []
        fake = type(sys)('huggingface_hub')
        fake.snapshot_download = _fake_snapshot(created)
        monkeypatch.setitem(sys.modules, 'huggingface_hub', fake)

        root = tmp_path / 'dataset'
        root.mkdir()
        ensure_dataset(root, 'org/name')
        assert created == [root]

    def test_missing_dependency_raises_import_error(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
        with pytest.raises(ImportError, match='dnanet\\[hub\\]'):
            ensure_dataset(tmp_path / 'dataset', 'org/name')

    def test_unexpected_repo_layout_raises(self, tmp_path, monkeypatch):
        fake = type(sys)('huggingface_hub')
        fake.snapshot_download = lambda **kwargs: ''
        monkeypatch.setitem(sys.modules, 'huggingface_hub', fake)
        with pytest.raises(FileNotFoundError):
            ensure_dataset(tmp_path / 'dataset', 'org/name')
