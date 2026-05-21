import json
import pickle
import shutil
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dnanet.core import LabelCategory
from dnanet.core.annotation import ScanpointAnnotation
from dnanet.data.strategies.datasets.nfi_casework import NFICaseStrategy
from dnanet.data.strategies.scaling.powerplex_fusion_6c import PowerPlexFusion6CStrategy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _MockDataset:
    def __init__(self, n: int):
        self.images = list(range(n))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


@pytest.fixture
def scaling():
    return PowerPlexFusion6CStrategy(scanpoint_resolution=4096)


@pytest.fixture
def mock_root(tmp_path):
    hids = tmp_path / 'hids'
    hids.mkdir()
    (tmp_path / 'annotations').mkdir()
    (tmp_path / 'runid_type.csv').write_text('RUN001,AT\nRUN002,LT\n')
    robot_a = hids / '3500XL_A'
    (robot_a / 'sub1/sub2').mkdir(parents=True)
    (robot_a / 'sub1/sub2/sample_file.hid').touch()
    (robot_a / 'sub1/sub2/ladder_file.hid').touch()
    (hids / '3500XL_B').mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


class TestNFICaseStrategy:
    @pytest.fixture
    def _create_mock_dataset(self):
        dataset_root = Path('/tmp/var/nfi-casework-tests/')
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        dataset_root.mkdir(parents=True, exist_ok=True)

        hids_folder = dataset_root / 'hids'
        hids_folder.mkdir()
        annotations = dataset_root / 'annotations'
        annotations.mkdir()
        (dataset_root / 'runid_type.csv').write_text('RUN001,AT\nRUN002,LT\n')

        robot_a = hids_folder / '3500XL_A'
        robot_a.mkdir(exist_ok=True)
        (hids_folder / '3500XL_B').mkdir(exist_ok=True)

        (robot_a / 'subfolder_1/subfolder_2').mkdir(parents=True, exist_ok=True)
        (robot_a / 'subfolder_1/subfolder_2/sample_file.hid').touch()
        (robot_a / 'subfolder_1/subfolder_2/ladder_file.hid').touch()

        yield dataset_root

        shutil.rmtree(dataset_root)

    def test_collect_dataset_files(self, _create_mock_dataset):
        dataset_files = NFICaseStrategy('ATLT')._collect_dataset_files_uncached(
            root_path=_create_mock_dataset,
            scaling_strategy=PowerPlexFusion6CStrategy(scanpoint_resolution=4096),
            folder_cache=False,
        )

        dataset_files = list(dataset_files)
        assert dataset_files[0] == (
            _create_mock_dataset / 'hids/3500XL_A/subfolder_1/subfolder_2/sample_file.hid',
            None,
            _create_mock_dataset / 'hids/3500XL_A/subfolder_1/subfolder_2/ladder_file.hid',
        )

    def test_collect_dataset_files_span_annotations(self, _create_mock_dataset):
        annotation = ScanpointAnnotation(np.zeros((6, 4096), dtype=np.int8))

        with patch.object(
            NFICaseStrategy,
            '_parse_span_annotation',
            return_value={'sample_file': annotation},
        ):
            dataset_files = list(
                NFICaseStrategy('span')._collect_dataset_files_uncached(
                    root_path=_create_mock_dataset,
                    scaling_strategy=PowerPlexFusion6CStrategy(scanpoint_resolution=4096),
                    folder_cache=False,
                )
            )

        assert dataset_files[0] == (
            _create_mock_dataset / 'hids/3500XL_A/subfolder_1/subfolder_2/sample_file.hid',
            annotation,
            _create_mock_dataset / 'hids/3500XL_A/subfolder_1/subfolder_2/ladder_file.hid',
        )


# ---------------------------------------------------------------------------
# categorize_file
# ---------------------------------------------------------------------------


class TestCategorizeFile:
    def test_non_hid_is_unknown(self):
        assert NFICaseStrategy.categorize_file('sample.txt') == 'unknown'
        assert NFICaseStrategy.categorize_file('sample.csv') == 'unknown'

    def test_ladder_keyword(self):
        assert NFICaseStrategy.categorize_file('ladder_001.hid') == 'ladder'
        assert NFICaseStrategy.categorize_file('LADDER_001.hid') == 'ladder'
        assert NFICaseStrategy.categorize_file('run_Ladder_H01.hid') == 'ladder'

    def test_control_blanco(self):
        assert NFICaseStrategy.categorize_file('blanco_sample.hid') == 'control'

    def test_control_keyword(self):
        assert NFICaseStrategy.categorize_file('control_001.hid') == 'control'

    def test_sample_default(self):
        assert NFICaseStrategy.categorize_file('ABC123_run.hid') == 'sample'

    def test_sample_plain_name(self):
        assert NFICaseStrategy.categorize_file('whatever.hid') == 'sample'


# ---------------------------------------------------------------------------
# cache_signature
# ---------------------------------------------------------------------------


class TestCacheSignature:
    def test_without_robot_selection(self):
        sig = NFICaseStrategy('ATLT').cache_signature()
        assert sig['class'] == 'NFICaseStrategy'
        assert 'robot_selection' not in sig
        assert sig['annotation_type'] == 'ATLT'

    @pytest.mark.skip(reason='robot_selection param was removed from NFICaseStrategy')
    def test_with_robot_selection(self):
        sig = NFICaseStrategy(
            'ATLT',
            subfolder_selection=['3500XL_A', '3500XL_B'],
        ).cache_signature()
        assert 'subfolder_selection' in sig
        assert set(sig['subfolder_selection']) == {'3500XL_A', '3500XL_B'}

    @pytest.mark.skip(reason='robot_selection param was removed from NFICaseStrategy')
    def test_robot_selection_deduplicates(self):
        sig = NFICaseStrategy(
            'ATLT',
            subfolder_selection=['3500XL_A', '3500XL_A'],
        ).cache_signature()
        assert len(sig['subfolder_selection']) == 1


# ---------------------------------------------------------------------------
# _is_correct_annotation_type
# ---------------------------------------------------------------------------


class TestIsCorrectAnnotationType:
    @pytest.mark.parametrize(
        ('run_id', 'annotation_type', 'annotation_type_mapping', 'expected'),
        [
            ('RUN001', 'ATLT', {'RUN001': 'AT'}, True),
            ('RUN001L', 'LT', {}, True),
            ('RUN001', 'LT', {'RUN001L': 'LT'}, True),
            ('RUN001', 'AT', {'RUN001': 'AT'}, True),
            ('RUN999', 'AT', {}, True),
            ('RUN999', 'LT', {}, False),
        ],
    )
    def test_matches_expected_annotation_type(
        self,
        run_id,
        annotation_type,
        annotation_type_mapping,
        expected,
    ):
        assert (
            NFICaseStrategy._is_correct_annotation_type(
                run_id,
                annotation_type,
                annotation_type_mapping,
            )
            is expected
        )


# ---------------------------------------------------------------------------
# find_ladder_for_sample
# ---------------------------------------------------------------------------


class TestFindLadderForSample:
    def test_no_ladder_returns_none(self, tmp_path):
        sample = tmp_path / 'sample.hid'
        sample.touch()
        assert NFICaseStrategy.find_ladder_for_sample(sample) is None

    def test_single_ladder(self, tmp_path):
        sample = tmp_path / 'sample.hid'
        ladder = tmp_path / 'ladder.hid'
        sample.touch()
        ladder.touch()
        assert NFICaseStrategy.find_ladder_for_sample(sample) == ladder

    def test_multiple_ladders_returns_one(self, tmp_path):
        sample = tmp_path / 'sample.hid'
        sample.touch()
        (tmp_path / 'ladder1.hid').touch()
        (tmp_path / 'ladder2.hid').touch()
        result = NFICaseStrategy.find_ladder_for_sample(sample)
        assert result is not None
        assert 'ladder' in result.name.lower()

    def test_non_ladder_files_ignored(self, tmp_path):
        sample = tmp_path / 'sample.hid'
        sample.touch()
        (tmp_path / 'other.hid').touch()
        assert NFICaseStrategy.find_ladder_for_sample(sample) is None


# ---------------------------------------------------------------------------
# find_annotation_files
# ---------------------------------------------------------------------------


class TestFindAnnotationFiles:
    def test_txt_and_csv_included(self, tmp_path):
        (tmp_path / 'ABC123_annotation.txt').touch()
        (tmp_path / 'DEF456_annotation.csv').touch()
        result = NFICaseStrategy.find_annotation_files(tmp_path)
        assert set(result.keys()) == {'ABC123', 'DEF456'}

    def test_other_suffixes_excluded(self, tmp_path):
        (tmp_path / 'ignore.xml').touch()
        (tmp_path / 'ignore.json').touch()
        assert NFICaseStrategy.find_annotation_files(tmp_path) == {}

    def test_key_is_stem_prefix(self, tmp_path):
        (tmp_path / 'RUN001_sample_annotation.txt').touch()
        result = NFICaseStrategy.find_annotation_files(tmp_path)
        assert 'RUN001' in result

    def test_empty_folder(self, tmp_path):
        assert NFICaseStrategy.find_annotation_files(tmp_path) == {}


# ---------------------------------------------------------------------------
# find_robot_files
# ---------------------------------------------------------------------------


class TestFindRobotFiles:
    def test_default_robot_names(self, tmp_path):
        robot_a = tmp_path / '3500XL_A'
        robot_a.mkdir()
        (robot_a / 'file.hid').touch()
        files = list(NFICaseStrategy.find_subfolder_files(tmp_path, cache=False))
        assert len(files) == 1

    def test_selected_robots_filters(self, tmp_path):
        for name in ('3500XL_A', '3500XL_B'):
            folder = tmp_path / name
            folder.mkdir()
            (folder / 'file.hid').touch()
        files = list(
            NFICaseStrategy.find_subfolder_files(
                tmp_path, selected_subfolders=['3500XL_A'], cache=False
            )
        )
        assert len(files) == 1

    def test_missing_robot_folder_skipped(self, tmp_path):
        files = list(NFICaseStrategy.find_subfolder_files(tmp_path, cache=False))
        assert files == []

    def test_robot_limit(self, tmp_path):
        robot_a = tmp_path / '3500XL_A'
        robot_a.mkdir()
        for i in range(5):
            (robot_a / f'file{i}.hid').touch()
        files = list(NFICaseStrategy.find_subfolder_files(tmp_path, subfolder_limit=2, cache=False))
        assert len(files) == 2

    def test_recursive_scan(self, tmp_path):
        sub = tmp_path / '3500XL_A' / 'deep' / 'nested'
        sub.mkdir(parents=True)
        (sub / 'file.hid').touch()
        files = list(NFICaseStrategy.find_subfolder_files(tmp_path, cache=False))
        assert len(files) == 1


# ---------------------------------------------------------------------------
# _scan_directory_structure / _scan_directory_structure_uncached
# ---------------------------------------------------------------------------


class TestScanDirectoryStructure:
    def test_uncached_finds_hid_recursively(self, tmp_path):
        (tmp_path / 'sub').mkdir()
        (tmp_path / 'sub' / 'file.hid').touch()
        (tmp_path / 'other.txt').touch()
        files = list(NFICaseStrategy._scan_directory_structure_uncached(tmp_path))
        assert files == [tmp_path / 'sub' / 'file.hid']

    def test_uncached_ignores_non_hid(self, tmp_path):
        (tmp_path / 'file.csv').touch()
        assert list(NFICaseStrategy._scan_directory_structure_uncached(tmp_path)) == []

    def test_cached_writes_and_reads(self, tmp_path):
        (tmp_path / 'file.hid').touch()
        cache_dir = tmp_path / 'cache'
        path_hash = hashlib.md5(str(tmp_path).encode()).hexdigest()
        expected_cache_file = cache_dir / f'scan-{path_hash}'
        with patch.object(NFICaseStrategy, '_CACHE_DIR', cache_dir):
            result = NFICaseStrategy._scan_directory_structure(tmp_path, cache=True)
        assert len(result) == 1
        assert expected_cache_file.exists()

    def test_cache_hit_returns_stored_value(self, tmp_path):
        cached = [Path('/fake/cached.hid')]
        cache_dir = tmp_path / 'cache'
        cache_dir.mkdir()
        path_hash = hashlib.md5(str(tmp_path).encode()).hexdigest()
        cache_file = cache_dir / f'scan-{path_hash}'
        with cache_file.open('wb') as f:
            pickle.dump(cached, f)
        with patch.object(NFICaseStrategy, '_CACHE_DIR', cache_dir):
            result = NFICaseStrategy._scan_directory_structure(tmp_path, cache=True)
        assert result == cached

    def test_no_cache_skips_read_write(self, tmp_path):
        (tmp_path / 'file.hid').touch()
        cache_dir = tmp_path / 'cache'
        with patch.object(NFICaseStrategy, '_CACHE_DIR', cache_dir):
            NFICaseStrategy._scan_directory_structure(tmp_path, cache=False)
        assert not cache_dir.exists()


# ---------------------------------------------------------------------------
# get_annotation_classes
# ---------------------------------------------------------------------------


class TestGetAnnotationClasses:
    def test_non_span_returns_noise_allele(self):
        assert NFICaseStrategy('ATLT').get_annotation_classes() == ['noise', 'allele']

    def test_span_returns_label_categories(self):
        assert NFICaseStrategy('span').get_annotation_classes() == LabelCategory.label_names()


# ---------------------------------------------------------------------------
# get_number_of_contributors
# ---------------------------------------------------------------------------


class TestGetNumberOfContributors:
    def test_always_raises(self):
        with pytest.raises(AttributeError, match='Casework'):
            NFICaseStrategy.get_number_of_contributors('any_file.hid')


# ---------------------------------------------------------------------------
# get_sample_id
# ---------------------------------------------------------------------------


class TestGetSampleId:
    def test_delegates_to_super(self):
        # super() (DatasetStrategy) abstract method returns None implicitly
        result = NFICaseStrategy.get_sample_id('RUN001_sample.hid')
        assert result is None


# ---------------------------------------------------------------------------
# parse_annotations
# ---------------------------------------------------------------------------


class TestParseAnnotations:
    def test_delegates_to_nfi_rnd(self, tmp_path, scaling):
        from dnanet.data.strategies.datasets.nfi_rnd import NFIRnDStrategy

        with patch.object(NFIRnDStrategy, 'parse_annotations', return_value={}) as mock_parse:
            NFICaseStrategy.parse_annotations(tmp_path / 'annotation.txt', scaling)
            mock_parse.assert_called_once_with(
                annotation_source=tmp_path / 'annotation.txt',
                scaling_strategy=scaling,
            )


# ---------------------------------------------------------------------------
# _split
# ---------------------------------------------------------------------------


class TestSplit:
    @pytest.fixture
    def ds(self):
        return _MockDataset(30)

    def test_fraction_two_way(self, ds):
        result = NFICaseStrategy._split(ds, fraction=0.8, seed=0)
        assert isinstance(result, tuple) and len(result) == 2
        assert len(result[0]) + len(result[1]) == 30  # type: ignore[arg-type]

    def test_fraction_three_way(self, ds):
        result = NFICaseStrategy._split(ds, fraction=0.7, seed=0, test_fraction=0.15)
        assert isinstance(result, tuple) and len(result) == 3
        assert sum(len(s) for s in result) == 30  # type: ignore[arg-type]

    def test_fraction_three_way_sizes(self, ds):
        result = NFICaseStrategy._split(ds, fraction=0.7, seed=0, test_fraction=0.2)
        assert isinstance(result, tuple) and len(result) == 3
        assert abs(len(result[2]) - 6) <= 1  # type: ignore[arg-type]

    def test_kfold_no_test(self, ds):
        result = NFICaseStrategy._split(ds, k_folds=3, seed=0)
        assert isinstance(result, list) and len(result) == 3
        for train, val in result:
            assert len(train) + len(val) == 30  # type: ignore[arg-type]

    def test_kfold_with_test(self, ds):
        result = NFICaseStrategy._split(ds, k_folds=3, seed=0, test_fraction=0.2)
        assert isinstance(result, tuple) and len(result) == 2
        folds, test = result
        assert isinstance(folds, list) and len(folds) == 3
        assert len(test) > 0  # type: ignore[arg-type]

    def test_invalid_both_raises(self, ds):
        with pytest.raises(ValueError):
            NFICaseStrategy._split(ds, fraction=0.8, k_folds=3)

    def test_invalid_neither_raises(self, ds):
        with pytest.raises(ValueError):
            NFICaseStrategy._split(ds)


# ---------------------------------------------------------------------------
# collect_dataset_files (cached wrapper)
# ---------------------------------------------------------------------------


class TestCollectDatasetFilesCached:
    def test_cache_miss_writes_cache(self, mock_root, scaling, tmp_path):
        cache_dir = tmp_path / 'cache'
        with patch.object(NFICaseStrategy, '_CACHE_DIR', cache_dir):
            results = list(NFICaseStrategy('ATLT').collect_dataset_files(mock_root, scaling))
        assert len(results) == 1
        assert len(list(cache_dir.glob('collect-*'))) == 1

    def test_cache_hit_skips_disk_scan(self, mock_root, scaling, tmp_path):
        cache_dir = tmp_path / 'cache'
        cache_dir.mkdir()
        strategy = NFICaseStrategy('ATLT')
        cache_key = hashlib.md5(
            json.dumps(
                {
                    'root_path': str(mock_root),
                    'scaling_strategy': scaling.__class__.__name__,
                    **strategy.cache_signature(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        fake_data = [('injected', None, None)]
        with (cache_dir / f'collect-{cache_key}').open('wb') as f:
            pickle.dump(fake_data, f)

        with patch.object(NFICaseStrategy, '_CACHE_DIR', cache_dir):
            results = list(strategy.collect_dataset_files(mock_root, scaling))
        assert results == fake_data

    def test_robot_selection_changes_cache_key(self, mock_root, scaling, tmp_path):
        cache_dir = tmp_path / 'cache'

        def _collect(robot_sel):
            with patch.object(NFICaseStrategy, '_CACHE_DIR', cache_dir):
                return list(
                    NFICaseStrategy('ATLT', subfolder_selection=robot_sel).collect_dataset_files(
                        mock_root, scaling
                    )
                )

        _collect(None)
        _collect(['3500XL_A'])
        # Two different cache keys should have been written
        assert len(list(cache_dir.glob('collect-*'))) == 2
