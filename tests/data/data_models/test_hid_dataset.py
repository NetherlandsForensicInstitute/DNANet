import os
import shutil
from collections import Counter

import numpy as np
import pytest

from DNAnet.data.data_models.hid_dataset import HIDDataset
from DNAnet.utils import get_prefix_from_filename


def test_hid_dataset_rd(hid_dataset_rd):
    assert len(hid_dataset_rd) == 2


def test_cached_data():
    cache_path = os.path.join(pytest.RESOURCES_DIR, "test_cache_arrow")
    dataset = HIDDataset(
        root=os.path.join(pytest.RESOURCES_DIR, "profiles", "RD"),
        panel=pytest.PANEL_PATH,
        annotations_path=os.path.join(pytest.RESOURCES_DIR, "profiles", "RD"),
        hid_to_annotations_path=os.path.join(pytest.RESOURCES_DIR, "profiles", "RD",
                                             "rd_hid_annotations_mapping.csv"),
        analysis_threshold_type="DTH",
        use_cache=False,
        cache_path=cache_path,
        skip_if_invalid_ladder=True,
    )

    dataset_cached = HIDDataset(
        root=os.path.join(pytest.RESOURCES_DIR, "profiles", "RD"),
        use_cache=True,
        cache_path=cache_path
    )

    assert len(dataset) == len(dataset_cached)
    assert np.array_equal(dataset._data[0].data, dataset_cached._data[0].data)
    shutil.rmtree(cache_path)


def test_skip_if_invalid_ladder():
    dataset = HIDDataset(
        root=os.path.join(pytest.RESOURCES_DIR, "profile_without_ladder"),
        panel=pytest.PANEL_PATH,
        annotations_path=os.path.join(pytest.RESOURCES_DIR, "profile_without_ladder"),
        hid_to_annotations_path=os.path.join(pytest.RESOURCES_DIR, "profile_without_ladder",
                                             "annotations_mapping.csv"),
        analysis_threshold_type="DTH",
        skip_if_invalid_ladder=False)
    assert len(dataset) == 1

    dataset = HIDDataset(
        root=os.path.join(pytest.RESOURCES_DIR, "profile_without_ladder"),
        panel=pytest.PANEL_PATH,
        annotations_path=os.path.join(pytest.RESOURCES_DIR, "profile_without_ladder"),
        hid_to_annotations_path=os.path.join(pytest.RESOURCES_DIR,
                                             "profile_without_ladder",
                                             "annotations_mapping.csv"),
        analysis_threshold_type="DTH",
        skip_if_invalid_ladder=True)
    assert len(dataset) == 0


def test_split_hid_dataset():
    dataset = HIDDataset(root='root',
                         use_cache=True,
                         cache_path=os.path.join(pytest.RESOURCES_DIR, 'cached_hid_dataset_M1'),
                         group_replicas_in_split=False)
    assert len(dataset) == 58

    d1, d2 = dataset.split(fraction=0.8, seed=42)
    assert len(d1), len(d2) == (46, 12)
    assert [im.path.stem for im in d2] == ['1C2_A08_02', '1B2_E02_14', '1C5_D03_12',
                                           '1D4_C09_09', '1E4_G08_20', '1D5_rerun_D04_10',
                                           '1A2_A06_03', '1C2_A03_03', '1B3_F02_17',
                                           '1D4_rerun_C04_07', '1A5_D06_12', '1B4_C07_07']

    prefixes_d1 = set([get_prefix_from_filename(im.path.stem) for im in d1])
    prefixes_d2 = set([get_prefix_from_filename(im.path.stem) for im in d2])
    assert prefixes_d1.intersection(prefixes_d2) == prefixes_d2

    # now we want to group replicas and balance donors
    dataset = HIDDataset(root='root',
                         use_cache=True,
                         cache_path=os.path.join(pytest.RESOURCES_DIR, 'cached_hid_dataset_M1'),
                         group_replicas_in_split=True)
    d1, d2 = dataset.split(fraction=0.8, seed=42)
    assert len(d1), len(d2) == (47, 11)
    assert [im.path.stem for im in d2] == ['1B3_B02_05', '1A2_A01_01', '1A5_D01_10', '1A5_H01_22',
                                           '1A5_D06_12', '1A2_A06_03', '1B3_B07_04', '1E4_G08_20',
                                           '1B3_F02_17', '1E4_G05_20', '1A2_E01_13']
    prefixes_d1 = set([get_prefix_from_filename(im.path.stem) for im in d1])
    prefixes_d2 = set([get_prefix_from_filename(im.path.stem) for im in d2])
    assert prefixes_d1.intersection(prefixes_d2) == set()
    # check that the number of profiles per noc are balanced
    nr_profiles_per_noc_d1 = Counter([im.path.stem[2] for im in d1])
    assert list(nr_profiles_per_noc_d1.values()) == [12, 12, 12, 11]
    nr_profiles_per_noc_d2 = Counter([im.path.stem[2] for im in d2])
    assert list(nr_profiles_per_noc_d2.values()) == [3, 3, 3, 2]


def test_split_k_fold_dataset():
    dataset = HIDDataset(root='root',
                         use_cache=True,
                         cache_path=os.path.join(pytest.RESOURCES_DIR, 'cached_hid_dataset_M1'),
                         group_replicas_in_split=False)
    assert len(dataset) == 58

    splits = dataset.split_k_fold(n_folds=4, seed=42)
    train, test = splits[0]
    train_prefixes = set([get_prefix_from_filename(im.path.stem) for im in train])
    test_prefixes = set([get_prefix_from_filename(im.path.stem) for im in test])
    assert test_prefixes == {'1B3', '1B2', '1A4', '1A2', '1E2', '1A5', '1B5', '1D5', '1E4', '1C3'}
    assert train_prefixes == {'1D2', '1D4', '1E2', '1B5', '1C2', '1D3', '1C4', '1B2', '1E3', '1C5',
                              '1A3', '1E4', '1B3', '1A2', '1E5', '1B4', '1D5', '1A4', '1C3'}
    nr_profiles_per_noc_train = Counter([im.path.stem[2] for im in train])
    assert list(nr_profiles_per_noc_train.values()) == [8, 12, 12, 11]
    nr_profiles_per_noc_test = Counter([im.path.stem[2] for im in test])
    assert list(nr_profiles_per_noc_test.values()) == [4, 6, 3, 2]

    # now group by replicas and balance number of donors
    dataset = HIDDataset(root='root',
                         use_cache=True,
                         cache_path=os.path.join(pytest.RESOURCES_DIR, 'cached_hid_dataset_M1'),
                         group_replicas_in_split=True)

    splits = dataset.split_k_fold(n_folds=4, seed=42)
    for train, test in splits:
        train_prefixes = set([get_prefix_from_filename(im.path.stem) for im in train])
        test_prefixes = set([get_prefix_from_filename(im.path.stem) for im in test])
        assert train_prefixes.intersection(test_prefixes) == set()

    assert test_prefixes == {'1A3', '1A4', '1A5', '1B2'}
    assert train_prefixes == {'1B5', '1C4', '1A2', '1C3', '1C2', '1E3', '1D4', '1E5',
                              '1E2', '1D2', '1B3', '1E4', '1D5', '1C5', '1B4', '1D3'}
    # check that the number of profiles per noc are balanced
    nr_profiles_per_noc_train = Counter([im.path.stem[2] for im in train])
    assert list(nr_profiles_per_noc_train.values()) == [11, 11, 12, 12]
    nr_profiles_per_noc_test = Counter([im.path.stem[2] for im in test])
    assert list(nr_profiles_per_noc_test.values()) == [3, 3, 3, 3]

    (train_set1, test_set1), (train_set2, test_set2) = splits[0], splits[1]
    paths_test1 = set([im.path for im in test_set1])
    paths_test2 = set([im.path for im in test_set2])
    # check that test sets are disjoint
    assert paths_test2.intersection(paths_test1) == set()
    # check that images in test1 are all in train2 and vice versa
    paths_train2 = set([im.path for im in train_set2])
    assert paths_train2.intersection(paths_test1) == paths_test1
    paths_train1 = set([im.path for im in train_set1])
    assert paths_train1.intersection(paths_test2) == paths_test2
