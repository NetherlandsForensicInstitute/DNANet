import pytest

from DNAnet.data.data_models.base import SimpleDataset
from DNAnet.data.data_models.hid_image import HIDImage


@pytest.fixture
def simple_dataset():
    return SimpleDataset(
        data=[HIDImage(path=f'dummy_path_{i}') for i in range(20)],
        shuffle=False
    )


def test_split_dataset(simple_dataset):
    assert len(simple_dataset) == 20
    d1, d2 = simple_dataset.split(fraction=0.8, seed=42)
    assert len(d1), len(d2) == (16, 4)
    assert [str(im.path) for im in d2] == ['dummy_path_4', 'dummy_path_14', 'dummy_path_5',
                                           'dummy_path_19']
    # check that shuffling remains the same when using a different fraction but the same seed
    d1, d2 = simple_dataset.split(fraction=0.2, seed=42)
    assert len(d1), len(d2) == (4, 16)
    d2_paths = [str(im.path) for im in d2]
    assert all(p in d2_paths for p in ['dummy_path_4', 'dummy_path_14', 'dummy_path_5',
                                       'dummy_path_19'])

    simple_dataset = SimpleDataset(data=simple_dataset._data[1:], shuffle=False)
    d1, d2 = simple_dataset.split(fraction=0.78)
    assert len(d1), len(d2) == (14, 5)  # 0.78*20=14.82, so split index is rounded down
