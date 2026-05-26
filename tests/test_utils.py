import pytest

from DNAnet.utils import get_noc_from_rd_file_name, get_prefix_from_filename, is_rd_hid_filename


@pytest.mark.parametrize('filename, expected', [
    ('1A2_A01_01.hid', True),
    ('1A2_A01_01', True),
    ('', False),
    ('12345_A01_01.hid', False),
    ('1234567_20230101001_ABCD1234NL#01_A01', False)])
def test_is_rd_hid_filename(filename, expected):
    assert is_rd_hid_filename(filename) == expected


def test_get_prefix_from_filename():
    assert get_prefix_from_filename('1A2_A01_01.hid') == '1A2'
    with pytest.raises(ValueError) as e:
        get_prefix_from_filename("This is some arbitrary filename")
    assert "Cannot take prefix" in str(e)


@pytest.mark.parametrize('filename, expected', [
    ('1A2_A01_01.hid', '2p'),
    ('2E5_B07_02', '5p'),
    ('1234_A01.hid', None)])
def test_get_noc_from_rd_file_name(filename, expected):
    assert get_noc_from_rd_file_name(filename) == expected
