import pytest

from DNAnet.data.data_models import Allele, Marker, Panel
from DNAnet.data.parsing import parse_called_alleles


def test_parse_called_alleles():
    markers = parse_called_alleles(
        pytest.RESOURCES_DIR / 'profiles' / 'RD' / 'Dataset 1 DTH_AlleleReport.txt',
        Panel(pytest.PANEL_PATH),
        '1_11148_1A2')

    assert markers[:4] == [
        Marker(dye_row=0, name='AMEL', alleles=[
            Allele(name='X', base_pair=81.5, left_bin=81.0, right_bin=82.0,
                   height=11148.0),
            Allele(name='Y', base_pair=87.68, left_bin=87.18, right_bin=88.18,
                   height=10495.0)]),
        Marker(dye_row=0, name='D3S1358', alleles=[
            Allele(name='14', base_pair=119.49, left_bin=118.99, right_bin=119.99,
                   height=3950.0),
            Allele(name='15', base_pair=123.76, left_bin=123.36, right_bin=124.16000000000001,
                   height=8780.0),
            Allele(name='17', base_pair=132.2, left_bin=131.7, right_bin=132.6,
                   height=6486.0)]),
        Marker(dye_row=0, name='D1S1656', alleles=[
            Allele(name='13', base_pair=174.88, left_bin=174.38, right_bin=175.38,
                   height=8682.0),
            Allele(name='15.3', base_pair=186.12, left_bin=185.62, right_bin=186.52,
                   height=6469.0),
            Allele(name='16', base_pair=187.06, left_bin=186.66, right_bin=187.56,
                   height=2751.0),
            Allele(name='18.3', base_pair=198.27, left_bin=197.77, right_bin=198.67000000000002,
                   height=3051.0)]),
        Marker(dye_row=0, name='D2S441', alleles=[
            Allele(name='11', base_pair=225.08, left_bin=224.58, right_bin=225.58,
                   height=16330.0)])]

    assert len(markers) == 27
