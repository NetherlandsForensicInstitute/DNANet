import pytest

from DNAnet.allele_callers import NearestBasePairCaller
from DNAnet.data.data_models import Panel


def test_translator_general(hid_image_with_ladder, ladder):
    allele_caller = NearestBasePairCaller()
    predicted_alleles = allele_caller.translate_pixels_to_alleles(hid_image_with_ladder.scaler,
                                                                  hid_image_with_ladder.annotation.image,
                                                                  hid_image_with_ladder.data,
                                                                  hid_image_with_ladder._panel)
    called_alleles = hid_image_with_ladder.meta['called_alleles']
    assert len(predicted_alleles) == len(called_alleles)
    assert sum([len(m.alleles) for m in predicted_alleles]) == sum(
        [len(m.alleles) for m in called_alleles]
    )

    predicted_strings = [
        f"{marker.name}-{allele.name}"
        for marker in predicted_alleles
        for allele in marker.alleles
    ]
    called_strings = [
        f"{marker.name}-{allele.name}"
        for marker in called_alleles
        for allele in marker.alleles
    ]
    assert set(predicted_strings) == set(called_strings)

    # there can be a deviation in allele heights,
    # i.e. the rfu in the data is not the called allele height.
    # Check it is not too large
    max_deviation_rfu = 0.20
    for marker_p, marker_c in zip(predicted_alleles, called_alleles):
        for allele_p, allele_c in zip(sorted(marker_p.alleles, key=lambda x: x.name),
                                      sorted(marker_c.alleles, key=lambda x: x.name)):
            assert allele_p.height * (1 - max_deviation_rfu) < allele_c.height < allele_p.height * (
                        1 + max_deviation_rfu), \
                f'{allele_p} differs too much in rfu from {allele_c}'


def test_translator_with_and_without_ladder(hid_image_with_ladder):
    allele_caller = NearestBasePairCaller()
    default_panel = Panel(pytest.PANEL_PATH)
    scaler, prediction_image, image, panel = hid_image_with_ladder.scaler, \
        hid_image_with_ladder.annotation.image, \
        hid_image_with_ladder.data, \
        hid_image_with_ladder._panel
    predicted_alleles = allele_caller.translate_pixels_to_alleles(
        scaler, prediction_image, image, default_panel)
    predicted_alleles_with_image_panel = allele_caller.translate_pixels_to_alleles(
        scaler, prediction_image, image, panel)
    diff = [(p, q) for p, q in zip(predicted_alleles, predicted_alleles_with_image_panel) if
            set([a.name for a in p.alleles]) != set([a.name for a in q.alleles])]
    assert len(diff) == 1  # there should be one marker with differences
    assert diff[0][0].name == 'D18S51'

    # The ladder can be used to fix the shift in marker D18S51. The shift is as follows:
    assert set([a.name for a in diff[0][0].alleles]) == {'15.1', '16.1', '13.1'}
    assert set([a.name for a in diff[0][1].alleles]) == {'15', '16', '13'}
