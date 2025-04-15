import logging
import os

import numpy as np
import pytest

from DNAnet.data.data_models import Allele, Marker, Panel
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.evaluation.visualizations import plot_profile


logging.getLogger('matplotlib.font_manager').disabled = True


def test_hid_image(hid_dataset_rd):
    assert len(hid_dataset_rd) == 2

    for hid_image in hid_dataset_rd:
        plot_profile([hid_image])
        image = hid_image.data
        annotation = hid_image.annotation.image
        # the annotation and the image should have the same shape
        assert annotation.shape == hid_image.data.shape

        ref_annotation = np.load(
            f'{pytest.RESOURCES_DIR}/profiles/RD/'
            f'{hid_image.path.stem.split("/")[-1]}_annotation.npy'
        )
        np.testing.assert_equal(annotation, ref_annotation)

        # each dye should have at least 1 annotated peak
        assert np.all(np.sum(annotation[:, :, 0], axis=1) > 0)
        ref_image = np.load(str(hid_image.path).replace(".hid", ".npy"))
        assert np.array_equal(image, ref_image)

        adjusted_image_top = hid_image.adjust_annotations("top")
        plot_profile([adjusted_image_top])
        # the nr of labeled peak tops should be less or equal than the
        # number of called alleles
        n_called_alleles = sum([len(marker.alleles) for marker in hid_image.meta["called_alleles"]])

        top_annotations = adjusted_image_top.annotation.image.copy()
        assert np.sum(top_annotations) <= n_called_alleles  # <= since
        # subthreshold 'tops' are not labeled

        adjusted_image_complete = hid_image.adjust_annotations("complete")
        plot_profile([adjusted_image_complete])
        for layer_peak_top, layer_complete_peak in zip(
            top_annotations, adjusted_image_complete.annotation.image
        ):
            layer_complete_peak = layer_complete_peak.flatten()
            # the number of labeled 'complete peaks' should be less or equal
            # than the number of peak tops
            n_groups_complete = np.count_nonzero(
                np.diff(layer_complete_peak, prepend=0, append=0) == 1
            )
            n_tops = np.count_nonzero(np.diff(layer_peak_top.flatten(), prepend=0, append=0) == 1)
            assert n_groups_complete <= n_tops  # <= since multiple tops may be
            # merged into one group

            # every top annotations should be in a peak annotation
            top_annotations = np.where(layer_peak_top == 1)[0]
            peak_annotations = np.where(layer_complete_peak == 1)[0]
            assert all([top in peak_annotations for top in top_annotations])


def test_hid_image_with_ladder(ladder):
    hid_image = HIDImage(
        path=os.path.join(pytest.RESOURCES_DIR, "profiles", "RD", "1A2_A01_01.hid"),
        annotations_file=os.path.join(pytest.RESOURCES_DIR, "profiles", "RD",
                                      "Dataset 1 DTH_AlleleReport.txt"),
        panel=Panel(pytest.PANEL_PATH),
    )

    assert hid_image._panel is not None
    ref_data = np.load(
        os.path.join(pytest.RESOURCES_DIR, "profiles", "RD",  "1A2_A01_01_data_with_ladder.npy"),
        allow_pickle=True
    )
    assert np.array_equal(hid_image.data, ref_data)


def test_ladder(ladder):
    default_panel = Panel(pytest.PANEL_PATH)
    assert ladder.data.shape == (6, 4096, 1)

    true_nr_peaks_per_dye = [89, 80, 89, 99, 76]
    for peak_idxs, true_nr in zip(ladder.peak_indices, true_nr_peaks_per_dye):
        assert len(peak_idxs) == true_nr

    ladder_panel = ladder.panel._panel
    assert ladder_panel[0] == Marker(
        dye_row=0,
        name="AMEL",
        alleles=[
            Allele(name="X", base_pair=81.43459915611814, left_bin=0.5, right_bin=0.5),
            Allele(name="Y", base_pair=87.59493670886076, left_bin=0.5, right_bin=0.5),
        ],
    )

    for marker_old_panel, marker_ladder_panel in zip(default_panel._panel, ladder_panel):
        assert len(marker_old_panel.alleles) == len(marker_ladder_panel.alleles)

    mapping = {}
    result_allele = ladder._extrapolate_base_pair(
        [93.47, 96],
        [93.12, 95.74],
        marker=Marker(dye_row=0, name="ABC", alleles=[]),
        allele=Allele("1", base_pair=97.82),
        marker_allele_to_bp=mapping,
    )
    assert result_allele == Allele(name="1", base_pair=97.62474308300393)
    assert mapping[("ABC", "1")] == 97.62474308300393
