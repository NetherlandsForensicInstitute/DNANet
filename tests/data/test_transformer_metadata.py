"""Tests for metadata-aware HID transformers."""

from __future__ import annotations

import numpy as np
import torch
from torch.testing import assert_close

import dnanet.data.transformer as transformer_module
from dnanet.core.panel import Panel
from dnanet.data.image import HIDImage
from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.annotation import AlleleAnnotation, ScanpointAnnotation
from dnanet.data.transformer import CombinedTransformerMetaData


def test_combined_transformer_metadata_collate_preserves_sample_metadata(monkeypatch):
    marker = Marker(
        name="D5S818",
        dye_row=0,
        alleles=frozenset([Allele(name="13", base_pair=100.0)]),
    )
    annotation = AlleleAnnotation([marker])
    image = HIDImage(
        path="sample.hid",
        adjusted_panel=Panel(markers=[marker]),
        allele_annotation=annotation,
        load_in_memory=True,
    )
    image._data = np.zeros((1, 8, 1), dtype=np.float32)
    image._annotation = ScanpointAnnotation(data=np.zeros((1, 8), dtype=np.int8))
    image._scaler = np.arange(8)

    monkeypatch.setattr(
        transformer_module,
        "extract_peaks_torch",
        lambda *args, **kwargs: (
            torch.ones((1, 1, 4), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.long),
            torch.zeros((1, 2), dtype=torch.long),
        ),
    )

    sample = CombinedTransformerMetaData(window_size=4)(image)
    inputs, targets, metadata = CombinedTransformerMetaData.collate_fn([sample])

    full_images, peak_windows, marker_idxs, peak_centers, peak_counts = inputs
    assert_close(full_images, torch.zeros((1, 1, 8), dtype=torch.float32))
    assert targets.shape == (1, 1, 8)
    assert peak_windows.shape == (1, 1, 4)
    assert marker_idxs.shape == (1,)
    assert peak_centers.shape == (1, 2)
    assert_close(peak_counts, torch.tensor([1], dtype=torch.long))
    assert metadata[0]["allele_annotation"] is annotation
    assert metadata[0]["panel"] is image.adjusted_panel
    assert metadata[0]["path"] == image.path
    assert_close(torch.as_tensor(metadata[0]["scaler"]), torch.arange(8))
    assert metadata[0]["signal_image"] is image.data
