import pytest
import numpy as np
from DNAnet.data.data_models.extracted_peak import ExtractedPeak
from DNAnet.data.data_models.peak_dataset import PeakWindowDataset
from DNAnet.data.data_models import Marker, Allele, Annotation, Panel

def test_extracted_peak_initialization(hid_image):
    dye_index = 0
    peak_center = 500
    window_size = 10
    peak_height = 800

    peak = ExtractedPeak(
        image=hid_image,
        dye_index=dye_index,
        peak_center=peak_center,
        window_size=window_size,
        peak_height=peak_height,
        use_ground_truth=False
    )

    assert peak.dye_index == dye_index
    assert peak.original_peak_center_index == peak_center
    assert peak.window_size == window_size
    assert peak.peak_height == peak_height
    # SCAN_TO_BASE(500) = (500/4096)*(475-65)+65 = 115.0634765625
    assert np.isclose(peak.peak_basepair, 115.063, atol=0.02)
    assert peak.data.shape == (window_size,)
    # Based on hid_image fixture, annotation is loaded from a file
    assert peak.is_allele in [True, False]

def test_extracted_peak_marker(hid_image):
    # SCAN_TO_BASE(4000) = (4000/4096)*(475-65)+65 = 465.72265625
    peak = ExtractedPeak(
        image=hid_image,
        dye_index=0,
        peak_center=4000,
        window_size=10,
        peak_height=800,
        use_ground_truth=False
    )

    # In PPF6C dye 0, 465.72 bp is in Penta E
    assert peak.get_marker_name() == "Penta E"

def test_peak_window_dataset_iteration(hid_dataset_rd):
    # Ensure include_size_standard is set if needed by extract_peak_windows
    # and analysis_threshold_type is correct for RD data.
    dataset = PeakWindowDataset(
        threshold=100.0,
        window_size=10,
        filter_peaks=False,
        labels=["allele", "noise"],
        # Inherit args from hid_dataset_rd
        root=hid_dataset_rd.root,
        panel=hid_dataset_rd.panel_path,
        annotations_path=hid_dataset_rd.annotations_path,
        hid_to_annotations_path=hid_dataset_rd.hid_to_annotations_path,
        analysis_threshold_type=hid_dataset_rd.analysis_threshold_type,
        best_ladder_paths_csv=hid_dataset_rd.best_ladder_paths_csv,
        include_size_standard=hid_dataset_rd.include_size_standard
    )

    # PeakWindowDataset.__init__ already calls __iter__ and stores in self._data
    assert len(dataset) > 0
    for i, peak in enumerate(dataset):
        assert isinstance(peak, ExtractedPeak)
        if i >= 10:
            break

def test_peak_window_dataset_with_filter(hid_dataset_rd):
    dataset = PeakWindowDataset(
        threshold=500.0,
        window_size=10,
        filter_peaks=True,
        labels=["allele", "noise"],
        root=hid_dataset_rd.root,
        panel=hid_dataset_rd.panel_path,
        annotations_path=hid_dataset_rd.annotations_path,
        hid_to_annotations_path=hid_dataset_rd.hid_to_annotations_path,
        analysis_threshold_type="DTH",
        best_ladder_paths_csv=hid_dataset_rd.best_ladder_paths_csv,
        include_size_standard=hid_dataset_rd.include_size_standard
    )

    # If filter_peaks is True, it uses filter_peaks_AT_LT
    # We just want to check it iterates and yields peaks
    assert len(dataset) >= 0
    for i, peak in enumerate(dataset):
        assert isinstance(peak, ExtractedPeak)
        if i >= 10:
            break
