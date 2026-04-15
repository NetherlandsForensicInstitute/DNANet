"""Tests for data transformers."""

import numpy as np
import torch
from torch.testing import assert_close

import dnanet.data.transformer as transformer_module
from dnanet.core.annotation import ScanpointAnnotation
from dnanet.data.extracted_peak import ExtractedPeak
from dnanet.data.image import HIDImage
from dnanet.data.strategies import PowerPlexFusion6CStrategy, NFIRnDStrategy
from dnanet.data.transformer import (
    CombinedTransformer,
    TransformDataCallable,
    SegmentationTransformer,
    ReconstructionTransformer,
    PeakClassificationTransformer,
)


def _make_fake_image(
    data: np.ndarray | None = None,
    annotation: np.ndarray | None = None,
) -> HIDImage:
    """Build an HIDImage with in-memory data only."""
    img = HIDImage(
        path='fake.hid',
        scaling_strategy=PowerPlexFusion6CStrategy(),
        dataset_strategy=NFIRnDStrategy(),
        load_in_memory=True,
    )
    img._data = data if data is not None else np.zeros((5, 8, 1), dtype=np.float32)
    if annotation is not None:
        img._annotation = ScanpointAnnotation(data=annotation)
    return img


class TestTransformDataCallable:
    def test_collate_fn_uses_default_collate(self):
        batch = [
            (torch.tensor([1.0, 2.0]), torch.tensor(0)),
            (torch.tensor([3.0, 4.0]), torch.tensor(1)),
        ]

        inputs, targets = TransformDataCallable.collate_fn(batch)

        assert inputs.shape == (2, 2)
        assert targets.shape == (2,)
        assert_close(inputs, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert_close(targets, torch.tensor([0, 1]))


class TestSegmentationTransformer:
    def test_returns_float_tensors_with_annotation(self):
        data = np.arange(20, dtype=np.float32).reshape(5, 4, 1)
        annotation = np.zeros((5, 4, 1), dtype=np.int8)
        annotation[2, 1:3, 0] = 1
        image = _make_fake_image(data=data, annotation=annotation)

        x, y = SegmentationTransformer()(image)

        assert x.shape == (5, 4, 1)
        assert y.shape == (5, 4, 1)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
        assert_close(x, torch.tensor(data, dtype=torch.float32))
        assert_close(y, torch.tensor(annotation, dtype=torch.float32))

    def test_returns_zero_target_without_annotation(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4, 1)
        image = _make_fake_image(data=data)

        x, y = SegmentationTransformer()(image)

        assert x.shape == (3, 4, 1)
        assert y.shape == (3, 4, 1)
        assert torch.all(y == 0)


class TestCombinedTransformer:
    def test_returns_full_image_peak_inputs_and_annotation_target(self, monkeypatch):
        data = np.arange(20, dtype=np.float32).reshape(5, 4, 1)
        annotation = np.zeros((5, 4, 1), dtype=np.int8)
        annotation[1, 2, 0] = 1
        image = _make_fake_image(data=data, annotation=annotation)

        peak_windows = torch.arange(12, dtype=torch.float32).reshape(2, 1, 6)
        marker_idxs = torch.tensor([1, 3], dtype=torch.long)
        peak_centers = torch.tensor([[0, 10], [4, 20]], dtype=torch.long)
        captured = {}

        def fake_extract_peaks(
            image_arg,
            *,
            scaling_strategy,
            threshold,
            window_size,
            include_max_pool_dyes,
        ):
            captured['image'] = image_arg
            captured['scaling_strategy'] = scaling_strategy
            captured['threshold'] = threshold
            captured['window_size'] = window_size
            captured['include_max_pool_dyes'] = include_max_pool_dyes
            return peak_windows, marker_idxs, peak_centers

        monkeypatch.setattr(transformer_module, 'extract_peaks_torch', fake_extract_peaks)

        transformer = CombinedTransformer(
            threshold=30,
            window_size=64,
            include_max_pool_dyes=True,
        )
        inputs, target = transformer(image)
        full_image, out_peak_windows, out_marker_idxs, out_peak_centers, n_peaks = inputs

        assert captured == {
            'image': image,
            'scaling_strategy': image.scaling_strategy,
            'threshold': 30,
            'window_size': 64,
            'include_max_pool_dyes': True,
        }
        assert_close(full_image, torch.tensor(data[:, :, 0], dtype=torch.float32))
        assert_close(out_peak_windows, peak_windows)
        assert_close(out_marker_idxs, marker_idxs)
        assert_close(out_peak_centers, peak_centers)
        assert n_peaks == 2
        assert target.dtype == torch.long
        assert_close(target, torch.tensor(annotation[:, :, 0], dtype=torch.long))

    def test_returns_zero_target_for_images_without_annotation(self, monkeypatch):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        image = _make_fake_image(data=data)

        monkeypatch.setattr(
            transformer_module,
            'extract_peaks_torch',
            lambda *args, **kwargs: (
                torch.ones((1, 1, 4), dtype=torch.float32),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([[0, 2]], dtype=torch.long),
            ),
        )

        inputs, target = CombinedTransformer()(image)
        full_image, _, _, _, n_peaks = inputs

        assert_close(full_image, torch.tensor(data, dtype=torch.float32))
        assert n_peaks == 1
        assert target.dtype == torch.long
        assert_close(target, torch.zeros((3, 4), dtype=torch.long))

    def test_collate_fn_stacks_images_and_concatenates_peak_tensors(self):
        batch = [
            (
                (
                    torch.tensor([[1.0, 2.0]]),
                    torch.tensor([[[10.0, 11.0]], [[12.0, 13.0]]]),
                    torch.tensor([0, 1], dtype=torch.long),
                    torch.tensor([[0, 5], [0, 7]], dtype=torch.long),
                    2,
                ),
                torch.tensor([[1, 0]], dtype=torch.long),
            ),
            (
                (
                    torch.tensor([[3.0, 4.0]]),
                    torch.tensor([[[14.0, 15.0]]]),
                    torch.tensor([2], dtype=torch.long),
                    torch.tensor([[1, 9]], dtype=torch.long),
                    1,
                ),
                torch.tensor([[0, 1]], dtype=torch.long),
            ),
        ]

        inputs, targets = CombinedTransformer.collate_fn(batch)
        full_images, peak_windows, marker_idxs, peak_centers, peak_counts = inputs

        assert full_images.shape == (2, 1, 2)
        assert targets.shape == (2, 1, 2)
        assert peak_windows.shape == (3, 1, 2)
        assert_close(marker_idxs, torch.tensor([0, 1, 2], dtype=torch.long))
        assert_close(
            peak_centers,
            torch.tensor([[0, 5], [0, 7], [1, 9]], dtype=torch.long),
        )
        assert_close(peak_counts, torch.tensor([2, 1], dtype=torch.long))


class TestReconstructionTransformer:
    def test_selects_requested_dyes_and_returns_scaled_input(self, monkeypatch):
        data = np.arange(24, dtype=np.float32).reshape(6, 4, 1)
        image = _make_fake_image(data=data)
        captured = {}

        def fake_scale_rfu(raw, log_scale, max_rfu):
            captured['raw'] = raw.clone()
            captured['log_scale'] = log_scale
            captured['max_rfu'] = max_rfu
            return raw + 0.5

        monkeypatch.setattr(transformer_module, 'scale_rfu_torch', fake_scale_rfu)

        preprocessed, raw = ReconstructionTransformer(
            n_dyes=4,
            log_scale=False,
            max_rfu=1000,
        )(image)

        expected_raw = torch.tensor(data[:4, :, 0], dtype=torch.float32)
        assert_close(raw, expected_raw)
        assert_close(preprocessed, expected_raw + 0.5)
        assert_close(captured['raw'], expected_raw)
        assert captured['log_scale'] is False
        assert captured['max_rfu'] == 1000


class TestPeakClassificationTransformer:
    def test_maps_marker_and_label_with_configured_strategies(self, ppf6c_kit, nfi_rnd_dataset):
        scaling_strategy = ppf6c_kit
        dataset_strategy = nfi_rnd_dataset
        marker_name = scaling_strategy.marker_names[0]
        peak = ExtractedPeak(
            data=np.ones((2, 120), dtype=np.float32),
            dye_index=0,
            peak_center=200,
            window_size=120,
            peak_height=500.0,
            label='allele',
            marker_name=marker_name,
        )

        inputs, target = PeakClassificationTransformer(
            scaling_strategy=scaling_strategy,
            dataset_strategy=dataset_strategy,
            include_marker=True,
        )(peak)
        peak_tensor, marker_tensor = inputs

        assert peak_tensor.shape == (2, 120)
        assert peak_tensor.dtype == torch.float32
        assert marker_tensor.item() == scaling_strategy.marker_to_idx[marker_name]
        assert target.item() == dataset_strategy.get_annotation_classes().index('allele')
        assert target.dtype == torch.long

    def test_uses_negative_marker_index_when_marker_embedding_disabled(self, ppf6c_kit, nfi_rnd_dataset):
        scaling_strategy = ppf6c_kit
        dataset_strategy = nfi_rnd_dataset
        peak = ExtractedPeak(
            data=np.ones((1, 120), dtype=np.float32),
            dye_index=0,
            peak_center=150,
            window_size=120,
            peak_height=250.0,
            label='noise',
            marker_name='ignored',
        )

        inputs, target = PeakClassificationTransformer(
            scaling_strategy=scaling_strategy,
            dataset_strategy=dataset_strategy,
            include_marker=False,
        )(peak)
        _, marker_tensor = inputs

        assert marker_tensor.item() == -1
        assert target.item() == 0
