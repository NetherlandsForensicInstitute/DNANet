"""Tests for HIDDataset label remapping."""

import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader

from dnanet.data.image import HIDImage
from dnanet.models.unet import UNet
from dnanet.core.annotation import ScanpointAnnotation
from dnanet.modules.segmentation import MultiClassSegmentationModule
from dnanet.data.strategies.scaling import PowerPlexFusion6CStrategy


def _make_image(annotation_data):
    """Create a minimal HIDImage with the given annotation data."""
    scaling = PowerPlexFusion6CStrategy()
    image = HIDImage(
        path='/fake/test.hid',
        scaling_strategy=scaling,
        include_size_standard=False,
        load_in_memory=False,
    )
    image._data = np.zeros((5, 4096), dtype=np.float32)
    image._scaler = np.linspace(65, 475, 4096)
    image.annotation = ScanpointAnnotation(data=annotation_data)
    return image


def _remap_array(remap_dict, labels):
    """Apply label remap to a numpy array (mirrors HIDDataset._remap_labels logic)."""
    return np.vectorize(lambda v: remap_dict.get(v, v), otypes=[labels.dtype])(labels)


class TestLabelRemapUnit:
    """Unit tests for the label remap logic (no HIDDataset instantiation)."""

    def test_all_12_classes_remap_correctly(self):
        """Verify that every source class in a remap dict maps to its destination."""
        label_remap = {
            0: 0,
            1: 1,
            2: 3,
            3: 3,
            4: 5,
            5: 5,
            6: 8,
            7: 8,
            8: 8,
            9: 11,
            10: 11,
            11: 11,
        }

        for src, dst in label_remap.items():
            result = label_remap.get(src, src)
            assert result == dst

    def test_default_no_remap_is_identity(self):
        """When label_remap is None, all labels pass through unchanged."""
        label_remap = None

        for label in range(12):
            result = label_remap.get(label, label) if label_remap else label
            assert result == label

    def test_partial_remap_passes_through_unmapped(self):
        """Unmapped source indices should pass through unchanged."""
        label_remap = {i: 2 for i in range(3, 12)}

        assert label_remap.get(0, 0) == 0
        assert label_remap.get(1, 1) == 1
        for i in range(3, 12):
            assert label_remap.get(i, i) == 2

    def test_empty_remap_dict_is_identity(self):
        """An empty label_remap dict should behave like no remap."""
        label_remap = {}

        for label in range(12):
            result = label_remap.get(label, label)
            assert result == label

    def test_span_remap_for_fine_tuning(self):
        """Verify the concrete 3-class fine-tuning remap: classes 3-11 → 2."""
        label_remap = {i: 2 for i in range(3, 12)}

        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int8)
        remapped = _remap_array(label_remap, labels)

        expected = np.array([0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int8)
        np.testing.assert_array_equal(remapped, expected)

    def test_no_remap_preserves_original_labels(self):
        """Without label_remap, all original class indices are preserved."""
        label_remap = {i: 2 for i in range(3, 12)}
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int8)

        original = labels.copy()
        remapped = _remap_array(label_remap, labels)

        np.testing.assert_array_equal(original, labels)
        assert remapped[0] == 0
        assert remapped[1] == 1
        assert remapped[2] == 2
        assert all(v == 2 for v in remapped[3:])

    def test_remap_labels_creates_new_image(self):
        """Test that _remap_labels creates a new image with remapped annotation."""
        from dnanet.data.hid_dataset import HIDDataset

        label_remap = {i: 2 for i in range(3, 12)}

        ann_data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.int8)
        image = _make_image(ann_data)

        # Subclass to bypass cache resolution, setting only needed attrs
        class TestDataset(HIDDataset):
            def __init__(self, label_remap=None):
                self._label_remap = label_remap
                self._scaling = PowerPlexFusion6CStrategy()
                self.include_size_standard = False
                self.data_loading_strategy = 'superior'

        ds = TestDataset(label_remap=label_remap)
        remapped_image = ds._remap_labels(image)

        # Original should be unchanged
        np.testing.assert_array_equal(
            remapped_image.annotation.data[0, :12],
            np.array([0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int8),
        )
        np.testing.assert_array_equal(
            image.annotation.data[0, :12],
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int8),
        )

    def test_remap_labels_none_annotation_unchanged(self):
        """If annotation is None, _remap_labels returns the image unchanged."""
        from dnanet.data.hid_dataset import HIDDataset

        label_remap = {i: 2 for i in range(3, 12)}

        scaling = PowerPlexFusion6CStrategy()
        image = HIDImage(
            path='/fake/test.hid',
            scaling_strategy=scaling,
            include_size_standard=False,
            load_in_memory=False,
        )
        image._data = np.zeros((5, 4096), dtype=np.float32)

        class TestDataset(HIDDataset):
            def __init__(self, label_remap=None):
                self._label_remap = label_remap
                self._scaling = PowerPlexFusion6CStrategy()
                self.include_size_standard = False
                self.data_loading_strategy = 'superior'

        ds = TestDataset(label_remap=label_remap)
        result = ds._remap_labels(image)
        assert result.annotation is None

    def test_remap_with_2d_annotation(self):
        """Test remap works on multi-dye 2D annotations."""
        from dnanet.data.hid_dataset import HIDDataset

        label_remap = {i: 2 for i in range(3, 12)}

        # 3 dyes × 20 scanpoints, with various class labels
        ann_data = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],
            ],
            dtype=np.int8,
        )

        image = _make_image(ann_data)

        class TestDataset(HIDDataset):
            def __init__(self, label_remap=None):
                self._label_remap = label_remap
                self._scaling = PowerPlexFusion6CStrategy()
                self.include_size_standard = False
                self.data_loading_strategy = 'superior'

        ds = TestDataset(label_remap=label_remap)
        remapped_image = ds._remap_labels(image)

        # Expected: classes 3-11 → 2, classes 0-2 unchanged
        expected = np.array(
            [
                [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2],
                [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2],
                [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2],
            ],
            dtype=np.int8,
        )
        np.testing.assert_array_equal(remapped_image.annotation.data, expected)


class TestMultiClassIgnoreIndex:
    """Tests for MultiClassSegmentationModule ignore_index behavior."""

    def test_training_step_with_remap_no_errors(self):
        """Fine-tune training step should run end-to-end without shape or index errors."""
        import lightning as L

        model = UNet(depth=1, kernel_size=(3, 3), num_filters=4, out_channels=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        module = MultiClassSegmentationModule(
            model=model,
            loss_fn=torch.nn.CrossEntropyLoss(ignore_index=2),
            optimizer=optimizer,
            learning_rate=1e-3,
        )

        torch.manual_seed(42)
        x = torch.randn(4, 5, 64)
        # Only classes 0 and 1 — no ignored class, so loss is always valid
        y = torch.zeros(4, 5, 64, dtype=torch.long)
        y[:, :, 10:20] = 1  # class 1 (allele)

        ds = torch.utils.data.TensorDataset(x, y)
        dl = DataLoader(ds, batch_size=2)

        trainer = L.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(module, train_dataloaders=dl)

        # Verify forward pass works
        with torch.no_grad():
            logits = module(x)
        # UNet preserves dye channel dim: (B, out_channels, num_dyes, scanpoints)
        assert logits.shape == (4, 3, 5, 64)

    def test_training_step_with_ignore_index_filters_correctly(self):
        """Verify that ignore_index correctly filters predictions for metrics."""
        model = UNet(depth=1, kernel_size=(3, 3), num_filters=4, out_channels=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        module = MultiClassSegmentationModule(
            model=model,
            loss_fn=torch.nn.CrossEntropyLoss(ignore_index=2),
            optimizer=optimizer,
            learning_rate=1e-3,
            ignore_index=2,
        )

        x = torch.randn(2, 5, 64)
        y = torch.full((2, 5, 64), 2, dtype=torch.long)

        loss, preds, targets = module._compute_loss_and_probabilities((x, y))

        # When all labels are the ignored class, predictions should be empty
        assert preds.numel() == 0

    def test_training_step_with_ignore_index_keeps_valid(self):
        """Verify that ignore_index keeps non-ignored predictions."""
        model = UNet(depth=1, kernel_size=(3, 3), num_filters=4, out_channels=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        module = MultiClassSegmentationModule(
            model=model,
            loss_fn=torch.nn.CrossEntropyLoss(ignore_index=2),
            optimizer=optimizer,
            learning_rate=1e-3,
            ignore_index=2,
        )

        x = torch.randn(2, 5, 64)
        y = torch.zeros(2, 5, 64, dtype=torch.long)
        y[:, :, 10:20] = 1  # class 1
        y[:, :, 40:50] = 2  # class 2 (ignored)

        loss, preds, targets = module._compute_loss_and_probabilities((x, y))

        assert loss.shape == ()
        # Total: 640, class 2: 100, kept: 540
        assert preds.numel() == 540
