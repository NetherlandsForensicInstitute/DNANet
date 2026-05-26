"""Integration tests for PeakNet model and module."""

import torch
import pytest
import torchmetrics

from dnanet.models.peaknet import CombinedClassifier, PeakOnlyClassifier
from dnanet.models.autoencoder import Conv1dAutoencoder, PerDyeConv1dAutoencoder
from dnanet.models.peak_classifier import PeakClassificationModel


class TestCombinedClassifierAutoInfer:
    """Test that CombinedClassifier auto-infers output shapes."""

    def test_auto_infer_from_conv1d_autoencoder(self):
        ae = Conv1dAutoencoder(
            in_channels=5,
            hidden_channels=16,
            depth=3,
            input_length=4096,
            kernel_size=7,
            compression=8,
        )
        pc = PeakClassificationModel(
            num_classes=2,
            width=120,
            n_markers=28,
            hidden_channels=[16, 32],
            pooling='flat',
        )
        model = CombinedClassifier(
            autoencoder=ae,
            peak_classifier=pc,
            hidden_dims=[32, 16],
            num_classes=2,
            combiner='mlp',
        )
        assert model is not None
        assert model._autoencoder_out_shape == ae.encoded_shape()

    def test_auto_infer_from_per_dye_autoencoder(self):
        ae = PerDyeConv1dAutoencoder(
            in_channels=5,
            hidden_channels=16,
            depth=3,
            input_length=4096,
            kernel_size=7,
            compression=8,
        )
        pc = PeakClassificationModel(
            num_classes=2,
            width=120,
            hidden_channels=[16, 32],
        )
        model = CombinedClassifier(
            autoencoder=ae,
            peak_classifier=pc,
            hidden_dims=[32],
            combiner='mlp',
        )
        assert model._autoencoder_out_shape == ae.encoded_shape()

    def test_explicit_shapes_still_work(self):
        ae = Conv1dAutoencoder(
            in_channels=5,
            hidden_channels=16,
            depth=3,
            input_length=4096,
            kernel_size=7,
            compression=8,
        )
        pc = PeakClassificationModel(
            num_classes=2,
            width=120,
            hidden_channels=[16, 32],
        )
        model = CombinedClassifier(
            autoencoder=ae,
            peak_classifier=pc,
            autoencoder_out_shape=ae.encoded_shape(),
            peak_classifier_out_features=pc.backbone_out_features(),
            hidden_dims=[32],
            combiner='mlp',
        )
        assert model is not None

    def test_forward_pass(self):
        ae = Conv1dAutoencoder(
            in_channels=5,
            hidden_channels=16,
            depth=3,
            input_length=4096,
            kernel_size=7,
            compression=8,
        )
        pc = PeakClassificationModel(
            num_classes=2,
            width=120,
            hidden_channels=[16, 32],
            embedding_dim=0,
            use_embedding=False,  # disable embedding for simpler test
        )
        model = CombinedClassifier(
            autoencoder=ae,
            peak_classifier=pc,
            hidden_dims=[32],
            combiner='mlp',
            freeze_autoencoder=True,
        )

        N, C, L = 2, 5, 4096
        P = 5  # total peaks across batch

        full_image = torch.randn(N, C, L)
        peak_windows = torch.randn(P, 1, 120)
        marker_idxs = torch.zeros(P, dtype=torch.long)
        peak_centers = torch.zeros(P, 2, dtype=torch.long)
        peak_centers[:, 0] = torch.randint(0, C, (P,))  # dye
        peak_centers[:, 1] = torch.randint(0, L, (P,))  # position
        peak_counts = torch.tensor([3, 2])  # 3 peaks img0, 2 peaks img1

        logits = model(full_image, peak_windows, marker_idxs, peak_centers, peak_counts)
        assert logits.shape == (N, 2, C, L)

    def test_film_combiner(self):
        ae = Conv1dAutoencoder(
            in_channels=5,
            hidden_channels=16,
            depth=3,
            input_length=4096,
            kernel_size=7,
            compression=8,
        )
        pc = PeakClassificationModel(
            num_classes=2,
            width=120,
            hidden_channels=[16, 32],
            embedding_dim=0,
            use_embedding=False,
        )
        model = CombinedClassifier(
            autoencoder=ae,
            peak_classifier=pc,
            hidden_dims=[32],
            combiner='film',
        )
        assert model.combiner_name == 'film'

    def test_attention_combiner(self):
        ae = Conv1dAutoencoder(
            in_channels=5,
            hidden_channels=16,
            depth=3,
            input_length=4096,
            kernel_size=7,
            compression=8,
        )
        pc = PeakClassificationModel(
            num_classes=2,
            width=120,
            hidden_channels=[16, 32],
            embedding_dim=0,
            use_embedding=False,
        )
        model = CombinedClassifier(
            autoencoder=ae,
            peak_classifier=pc,
            hidden_dims=[32],
            combiner='attention',
        )
        assert model.combiner_name == 'attention'


class TestPeakOnlyClassifier:
    def test_forward_pass(self):
        pc = PeakClassificationModel(
            num_classes=2,
            width=120,
            hidden_channels=[16, 32],
            embedding_dim=0,
            use_embedding=False,
        )
        model = PeakOnlyClassifier(peak_classifier=pc, num_classes=2)

        N, C, L = 2, 5, 4096
        P = 5

        full_image = torch.randn(N, C, L)
        peak_windows = torch.randn(P, 1, 120)
        marker_idxs = torch.zeros(P, dtype=torch.long)
        peak_centers = torch.zeros(P, 2, dtype=torch.long)
        peak_centers[:, 0] = torch.randint(0, C, (P,))
        peak_centers[:, 1] = torch.randint(0, L, (P,))
        peak_counts = torch.tensor([3, 2])

        logits = model(full_image, peak_windows, marker_idxs, peak_centers, peak_counts)
        assert logits.shape == (N, 2, C, L)


class TestPeakNetModule:
    def test_training_step(self):
        from dnanet.modules.peaknet import PeakNetModule

        ae = Conv1dAutoencoder(
            in_channels=5,
            hidden_channels=16,
            depth=3,
            input_length=4096,
            kernel_size=7,
            compression=8,
        )
        pc = PeakClassificationModel(
            num_classes=2,
            width=120,
            hidden_channels=[16, 32],
            embedding_dim=0,
            use_embedding=False,
        )
        network = CombinedClassifier(
            autoencoder=ae,
            peak_classifier=pc,
            hidden_dims=[32],
            combiner='mlp',
        )
        metrics_cfg = {
            'accuracy': torchmetrics.classification.MulticlassAccuracy(
                num_classes=2, average='micro'
            ),
            'f1': torchmetrics.classification.MulticlassF1Score(num_classes=2, average='macro'),
        }
        metrics = torchmetrics.MetricCollection(metrics_cfg)

        module = PeakNetModule(
            model=network,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(network.parameters(), lr=1e-3),
            num_classes=2,
            metrics=metrics,
        )

        N, C, L = 2, 5, 4096
        P = 5
        batch = (
            torch.randn(N, C, L),  # full_images
            torch.randn(P, 1, 120),  # peak_windows
            torch.zeros(P, dtype=torch.long),  # marker_idxs
            torch.stack(
                [
                    torch.randint(0, C, (P,)),
                    torch.randint(0, L, (P,)),
                ],
                dim=1,
            ),  # peak_centers
            torch.tensor([3, 2]),  # peak_counts
            torch.randint(0, 2, (N, C, L)),  # targets
        )

        loss = module.training_step(batch, 0)
        assert loss.dim() == 0  # scalar loss
        assert loss.item() > 0


class TestHydraInstantiation:
    """Test that Hydra can instantiate the peaknet model from config."""

    def test_instantiate_combined_classifier(self):
        from omegaconf import OmegaConf
        from hydra.utils import instantiate

        cfg = OmegaConf.create(
            {
                '_target_': 'dnanet.models.peaknet.CombinedClassifier',
                '_recursive_': True,
                'autoencoder': {
                    '_target_': 'dnanet.models.autoencoder.Conv1dAutoencoder',
                    'in_channels': 5,
                    'hidden_channels': 16,
                    'depth': 3,
                    'input_length': 4096,
                    'kernel_size': 7,
                    'compression': 8,
                },
                'peak_classifier': {
                    '_target_': 'dnanet.models.peak_classifier.PeakClassificationModel',
                    'num_classes': 2,
                    'width': 120,
                    'hidden_channels': [16, 32],
                    'embedding_dim': 0,
                    'use_embedding': False,
                },
                'hidden_dims': [32, 16],
                'num_classes': 2,
                'combiner': 'mlp',
            }
        )

        model = instantiate(cfg)
        assert isinstance(model, CombinedClassifier)
        assert model.num_classes == 2
