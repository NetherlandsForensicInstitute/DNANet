"""Tests for U-Net architecture."""

import pytest
import torch

from dnanet.models.unet import DecoderBlock, DoubleConv, EncoderBlock, UNet


class TestDoubleConv:
    """Tests for the DoubleConv building block."""

    def test_output_shape(self):
        """Output should preserve spatial dimensions."""
        block = DoubleConv(1, 32, kernel_size=(3, 5))
        x = torch.randn(2, 1, 5, 256)
        out = block(x)
        assert out.shape == (2, 32, 5, 256)

    def test_channel_change(self):
        """Should change channel dimension from in to out."""
        block = DoubleConv(16, 64, kernel_size=(3, 5))
        x = torch.randn(1, 16, 5, 128)
        out = block(x)
        assert out.shape[1] == 64


class TestEncoderBlock:
    """Tests for the encoder block."""

    def test_output_shapes(self):
        """Should return skip (same spatial) and pooled (halved width)."""
        block = EncoderBlock(1, 32, kernel_size=(3, 5))
        x = torch.randn(2, 1, 5, 256)
        skip, pooled = block(x)
        assert skip.shape == (2, 32, 5, 256)
        assert pooled.shape == (2, 32, 5, 128)

    def test_height_preserved(self):
        """Pooling should not affect the height (dye) dimension."""
        block = EncoderBlock(1, 16, kernel_size=(3, 5))
        x = torch.randn(1, 1, 5, 512)
        skip, pooled = block(x)
        assert skip.shape[2] == 5
        assert pooled.shape[2] == 5


class TestDecoderBlock:
    """Tests for the decoder block."""

    def test_output_shape(self):
        """Should upsample width by 2x after concatenating skip."""
        block = DecoderBlock(64, 32, kernel_size=(3, 5))
        x = torch.randn(2, 64, 5, 64)
        skip = torch.randn(2, 32, 5, 128)
        out = block(x, skip)
        assert out.shape == (2, 32, 5, 128)


class TestUNet:
    """Tests for the full U-Net."""

    @pytest.fixture
    def unet(self) -> UNet:
        """A small U-Net for testing."""
        return UNet(depth=2, kernel_size=(3, 5), num_filters=8)

    def test_output_shape_matches_input(self, unet):
        """Output spatial dims should match input."""
        x = torch.randn(2, 1, 5, 256)
        out = unet(x)
        assert out.shape == (2, 1, 5, 256)

    def test_default_config(self):
        """Default UNet should work with typical EPG dimensions."""
        unet = UNet()  # depth=4, kernel_size=(3,5), num_filters=32
        x = torch.randn(1, 1, 5, 4096)
        out = unet(x)
        assert out.shape == (1, 1, 5, 4096)

    def test_custom_channels(self):
        """Should support custom in/out channels."""
        unet = UNet(depth=2, kernel_size=(3, 3), num_filters=8,
                     in_channels=3, out_channels=5)
        x = torch.randn(1, 3, 5, 128)
        out = unet(x)
        assert out.shape == (1, 5, 5, 128)

    def test_kernel_size_as_list(self):
        """Hydra passes kernel_size as a list — should work."""
        unet = UNet(depth=2, kernel_size=[3, 5], num_filters=8)
        x = torch.randn(1, 1, 5, 256)
        out = unet(x)
        assert out.shape == (1, 1, 5, 256)

    def test_single_depth(self):
        """Depth=1 should still produce valid output."""
        unet = UNet(depth=1, kernel_size=(3, 3), num_filters=8)
        x = torch.randn(1, 1, 5, 64)
        out = unet(x)
        assert out.shape == (1, 1, 5, 64)

    def test_gradients_flow(self, unet):
        """Gradients should flow through the entire network."""
        x = torch.randn(1, 1, 5, 256, requires_grad=True)
        out = unet(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_parameter_count_scales_with_depth(self):
        """Deeper networks should have more parameters."""
        small = UNet(depth=2, kernel_size=(3, 3), num_filters=8)
        large = UNet(depth=4, kernel_size=(3, 3), num_filters=8)
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        assert large_params > small_params

    def test_batch_independence(self, unet):
        """Each sample in a batch should be processed independently."""
        x1 = torch.randn(1, 1, 5, 256)
        x2 = torch.randn(1, 1, 5, 256)
        unet.eval()
        with torch.no_grad():
            out1 = unet(x1)
            out2 = unet(x2)
            out_batch = unet(torch.cat([x1, x2], dim=0))
        torch.testing.assert_close(out_batch[0:1], out1)
        torch.testing.assert_close(out_batch[1:2], out2)

    def test_hydra_config_instantiation(self):
        """Should be instantiable with Hydra-style config."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "_target_": "dnanet.models.unet.UNet",
            "depth": 2,
            "kernel_size": [3, 5],
            "num_filters": 8,
        })
        unet = instantiate(cfg)
        assert isinstance(unet, UNet)
        x = torch.randn(1, 1, 5, 128)
        out = unet(x)
        assert out.shape == x.shape
