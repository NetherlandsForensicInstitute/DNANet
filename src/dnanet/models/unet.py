"""U-Net architecture for EPG segmentation.

This module implements a U-Net tailored for 2D electropherogram (EPG) data
where the spatial dimensions are ``(num_dyes, signal_length)`` — height is
very small (5 dyes) while width is large (4096 scan points).

Design pattern: **Composite**
    The U-Net is composed from reusable building blocks (DoubleConv,
    EncoderBlock, DecoderBlock) that form a symmetric encoder-decoder
    architecture with skip connections.

Key design decisions:
    - Pooling and upsampling operate **only along the width axis** (signal
      length) via ``MaxPool2d(1, 2)`` and ``ConvTranspose2d(1, 2)``.
      The height (dye channels) is preserved throughout.
    - Filter count doubles at each encoder level: 32 → 64 → 128 → 256 → 512
      (for depth=4, num_filters=32).
    - Single-channel inputs are provided as ``(B, H, W)`` and a singleton
      channel dimension is added internally for the 2D convolutions.
    - The default output is a single-channel logit map with the same spatial
      dimensions as the input, returned as ``(B, H, W)``.

Ported from the original DNANet U-Net with improved type safety and
Hydra-compatible configuration (no ``device`` parameter — Lightning
handles device placement).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks.

    Padding is chosen to preserve spatial dimensions exactly.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: ``(height, width)`` of the convolution kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
    ) -> None:
        super().__init__()
        kh, kw = kernel_size
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=(kh, kw),
                padding=(kh // 2, kw // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(kh, kw),
                padding=(kh // 2, kw // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Encoder: DoubleConv → MaxPool along width.

    Returns both the pre-pooled features (for the skip connection) and
    the pooled output for the next level.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``(skip_features, pooled_output)``."""
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder: ConvTranspose2d upsample → concat skip → DoubleConv.

    Upsampling doubles width while preserving height, then the skip
    connection from the corresponding encoder level is concatenated
    along the channel dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=(1, 2),
            stride=(1, 2),
        )
        self.conv = DoubleConv(out_channels * 2, out_channels, kernel_size)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for DNA profile segmentation.

    Args:
        depth: Number of encoder/decoder levels.
        kernel_size: ``(height, width)`` convolution kernel size.
        num_filters: Number of filters in the first encoder level.
            Doubles at each subsequent level.
        in_channels: Number of input channels for legacy 4D inputs. Must be 1
            when using the channel-less ``(B, H, W)`` input convention.
        out_channels: Number of output channels (default 1 for binary
            segmentation). A single output channel is returned as ``(B, H, W)``
            when the input was provided without an explicit channel dimension.
    """

    def __init__(
        self,
        depth: int = 4,
        kernel_size: tuple[int, int] | list[int] = (3, 5),
        num_filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder path
        self.encoders = nn.ModuleList()
        ch_in = in_channels
        for i in range(depth):
            ch_out = num_filters * 2**i
            self.encoders.append(EncoderBlock(ch_in, ch_out, kernel_size))
            ch_in = ch_out

        # Bottleneck
        bottleneck_channels = num_filters * 2**depth
        self.bottleneck = DoubleConv(ch_in, bottleneck_channels, kernel_size)

        # Decoder path
        self.decoders = nn.ModuleList()
        ch_in = bottleneck_channels
        for i in range(depth):
            ch_out = num_filters * 2 ** (depth - 1 - i)
            self.decoders.append(DecoderBlock(ch_in, ch_out, kernel_size))
            ch_in = ch_out

        # Final 1×1 conv to produce logits
        self.head = nn.Conv2d(ch_in, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, H, W)`` for the standard
                single-channel case, or legacy ``(B, C_in, H, W)``.

        Returns:
            Logits of shape ``(B, H, W)`` for single-channel outputs when the
            input was channel-less, otherwise ``(B, C_out, H, W)``.
        """
        added_channel_dim = False
        if x.ndim == 3:
            if self.in_channels != 1:
                raise ValueError(
                    "Channel-less inputs require in_channels=1; "
                    f"got in_channels={self.in_channels}."
                )
            x = x.unsqueeze(1)
            added_channel_dim = True
        elif x.ndim != 4:
            raise ValueError(
                "UNet expects input with shape (B, H, W) or (B, C, H, W); "
                f"got shape {tuple(x.shape)}."
            )

        # Encode — collect skip connections
        skips: list[Tensor] = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode — consume skip connections in reverse
        for decoder in self.decoders:
            x = decoder(x, skips.pop())

        logits = self.head(x)
        if added_channel_dim and self.out_channels == 1:
            return logits.squeeze(1)
        return logits
