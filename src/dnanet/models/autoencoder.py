"""1D convolutional autoencoder architectures for EPG reconstruction.

This module provides autoencoders that compress and reconstruct
electropherogram signals. Four architecture variants are available:

- :class:`Conv1dAutoencoder` — Shared multi-channel CNN autoencoder.
- :class:`PerDyeConv1dAutoencoder` — Independent sub-encoder per dye channel.
- :class:`SharedWeightPerDyeConv1dAutoencoder` — Per-dye processing with
  weight tying (single shared sub-encoder reused across channels).
- :class:`FourierAutoencoder` — Non-trainable baseline using truncated rFFT.

Design pattern: **Template Method**
    :class:`AbstractAutoencoder` defines the ``forward = decode(encode(x))``
    skeleton. Subclasses fill in the encoder/decoder implementations.

Design pattern: **Composite**
    :class:`PerDyeConv1dAutoencoder` composes multiple
    :class:`Conv1dAutoencoder` instances (one per dye channel).
"""

from __future__ import annotations

import abc

import torch
from torch import Tensor, nn


class AbstractAutoencoder(nn.Module, abc.ABC):
    """Abstract base class for 1D autoencoders.

    Subclasses must implement :meth:`encode`, :meth:`decode`, and
    :meth:`encoded_shape`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    @abc.abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation."""

    @abc.abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation back to input space."""

    @abc.abstractmethod
    def encoded_shape(self) -> tuple[int, ...]:
        """Shape of the latent representation (excluding batch dim)."""


class Conv1dAutoencoder(AbstractAutoencoder):
    """Multi-channel 1D convolutional autoencoder.

    Treats all input channels jointly — convolutions operate across all
    ``in_channels`` together.

    Architecture:
        - **Encoder**: ``depth`` strided Conv1d blocks (stride=2) that
          downsample the time axis, followed by a 1×1 Conv1d to set the
          latent channel count.
        - **Decoder**: 1×1 Conv1d to expand channels, then repeated
          ``(Upsample ×2 + Conv1d)`` blocks, ending with a final Conv1d
          to ``in_channels`` (optional Sigmoid).

    Args:
        in_channels: Number of input channels (dye channels).
        hidden_channels: Number of channels in intermediate conv layers.
        depth: Number of encoder/decoder levels (each halves/doubles length).
        input_length: Expected signal length (e.g. 4096).
        kernel_size: Convolution kernel size (must be odd).
        compression: Target compression ratio (input_length / latent_size).
        use_sigmoid: Whether to apply sigmoid to decoder output.
        use_batchnorm: Whether to add BatchNorm1d after each conv layer.
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 32,
        depth: int = 5,
        input_length: int = 4096,
        kernel_size: int = 7,
        compression: int = 8,
        use_sigmoid: bool = True,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding.")

        self.in_channels = in_channels
        self.depth = depth
        self.input_length = input_length
        padding = kernel_size // 2

        # Calculate latent dimensions
        latent_size = input_length // compression
        latent_length = input_length // (2**depth)
        channels_latent = latent_size // latent_length

        if channels_latent < 1:
            raise ValueError(
                "input_length too small for requested depth and compression."
            )

        # Verify exact compression is achievable
        lz = input_length
        for _ in range(depth):
            lz = (lz + 1) // 2
        if latent_size % lz != 0:
            raise ValueError(
                f"Cannot achieve exact {compression}x compression with "
                f"depth={depth}: latent_size={latent_size} not divisible by Lz={lz}"
            )

        self._encoded_channels = channels_latent
        self._encoded_length = latent_length

        # Build encoder
        encoder_layers: list[nn.Module] = []
        ch_in = in_channels
        channels_per_level: list[int] = []
        for _ in range(depth):
            encoder_layers.append(
                nn.Conv1d(
                    ch_in, hidden_channels,
                    kernel_size=kernel_size, stride=2,
                    padding=padding, padding_mode="reflect",
                )
            )
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(hidden_channels))
            encoder_layers.append(nn.ReLU())
            channels_per_level.append(hidden_channels)
            ch_in = hidden_channels

        # Bottleneck projection to latent channels
        encoder_layers.append(
            nn.Conv1d(ch_in, channels_latent, kernel_size=1)
        )
        self.encoder_net = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers: list[nn.Module] = []
        rev = channels_per_level[::-1]

        decoder_layers.append(
            nn.Conv1d(channels_latent, rev[0], kernel_size=1)
        )

        for i in range(len(rev) - 1):
            decoder_layers.append(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
            )
            decoder_layers.append(
                nn.Conv1d(
                    rev[i], rev[i + 1],
                    kernel_size=kernel_size, padding=padding,
                    padding_mode="reflect",
                )
            )
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(rev[i + 1]))
            decoder_layers.append(nn.ReLU())

        # Final upsample + project back to in_channels
        decoder_layers.append(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        )
        decoder_layers.append(
            nn.Conv1d(
                rev[-1], in_channels,
                kernel_size=kernel_size, padding=padding,
                padding_mode="reflect",
            )
        )
        if use_sigmoid:
            decoder_layers.append(nn.Sigmoid())

        self.decoder_net = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder_net(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder_net(z)

    def encoded_shape(self) -> tuple[int, ...]:
        return (self._encoded_channels, self._encoded_length)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


class PerDyeConv1dAutoencoder(AbstractAutoencoder):
    """Per-channel 1D autoencoder with independent weights per dye.

    Creates ``in_channels`` separate :class:`Conv1dAutoencoder` instances,
    each with ``in_channels=1``. No cross-channel mixing occurs — each dye
    is encoded/decoded independently.

    Args:
        in_channels: Number of dye channels.
        hidden_channels: Channels per sub-encoder.
        depth: Encoder/decoder depth.
        input_length: Signal length.
        kernel_size: Convolution kernel size.
        compression: Compression ratio.
        use_sigmoid: Apply sigmoid to output.
        use_batchnorm: Use BatchNorm.
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 16,
        depth: int = 2,
        input_length: int = 4096,
        kernel_size: int = 9,
        compression: int = 8,
        use_sigmoid: bool = True,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.per_channel_nets = nn.ModuleList([
            Conv1dAutoencoder(
                in_channels=1,
                hidden_channels=hidden_channels,
                depth=depth,
                input_length=input_length,
                kernel_size=kernel_size,
                compression=compression,
                use_sigmoid=use_sigmoid,
                use_batchnorm=use_batchnorm,
            )
            for _ in range(in_channels)
        ])

        sub = self.per_channel_nets[0]
        enc_ch = sub._encoded_channels
        enc_len = sub._encoded_length
        self._encoded_shape = (in_channels * enc_ch, enc_len)

    def encode(self, x: Tensor) -> Tensor:
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        encoded = [
            self.per_channel_nets[ch].encode(x[:, ch : ch + 1, :])
            for ch in range(self.in_channels)
        ]
        return torch.cat(encoded, dim=1)

    def decode(self, z: Tensor) -> Tensor:
        enc_ch = self.per_channel_nets[0]._encoded_channels
        decoded = [
            self.per_channel_nets[ch].decode(
                z[:, ch * enc_ch : (ch + 1) * enc_ch, :]
            )
            for ch in range(self.in_channels)
        ]
        return torch.cat(decoded, dim=1)

    def encoded_shape(self) -> tuple[int, ...]:
        return self._encoded_shape


class SharedWeightPerDyeConv1dAutoencoder(PerDyeConv1dAutoencoder):
    """Per-channel autoencoder with shared weights across dye channels.

    Same encode/decode flow as :class:`PerDyeConv1dAutoencoder`, but all
    channels share a single :class:`Conv1dAutoencoder` — reducing model
    size and enforcing the same transform per channel.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Replace the independent per-channel nets with a single shared one
        shared_net = Conv1dAutoencoder(
            in_channels=1,
            hidden_channels=kwargs.get("hidden_channels", 16),
            depth=kwargs.get("depth", 2),
            input_length=kwargs.get("input_length", 4096),
            kernel_size=kwargs.get("kernel_size", 9),
            compression=kwargs.get("compression", 8),
            use_sigmoid=kwargs.get("use_sigmoid", True),
            use_batchnorm=kwargs.get("use_batchnorm", False),
        )
        self.per_channel_nets = nn.ModuleList(
            [shared_net] * kwargs.get("in_channels", 5)
        )


class FourierAutoencoder(AbstractAutoencoder):
    """Non-trainable autoencoder using truncated real FFT.

    Encodes by keeping only the lowest ``latent_coeffs`` frequency
    components of the rFFT. Decodes by zero-padding the spectrum and
    applying the inverse rFFT.

    This is a **baseline model** — it has no learnable parameters.

    Args:
        in_channels: Number of input channels.
        signal_length: Length of the 1D signal.
        latent_coeffs: Number of low-frequency rFFT coefficients to keep.
        norm: FFT normalization mode.
    """

    def __init__(
        self,
        in_channels: int = 5,
        signal_length: int = 4096,
        latent_coeffs: int = 256,
        norm: str = "backward",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.full_freq_len = signal_length // 2 + 1
        self.latent_coeffs = latent_coeffs
        self.norm = norm

        if latent_coeffs > self.full_freq_len:
            raise ValueError(
                f"latent_coeffs={latent_coeffs} exceeds max rFFT length "
                f"({self.full_freq_len})"
            )

    def encode(self, x: Tensor) -> Tensor:
        """Encode via truncated rFFT → (B, C, K, 2) real/imag."""
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        spectrum = torch.fft.rfft(x, n=self.signal_length, dim=-1, norm=self.norm)
        truncated = spectrum[..., : self.latent_coeffs]
        return torch.stack((truncated.real, truncated.imag), dim=-1)

    def decode(self, z: Tensor) -> Tensor:
        """Decode via zero-padded inverse rFFT."""
        truncated = torch.complex(z[..., 0], z[..., 1])
        b, c, k = truncated.shape

        full = torch.zeros(
            (b, c, self.full_freq_len),
            dtype=truncated.dtype, device=truncated.device,
        )
        full[..., :k] = truncated

        return torch.fft.irfft(full, n=self.signal_length, dim=-1, norm=self.norm)

    def encoded_shape(self) -> tuple[int, ...]:
        return (self.in_channels, self.latent_coeffs, 2)
