from typing import Tuple, Optional

import torch

from DNAnet.models.reconstruction.autoencoder_architectures import AbstractAutoEncoder


class FourierAutoEncoder(AbstractAutoEncoder):
    """
    Autoencoder that encodes a 1D signal via a truncated real FFT (rFFT)
    and decodes by zero-padding the spectrum and applying the inverse rFFT.

    Args:
        device: 'cpu' or 'cuda'
        in_channels: number of input channels
        signal_length: length of the 1D signal (last dimension of x)
        latent_coeffs: number of low-frequency rFFT coefficients to keep per channel
        norm: normalization mode passed to torch.fft.rfft/irfft ('backward', 'forward', or 'ortho')
    """
    def __init__(
            self,
            device: str,
            in_channels: int,
            signal_length: int,
            latent_coeffs: int,
            norm: str = "backward",
    ):
        super().__init__(device=device, in_channels=in_channels)

        self.signal_length = signal_length
        # rfft output length for real input of size N is N//2 + 1
        self.full_freq_len = signal_length // 2 + 1
        if latent_coeffs > self.full_freq_len:
            raise ValueError(
                f"latent_coeffs={latent_coeffs} exceeds maximum rFFT length={self.full_freq_len}"
            )
        self.latent_coeffs = latent_coeffs
        self.norm = norm

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, signal_length), real-valued
        returns: (batch, in_channels, latent_coeffs, 2) with real/imag parts
        """
        # Ensure correct shape
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.shape[-1] != self.signal_length:
            raise ValueError(
                f"Expected last dim={self.signal_length}, got {x.shape[-1]}"
            )

        # rFFT over last dimension -> (B, C, full_freq_len) complex
        X = torch.fft.rfft(x, n=self.signal_length, dim=-1, norm=self.norm)

        # Truncate to lowest latent_coeffs frequencies
        X_trunc = X[..., : self.latent_coeffs]  # (B, C, K)

        # Store real and imag parts explicitly as last dimension of size 2, so latent is real tensor
        z = torch.stack((X_trunc.real, X_trunc.imag), dim=-1)  # (B, C, K, 2)
        return z

    def encoded_shape(self) -> Tuple[int, ...]:
        """
        Shape of encoded representation per sample (without batch dimension).
        """
        # (in_channels, latent_coeffs, 2) -> real+imag
        return (self.in_channels, self.latent_coeffs, 2)

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, in_channels, latent_coeffs, 2)
        returns: (batch, in_channels, signal_length), real-valued reconstruction
        """
        if z.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected in_channels={self.in_channels}, got {z.shape[1]}"
            )
        if z.shape[2] != self.latent_coeffs or z.shape[3] != 2:
            raise ValueError(
                f"Expected latent shape (*, {self.in_channels}, {self.latent_coeffs}, 2), got {tuple(z.shape)}"
            )

        # Recombine real and imag parts -> complex tensor of shape (B, C, K)
        X_trunc = torch.complex(z[..., 0], z[..., 1])

        # Zero-pad in frequency domain back to full_freq_len along last dim
        B, C, K = X_trunc.shape
        if K != self.latent_coeffs:
            raise ValueError(f"Unexpected latent_coeffs in decoder: {K}")
        X_full = torch.zeros(
            (B, C, self.full_freq_len),
            dtype=X_trunc.dtype,
            device=X_trunc.device,
        )
        X_full[..., :K] = X_trunc

        # Inverse rFFT over last dimension -> real signal (B, C, signal_length)
        x_rec = torch.fft.irfft(X_full, n=self.signal_length, dim=-1, norm=self.norm)
        return x_rec

