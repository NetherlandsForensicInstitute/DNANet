import abc
from typing import Tuple

import torch
from torch import nn


class AbstractAutoEncoder(nn.Module, abc.ABC):
    """
    Abstract base class for 1D autoencoders.

    Defines the common interface: an `encoder` that maps an input sequence to a latent
    representation and a `decoder` that reconstructs the input from the latent space.
    Subclasses must implement `encoder`, `decoder`, and `encoded_shape`.
    """

    def __init__(self, device: str, in_channels):
        super().__init__()
        self.device = device
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder.
        :param x: Input tensor.
        :return: Output tensor after encoding and decoding.
        """
        return self.decoder(self.encoder(x))


    @abc.abstractmethod
    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder part of the autoencoder.
        :param x: Input tensor.
        :return: Encoded tensor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def encoded_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the encoded representation.
        :return: Shape of the encoded tensor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decoder part of the autoencoder.
        :param x: Encoded tensor.
        :return: Decoded tensor.
        """
        raise NotImplementedError


class Conv1dAutoencoder(AbstractAutoEncoder):
    """
    Multichannel 1D convolutional autoencoder (single shared network over all input channels).

    Architecture:
    - Encoder: `depth` strided Conv1d blocks (stride=2) to downsample the fragment length (basepair) axis,
      followed by a 1x1 Conv1d to set the latent channel count.
    - Decoder: 1x1 Conv1d to expand channels, then repeated (Upsample x2 + Conv1d) blocks
      to restore resolution, ending with a final Conv1d to `in_channels` (optional Sigmoid).

    Notes:
    - Treats channels jointly: convolutions operate across all `in_channels` together.
    - Compression is achieved by temporal downsampling and choosing latent channels so the
      latent tensor matches the requested total latent size.
    """


    def __init__(
            self,
            device: str,
            in_channels: int,
            hidden_channels: int,
            depth: int,
            input_length: int,
            kernel_size: int,
            use_sigmoid: bool = True,
            use_batchnorm: bool = False,
            compression: int = 8,
    ):

        super().__init__(device, in_channels)

        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length via symmetric padding.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.input_length = input_length
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        padding_mode = "reflect"

        # calculate the size of the latent representation
        latent_size = input_length // compression
        latent_length = input_length // (2 ** depth)
        channels_latent = latent_size // latent_length

        if channels_latent < 1:
            raise ValueError("input_length too small for the requested depth and compression.")
        Lz = input_length
        for _ in range(depth):
            Lz = (Lz + 1) // 2   # integer ceil(L/2)
        if latent_size % Lz != 0:
            raise ValueError(f"Cannot get exact {compression}x with depth={depth}: "
                             f"latent_size={latent_size} not divisible by Lz={Lz}")

        self.encoded_channels = channels_latent
        self.encoded_len = latent_length


        # Encoder
        encoder_layers = []
        in_ch = in_channels
        channels = []
        for level in range(depth):
            out_ch = hidden_channels
            encoder_layers.append(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    padding_mode=padding_mode,
                )
            )
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(out_ch))
            encoder_layers.append(nn.ReLU())
            channels.append(out_ch)
            in_ch = out_ch

        encoder_layers.append(
            nn.Conv1d(
                in_ch,
                channels_latent,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.encoder_module = nn.Sequential(*encoder_layers).to(device)


        # Decoder
        decoder_layers = []
        rev_channels = channels[::-1]

        decoder_layers.append(
            nn.Conv1d(
                channels_latent,
                rev_channels[0],
                kernel_size=1,
                stride=1,
            )
        )

        for i in range(len(rev_channels) - 1):
            in_ch = rev_channels[i]
            out_ch = rev_channels[i + 1]
            decoder_layers.append(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
            )
            decoder_layers.append(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    padding_mode=padding_mode,
                )
            )
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(out_ch))
            decoder_layers.append(nn.ReLU())



        decoder_layers.append(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        )
        decoder_layers.append(
            nn.Conv1d(
                rev_channels[-1],
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            )
        )
        if use_sigmoid:
            decoder_layers.append(nn.Sigmoid())
        self.decoder_module = nn.Sequential(*decoder_layers).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_module(x)
        y = self.decoder_module(z)
        return y

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_module(x)

    def encoded_shape(self) -> Tuple[int, ...]:
        shape = (self.encoded_channels, self.encoded_len)
        return shape

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_module(x)




class PerDyeConv1dAutoencoder(AbstractAutoEncoder):
    """
    Per-channel (per-dye) 1D convolutional autoencoder with independent weights.

    Architecture:
    - Creates `in_channels` separate `Conv1dAutoencoder` subnets, each configured with
      `in_channels=1`.
    - Encoder: runs each channel through its own sub-encoder and concatenates latents
      along the channel dimension.
    - Decoder: splits the latent back into per-channel chunks, decodes with the matching
      sub-decoder, and concatenates reconstructions.

    Difference vs `Conv1dAutoencoder`:
    - No cross-channel mixing: each channel is encoded/decoded independently.
    - Parameters scale with `in_channels` because each channel has its own network.
    """

    def __init__(
            self,
            device: str,
            in_channels: int = 5,
            hidden_channels: int = 16,
            depth: int = 2,
            input_length: int = 4096,
            kernel_size: int = 9,
            use_sigmoid: bool = True,
            use_batchnorm: bool = False,
            compression: int = 8,
    ):
        super().__init__(device, in_channels)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.input_length = input_length
        self.kernel_size = kernel_size

        # One independent sub-AE per channel (each with in_channels=1)
        self.per_channel_nets = nn.ModuleList(
            [
                Conv1dAutoencoder(
                    device=device,
                    in_channels=1,
                    hidden_channels=hidden_channels,
                    depth=depth,
                    input_length=input_length,
                    kernel_size=kernel_size,
                    use_sigmoid=use_sigmoid,
                    use_batchnorm=use_batchnorm,
                    compression=compression
                )
                for _ in range(in_channels)
            ]
        )

        encoded_len = self.per_channel_nets[0].encoded_len
        encoded_ch = self.per_channel_nets[0].encoded_channels
        self._encoded_shape = (in_channels * encoded_ch, encoded_len)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(self.encoder(x))
        return decoded

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (N, C, L, 1) or (N, C, L)
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        batch_size, channels, length = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")

        encoded_list = []
        for ch_idx in range(self.in_channels):
            x_ch = x[:, ch_idx : ch_idx + 1, :]  # (N, 1, L)
            z_ch = self.per_channel_nets[ch_idx].encoder(x_ch)
            encoded_list.append(z_ch)

        z = torch.cat(encoded_list, dim=1)  # (N, in_channels * encoded_ch, encoded_len)
        return z

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, total_ch, enc_len = z.shape
        encoded_ch = self.per_channel_nets[0].encoded_channels

        if total_ch != self.in_channels * encoded_ch:
            raise ValueError(
                f"Encoded channels mismatch: expected {self.in_channels * encoded_ch}, got {total_ch}."
            )

        decoded_list = []
        for ch_idx in range(self.in_channels):
            start = ch_idx * encoded_ch
            end = (ch_idx + 1) * encoded_ch
            z_ch = z[:, start:end, :]  # (N, encoded_ch, enc_len)
            y_ch = self.per_channel_nets[ch_idx].decoder(z_ch)  # (N, 1, L)
            decoded_list.append(y_ch)

        y = torch.cat(decoded_list, dim=1)  # (N, C, L)
        return y

    def encoded_shape(self) -> Tuple[int, ...]:
        return self._encoded_shape



class SharedWeightPerDyeConv1dAutoencoder(PerDyeConv1dAutoencoder):
    """
    Per-channel 1D convolutional autoencoder with shared weights across channels.

    Architecture:
    - Same encode/decode flow as `PerDyeConv1dAutoencoder` (split per channel, concatenate
      latents, then split/concatenate on decode).
    - Uses a single `Conv1dAutoencoder` instance reused for all channels (weight tying).

    Difference vs `PerDyeConv1dAutoencoder`:
    - Preserves channel independence in computation flow, but all channels share the same
      encoder/decoder parameters, reducing model size and enforcing the same transform
      per channel.
    """
    def __init__(
            self,
            device: str,
            in_channels: int = 5,
            hidden_channels: int = 16,
            depth: int = 2,
            input_length: int = 4096,
            kernel_size: int = 9,
            use_sigmoid: bool = True,
            use_batchnorm: bool = False,
            compression: int = 8,
    ):
        super().__init__(
            device=device,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            depth=depth,
            input_length=input_length,
            kernel_size=kernel_size,
            use_sigmoid=use_sigmoid,
            use_batchnorm=use_batchnorm,
            compression=compression
        )

        # Override the per_channel_nets with shared weights
        shared_net = Conv1dAutoencoder(
                    device=device,
                    in_channels=1,
                    hidden_channels=hidden_channels,
                    depth=depth,
                    input_length=input_length,
                    kernel_size=kernel_size,
                    use_sigmoid=use_sigmoid,
                    use_batchnorm=use_batchnorm,
                    compression=compression
                )

        self.per_channel_nets = nn.ModuleList([shared_net for _ in range(in_channels)])
