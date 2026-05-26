"""Peak-level classifier for extracted electropherogram windows.

This model looks at a small window around each detected peak and predicts
whether that peak is allelic or artefactual. It is used in two ways:

- as a standalone classifier for faster iteration on peak-level changes;
- as the local feature extractor inside :class:`dnanet.models.peaknet.CombinedClassifier`.

The architecture is intentionally 1D. For peak classification, the useful
signal is mostly in local shape along the scan axis: peak width, symmetry,
and nearby stutter-like structure. A 1D CNN fits that use case well without
assuming that neighbouring dye channels form a meaningful spatial pattern.

An optional marker embedding lets one model adapt to locus-specific peak
behaviour without maintaining separate models per marker.
"""

from __future__ import annotations

import abc

import torch
from torch import Tensor, nn
from loguru import logger


class BackboneModule(nn.Module, abc.ABC):
    """Base class for backbone + head architectures.

    Subclasses must implement :meth:`backbone`, :meth:`head`, and
    :meth:`backbone_out_features`.
    """

    @abc.abstractmethod
    def backbone(self, x: Tensor | tuple[Tensor, ...]) -> Tensor:
        """Extract features from input."""

    @abc.abstractmethod
    def head(self, features: Tensor) -> Tensor:
        """Classify features into logits."""

    @abc.abstractmethod
    def backbone_out_features(self) -> int:
        """Dimensionality of the backbone output."""

    def forward(self, x: Tensor | tuple[Tensor, ...]) -> Tensor:
        return self.head(self.backbone(x))


class PeakClassificationModel(BackboneModule):
    """Conv1d classifier for peak windows with optional marker embedding.

    The convolutional backbone learns local peak-shape cues from a fixed
    window around each detected peak. The classifier head can be used
    directly for per-peak labels, while :meth:`backbone` is also exposed so
    PeakNet can reuse the learned peak representation.

    When ``use_embedding=True``, the model expects a marker index per peak.
    This gives the network locus context, which is useful because the same
    peak shape can mean different things in different marker regions.

    Args:
        num_classes: Number of output classes.
        width: Peak window width in scan points.
        n_markers: Number of distinct markers (for embedding table).
        embedding_dim: Marker embedding dimension. Set to 0 to disable.
        include_max_pool_dyes: If True, input has 2 channels (peak dye +
            max-pooled other dyes). If False, 1 channel.
        hidden_channels: Channel counts for each Conv1d block.
        kernel_size: Convolution kernel size.
        pooling: Feature pooling strategy — ``"flat"``, ``"avg"``, or
            ``"attn"`` (attention pooling).
        activation: Activation function — ``"relu"``, ``"tanh"``, or ``"gelu"``.
        use_batchnorm: Add BatchNorm1d after each conv.
        bn_momentum: BatchNorm momentum.
        conv_dropout_p: Dropout probability after each conv block.
        head_dropout_p: Dropout probability in the classification head.
        downsample: Downsampling strategy — ``"maxpool"`` or ``"conv"``.
    """

    def __init__(
        self,
        num_classes: int = 2,
        width: int = 120,
        n_markers: int = 28,
        embedding_dim: int = 8,
        use_embedding: bool = True,
        include_max_pool_dyes: bool = False,
        hidden_channels: list[int] | None = None,
        kernel_size: int = 3,
        pooling: str = 'flat',
        activation: str = 'relu',
        use_batchnorm: bool = False,
        bn_momentum: float = 0.1,
        conv_dropout_p: float = 0.0,
        head_dropout_p: float = 0.0,
        downsample: str = 'maxpool',
    ) -> None:
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64]

        if pooling not in {'flat', 'avg', 'attn'}:
            raise ValueError(f"pooling must be 'flat', 'avg', or 'attn', got '{pooling}'")
        if downsample not in {'maxpool', 'conv'}:
            raise ValueError(f"downsample must be 'maxpool' or 'conv', got '{downsample}'")
        if activation not in {'relu', 'tanh', 'gelu'}:
            raise ValueError(f"activation must be 'relu', 'tanh', or 'gelu', got '{activation}'")

        if embedding_dim <= 0 and use_embedding:
            raise ValueError('embedding_dim must be > 0 if use_embedding=True')

        if not use_embedding and embedding_dim > 0:
            logger.warning('embedding_dim > 0 but use_embedding=False. Ignoring.')

        self.pooling = pooling
        in_channels = 2 if include_max_pool_dyes else 1

        # Activation factory
        act_cls = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}[activation]

        # Conv + downsample stack
        conv_blocks: list[nn.Module] = []
        prev_ch = in_channels
        for out_ch in hidden_channels:
            block: list[nn.Module] = [
                nn.Conv1d(
                    prev_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
                ),
            ]
            if use_batchnorm:
                block.append(nn.BatchNorm1d(out_ch, momentum=bn_momentum))
            block.append(act_cls())
            if conv_dropout_p > 0:
                block.append(nn.Dropout(p=conv_dropout_p))
            if downsample == 'maxpool':
                block.append(nn.MaxPool1d(kernel_size=2, stride=2))
            else:
                block.append(nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            conv_blocks.append(nn.Sequential(*block))
            prev_ch = out_ch

        self.conv = nn.Sequential(*conv_blocks)
        self._out_channels = prev_ch

        # Attention pooling projection (if needed)
        if pooling == 'attn':
            self.attn_proj = nn.Linear(self._out_channels, 1)

        # Optional marker embedding
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embed = nn.Embedding(num_embeddings=n_markers, embedding_dim=embedding_dim)

        # Infer feature dimension with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, width)
            conv_out = self.conv(dummy)
            feat_dim = self._pool(conv_out).shape[1]

        in_features = feat_dim + (embedding_dim if self.use_embedding else 0)
        self._backbone_out = in_features

        # Classification head
        head_layers: list[nn.Module] = [nn.Linear(in_features, 64), nn.ReLU()]
        if head_dropout_p > 0:
            head_layers.append(nn.Dropout(p=head_dropout_p))
        head_layers.append(nn.Linear(64, num_classes))
        self._head = nn.Sequential(*head_layers)

    def _pool(self, x: Tensor) -> Tensor:
        """Apply pooling to conv output (B, C, T) → (B, F)."""
        if self.pooling == 'flat':
            return torch.flatten(x, start_dim=1)
        elif self.pooling == 'avg':
            return x.mean(dim=-1)
        else:  # attn
            scores = self.attn_proj(x.permute(0, 2, 1)).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            return (x * weights.unsqueeze(1)).sum(dim=-1)

    def backbone(self, x: Tensor | tuple[Tensor, Tensor]) -> Tensor:
        """Extract features from peak window + optional marker index.

        Args:
            x: Either a tensor of shape ``(B, C, W)`` or a tuple
               ``(peak_tensor, marker_idx)`` where ``marker_idx`` is
               ``(B,)`` long tensor.

        Returns:
            Feature tensor of shape ``(B, F)``.
        """
        match x:
            case (pd, mixd):
                peak_data = pd
                marker_idx = mixd
            case _:
                peak_data = x
                marker_idx = None

        # peak_data: (B, C, W), where C = 1 or 2 (dye + max-pooled other dyes)
        # marker_idx: (B, 1), or None

        features = self._pool(self.conv(peak_data))  # (B, F_p)

        if self.use_embedding:
            if marker_idx is None:
                raise ValueError('Marker index is required for embedding')
            if torch.any(marker_idx < 0) or torch.any(marker_idx >= self.embed.num_embeddings):
                raise ValueError(
                    f'marker_idx out of range: min={marker_idx.min().item()}, '
                    f'max={marker_idx.max().item()}, '
                    f'allowed=[0, {self.embed.num_embeddings - 1}]'
                )
            marker_idx = marker_idx.squeeze(-1)  # (B,)
            emb = self.embed(marker_idx)  # (B, F_e), where F_e = embedding_dim
            features = torch.cat((features, emb), dim=1)  # (B, F), where F = F_p + F_e

        return features

    def head(self, features: Tensor) -> Tensor:
        return self._head(features)

    def backbone_out_features(self) -> int:
        return self._backbone_out
