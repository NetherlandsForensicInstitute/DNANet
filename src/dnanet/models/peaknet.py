"""Combined PeakNet architecture for per-scan-point classification.
PeakNet is a dual-branch model that produces per-position class logits
over the entire electropherogram:

1. **Global branch** — An autoencoder encoder compresses the full EPG
   into a latent representation, providing whole-profile context.
2. **Local branch** — A peak classifier backbone extracts features from
   individual peak windows.
3. **Combiner** — Merges global + local features per peak and produces
   class logits, which are scattered back to image coordinates.

Three combiner strategies are available:

- :class:`MLPCombiner` — Simple concatenation + MLP.
- :class:`FiLMCombiner` — Feature-wise Linear Modulation (global
  conditions local features via learned scale + shift).
- :class:`CrossAttentionCombiner` — Local features *query* the global
  signal map via multi-head cross-attention.

A :class:`PeakOnlyClassifier` variant operates without the autoencoder,
using only peak-level classification.

Design pattern: **Strategy**
    The combiner is selected at construction time and swapped
    transparently. The forward pass is identical regardless of combiner.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Combiner strategies
# ---------------------------------------------------------------------------


class MLPCombiner(nn.Module):
    """Concatenate global + local features → MLP → class logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        out_dim: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for hd in hidden_dims:
            layers += [nn.Linear(current, hd), nn.ReLU()]
            current = hd
        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        global_features: Tensor,
        local_features: Tensor,
        **_kwargs,
    ) -> Tensor:
        return self.net(torch.cat((global_features, local_features), dim=1))


class FiLMCombiner(nn.Module):
    """Feature-wise Linear Modulation: global conditions local features.

    The global context generates per-feature scale (γ) and shift (β)
    parameters that modulate the local peak features before classification.
    """

    def __init__(
        self,
        global_dim: int,
        local_dim: int,
        hidden_dims: list[int],
        out_dim: int,
    ) -> None:
        super().__init__()

        # FiLM generator: global → (γ, β)
        self.film_generator = nn.Linear(global_dim, 2 * local_dim)
        # Initialize γ→1, β→0 for identity at start
        nn.init.normal_(self.film_generator.weight, 0, 0.02)
        nn.init.zeros_(self.film_generator.bias)
        self.film_generator.bias.data[:local_dim] = 1.0

        # Classifier on modulated features
        layers: list[nn.Module] = []
        current = local_dim
        for hd in hidden_dims:
            layers += [nn.Linear(current, hd), nn.ReLU()]
            current = hd
        layers.append(nn.Linear(current, out_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        global_features: Tensor,
        local_features: Tensor,
        **_kwargs,
    ) -> Tensor:
        film_params = self.film_generator(global_features)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        modulated = torch.relu(gamma * local_features + beta)
        return self.classifier(modulated)


class CrossAttentionCombiner(nn.Module):
    """Cross-attention: local peak features query global signal map.

    The local features become queries attending over the spatial
    positions of the autoencoder's encoded signal.
    """

    def __init__(
        self,
        global_channels: int,
        local_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        hidden_dims: list[int] | None = None,
        out_dim: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.global_proj = nn.Conv1d(global_channels, embed_dim, kernel_size=1)
        self.local_proj = nn.Linear(local_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, embed_dim, 1))

        layers: list[nn.Module] = []
        current = local_dim + embed_dim
        for hd in hidden_dims:
            layers += [nn.Linear(current, hd), nn.ReLU()]
            current = hd
        layers.append(nn.Linear(current, out_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        global_features: Tensor,
        local_features: Tensor,
        *,
        peak_to_image: Tensor,
        global_signal: Tensor,
    ) -> Tensor:
        """Args:
        global_features: Unused (kept for interface compat).
        local_features: (P, D_local) per-peak features.
        peak_to_image: (P,) mapping each peak to its source image.
        global_signal: (N, C_global, W) autoencoder encoded output.
        """
        g = self.global_proj(global_signal) + self.positional_encoding
        g = g.transpose(1, 2)  # (N, W, E)
        peak_context = g[peak_to_image]  # (P, W, E)

        query = self.local_proj(local_features).unsqueeze(1)  # (P, 1, E)
        attn_out, _ = self.mha(query, peak_context, peak_context)
        attn_out = self.norm(attn_out.squeeze(1))  # (P, E)

        combined = torch.cat([local_features, attn_out], dim=1)
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------


class CombinedClassifier(nn.Module):
    """Dual-branch model: autoencoder (global) + peak classifier (local).

    Args:
        autoencoder: Encoder-decoder module (only encoder is used in forward).
        autoencoder_out_shape: Shape of encoder output (excluding batch).
        peak_classifier: Peak classification backbone module.
        peak_classifier_out_features: Output dimension of the backbone.
        hidden_dims: Hidden layer sizes for the combiner MLP.
        num_classes: Number of output classes.
        default_class: Index of the default (background/noise) class.
        freeze_autoencoder: Freeze autoencoder weights during training.
        combiner: Combiner strategy — ``"mlp"``, ``"film"``, or
            ``"attention"``.
    """

    def __init__(
        self,
        autoencoder: nn.Module,
        peak_classifier: nn.Module,
        autoencoder_out_shape: tuple[int, ...] | None = None,
        peak_classifier_out_features: int | None = None,
        hidden_dims: list[int] | None = None,
        num_classes: int = 2,
        default_class: int = 0,
        freeze_autoencoder: bool = True,
        combiner: str = 'mlp',
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.autoencoder = autoencoder
        self.peak_classifier = peak_classifier
        self.freeze_autoencoder = freeze_autoencoder
        self.num_classes = num_classes
        self.default_class_idx = default_class
        self.combiner_name = combiner

        if freeze_autoencoder:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
            self.autoencoder.eval()

        # Auto-infer output shapes from sub-models when not explicitly given
        if autoencoder_out_shape is None:
            if hasattr(autoencoder, 'encoded_shape'):
                autoencoder_out_shape = autoencoder.encoded_shape()
            else:
                raise ValueError(
                    'Cannot infer autoencoder output shape. Pass '
                    'autoencoder_out_shape or use an autoencoder with '
                    'an encoded_shape() method.'
                )

        if peak_classifier_out_features is None:
            if hasattr(peak_classifier, 'backbone_out_features'):
                peak_classifier_out_features = peak_classifier.backbone_out_features()
            else:
                raise ValueError(
                    'Cannot infer peak_classifier output features. Pass '
                    'peak_classifier_out_features or use a classifier with '
                    'a backbone_out_features() method.'
                )

        self._autoencoder_out_shape = autoencoder_out_shape

        flat_ae = int(np.prod(autoencoder_out_shape))
        flat_pc = int(peak_classifier_out_features)

        if combiner == 'mlp':
            self.combiner = MLPCombiner(
                flat_ae + flat_pc,
                hidden_dims,
                num_classes,
            )
        elif combiner == 'film':
            self.combiner = FiLMCombiner(
                flat_ae,
                flat_pc,
                hidden_dims,
                num_classes,
            )
        elif combiner == 'attention':
            self.combiner = CrossAttentionCombiner(
                global_channels=autoencoder_out_shape[0],
                local_dim=flat_pc,
                hidden_dims=hidden_dims,
                out_dim=num_classes,
            )
        else:
            raise ValueError(f'Unknown combiner strategy: {combiner}')

    def forward(
        self,
        full_image: Tensor,
        peak_windows: Tensor,
        marker_idxs: Tensor,
        peak_centers: Tensor,
        peak_counts: Tensor,
    ) -> Tensor:
        """Forward pass producing per-position class logits.

        Args:
            full_image: (N, C, L) full EPG profiles.
            peak_windows: (P, C_peak, W) concatenated peak windows.
            marker_idxs: (P,) marker indices per peak.
            peak_centers: (P, 2) each row is ``[dye_idx, position]``.
            peak_counts: (N,) number of peaks per image.

        Returns:
            Logits of shape (N, num_classes, C, L).
        """
        N, C, L = full_image.shape

        # Map each peak back to its source image
        peak_to_image = torch.repeat_interleave(
            torch.arange(N, device=full_image.device), peak_counts
        )

        # Global branch
        ctx = torch.no_grad() if self.freeze_autoencoder else torch.enable_grad()
        with ctx:
            ae_encoded = self.autoencoder.encode(full_image)
        ae_flat = torch.flatten(ae_encoded, start_dim=1)  # (N, F_a)
        ae_per_peak = ae_flat[peak_to_image]  # (P, F_a)

        # Local branch
        local_features = self.peak_classifier.backbone((peak_windows, marker_idxs))  # (P, F_p)

        # Combine
        if self.combiner_name == 'attention':
            logits = self.combiner(
                ae_per_peak,
                local_features,
                peak_to_image=peak_to_image,
                global_signal=ae_encoded,
            )
        else:
            logits = self.combiner(ae_per_peak, local_features)

        # Scatter logits back to image coordinates
        segmented = torch.zeros(
            (N, C, L, self.num_classes),
            device=logits.device,
            dtype=logits.dtype,
        )
        segmented[:, :, :, self.default_class_idx] = 8.0

        dye_idx = peak_centers[:, 0].long()
        pos_idx = peak_centers[:, 1].long()
        segmented[peak_to_image, dye_idx, pos_idx, :] = logits

        return segmented.permute(0, 3, 1, 2)  # (N, K, C, L)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_autoencoder:
            self.autoencoder.eval()
        return self


class PeakOnlyClassifier(nn.Module):
    """Peak-only classifier (no autoencoder global branch).

    Uses only peak-level classification features, scattered back to
    full image coordinates.

    Args:
        peak_classifier: Peak classification backbone.
        num_classes: Number of output classes.
        default_class: Background/noise class index.
    """

    def __init__(
        self,
        peak_classifier: nn.Module,
        num_classes: int = 2,
        default_class: int = 0,
    ) -> None:
        super().__init__()
        self.peak_classifier = peak_classifier
        self.num_classes = num_classes
        self.default_class_idx = default_class

    def forward(
        self,
        full_image: Tensor,
        peak_windows: Tensor,
        marker_idxs: Tensor,
        peak_centers: Tensor,
        peak_counts: Tensor,
    ) -> Tensor:
        """Forward pass. Same signature as :class:`CombinedClassifier`."""
        N, C, L = full_image.shape

        peak_to_image = torch.repeat_interleave(
            torch.arange(N, device=full_image.device), peak_counts
        )

        logits = self.peak_classifier((peak_windows, marker_idxs))

        segmented = torch.zeros(
            (N, C, L, self.num_classes),
            device=logits.device,
            dtype=logits.dtype,
        )
        segmented[:, :, :, self.default_class_idx] = 8.0

        dye_idx = peak_centers[:, 0].long()
        pos_idx = peak_centers[:, 1].long()
        segmented[peak_to_image, dye_idx, pos_idx, :] = logits

        return segmented.permute(0, 3, 1, 2)  # (N, K, C, L)
