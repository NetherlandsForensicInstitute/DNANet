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
        # print(f"shape autoencoder_out_per_peak: {autoencoder_out_per_peak.shape}, shape local_features: {local_features.shape}")
        # Concatenate global+local features per peak -> (P, F_a + F_p)
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
        film_params = self.film_generator(global_features) # (P, 2 * F_p)
        gamma, beta = torch.chunk(film_params, 2, dim=1) # Each is (P, F_p)
        # Apply non-linearity after modulation with ReLu (common in FiLM blocks)
        modulated = torch.relu(gamma * local_features + beta) # (P, F_p)
        return self.classifier(modulated) # (P, num_classes)


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
        E: embedding dimension.
        """
        g = self.global_proj(global_signal) + self.positional_encoding # (N, E, W)
        g = g.transpose(1, 2)  # (N, W, E)
        peak_context = g[peak_to_image]  # (P, W, E)

        query = self.local_proj(local_features).unsqueeze(1)  # (P, 1, E)
        # attn_out shape: (P, 1, E) - the attended global context for each peak
        attn_out, _ = self.mha(query, peak_context, peak_context)
        attn_out = self.norm(attn_out.squeeze(1))  # (P, E)

        combined = torch.cat([local_features, attn_out], dim=1) # (P, D_local + E)
        return self.classifier(combined) # (P, num_classes)


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
        ### Dimension definitions ###
        # N: Number of images in batch
        # C: Number of channels/dyes
        # L: Length of electropherogram (4096)
        # C_peak: 
        # P: Total number of peaks across batch (sum of N_p over N)
        # W: Width of peak window
        # F_a: Dimension of autoencoder features (when flattened)
        # F_p: Dimension of peak classifier features (when flattened)

        N, C, L = full_image.shape

        # Map each peak back to its source image
        peak_to_image = torch.repeat_interleave(
            torch.arange(N, device=full_image.device), peak_counts
        )  # (P,)

        # 1) GLOBAL BRANCH (per image)
        ctx = torch.no_grad() if self.freeze_autoencoder else torch.enable_grad()
        with ctx:
            ae_encoded = self.autoencoder.encode(full_image) # Output has no defined shape
        ae_flat = torch.flatten(ae_encoded, start_dim=1)  # (N, F_a)
        
        # Map each peak to its image’s global features -> (P, F_a)
        ae_per_peak = ae_flat[peak_to_image]  # (P, F_a)


        # 2) LOCAL BRANCH (per peak)
        # peaks:   (P, C, W)
        # markers: (P,)
        local_features = self.peak_classifier.backbone((peak_windows, marker_idxs))  # (P, F_p)


        # 3) COMBINE + CLASSIFY (per peak)
        if self.combiner_name == 'attention':
            # Need to pass full autoencoder output (N, C, W) for attention mechanism, does not use the flattened features
            logits = self.combiner(
                ae_per_peak,
                local_features,
                peak_to_image=peak_to_image,
                global_signal=ae_encoded,
            ) # (P, num_classes)
        else:
            # For the other combination strategies we don't need the full autoencoder output
            logits = self.combiner(ae_per_peak, local_features) # (P, num_classes)
        # logits now contains a classification for each peak: (P, num_classes)


        # 4) MAP BACK TO IMAGE
        # Scatter logits back to image coordinates
        segmented = torch.zeros(
            (N, C, L, self.num_classes),
            device=logits.device,
            dtype=logits.dtype,
        ) # (N, C, L, num_classes)

        # Set default class (background noise) logits to 8, while other classes are 0,
        # this should make sure that non-peak regions are classified as noise
        # The number 8 was chosen arbitrarily to push the model to predict default in any areas without peaks.
        segmented[:, :, :, self.default_class_idx] = 8.0

        # peak_locations: (P, 2) -> dye_idx and position
        # This assumes top annotation
        dye_idx = peak_centers[:, 0].long()
        pos_idx = peak_centers[:, 1].long()

        # Assign logits to the correct positions in the output tensor
        # segmented[peak_to_image, dye_idx, pos_idx, :] and logits are of shape (P, num_classes)
        segmented[peak_to_image, dye_idx, pos_idx, :] = logits

        # Move dimensions for loss function compatibility
        return segmented.permute(0, 3, 1, 2)  # (N, num_classes, C, L)

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
        ### DEFINITION OF DIMENSIONS:
        # N: Number of images in batch
        # C: Number of channels/dyes
        # L: Length of electropherogram (4096)
        # N_p: Number of peaks per image, this is variable and changes from image to image
        # P: Total number of peaks across batch (sum of N_p over N)
        # W: Width of peak window
        # F_a: Dimension of autoencoder features (when flattened)
        # F_p: Dimension of peak classifier features (when flattened)
        N, C, L = full_image.shape

        peak_to_image = torch.repeat_interleave(
            torch.arange(N, device=full_image.device), peak_counts
        ) # (P,)

        logits = self.peak_classifier((peak_windows, marker_idxs)) # (P, num_classes)

        segmented = torch.zeros(
            (N, C, L, self.num_classes),
            device=logits.device,
            dtype=logits.dtype,
        ) # (N, C, L, num_classes)
        
        # See notes from CombinedClassifier
        segmented[:, :, :, self.default_class_idx] = 8.0

        dye_idx = peak_centers[:, 0].long() # (P,)
        pos_idx = peak_centers[:, 1].long() # (P,)
        segmented[peak_to_image, dye_idx, pos_idx, :] = logits # (N, C, L, num_classes)

        return segmented.permute(0, 3, 1, 2)  # (N, num_classes, C, L)
