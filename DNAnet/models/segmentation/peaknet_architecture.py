from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from DNAnet.models.classification.peak_classification_torch import BackboneModule
from DNAnet.models.reconstruction.autoencoder_architectures import AbstractAutoEncoder


class CombinedClassifier(nn.Module):
    def     __init__(self,
                     autoencoder: nn.Module,
                     autoencoder_out_shape: tuple[int, ...],
                     peak_classifier: nn.Module,
                     peak_classifier_out_shape: tuple[int, ...],
                     hidden_dims: List[int] = [256, 128],
                     freeze_autoencoder: bool = True,
                     num_classes : int = 2,
                     default_class: int = 0,
                     combiner_strategy: str = 'mlp',
                     ):
        super().__init__()
        self.autoencoder = autoencoder
        self.peak_classifier = peak_classifier
        self.freeze_autoencoder = freeze_autoencoder
        self.num_classes = num_classes
        self.default_class_idx = default_class
        self.combiner_strategy = combiner_strategy

        if self.freeze_autoencoder:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
            self.autoencoder.eval()

        flat_shape_autoencoder = np.prod(autoencoder_out_shape)
        print(f"Autoencoder output shape: {autoencoder_out_shape}, flattened size: {flat_shape_autoencoder}")
        flat_shape_peak_classifier = np.prod(peak_classifier_out_shape)
        print(f"Peak classifier output shape: {peak_classifier_out_shape}, flattened size: {flat_shape_peak_classifier}")
        flat_shape = flat_shape_autoencoder + flat_shape_peak_classifier
        print(f"Combined feature size: {flat_shape}")


        if combiner_strategy == 'mlp':
            self.combiner = MLPCombiner(
                input_dim=flat_shape,
                hidden_dims=hidden_dims,
                out_dim=self.num_classes,
            )
        elif combiner_strategy == 'film':
            self.combiner = FiLMCombiner(
                global_dim=flat_shape_autoencoder,
                local_dim=flat_shape_peak_classifier,
                hidden_dims=hidden_dims,
                out_dim=self.num_classes,
            )
        elif combiner_strategy == 'attention':
            self.combiner = CrossAttentionCombiner1D(
                global_channels=autoencoder_out_shape[0],  # C from (C, H, W)
                local_dim=flat_shape_peak_classifier,
                embed_dim=32,
                num_heads=4,
                hidden_dims=hidden_dims,
                out_dim=self.num_classes,
            )
        else:
            raise ValueError(f"Unknown combiner strategy: {combiner_strategy}")


    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the combined classifier.
        Args:
            x: Tuple containing:
                - full_image: Tensor of shape (N, C, 4096)
                - peak_windows: NestedTensor of shape (N, N_p, C, W)
                - marker_idxs: NestedTensor of shape (N, N_p)
                - peak_centers: NestedTensor of shape (N, N_p, 2)

        Returns: Tensor of shape (N, num_classes, C, 4096) containing the logits for each class at each position.

        """
        ### DEFINITION OF DIMENSIONS:
        # N: Number of images in batch
        # C: Number of channels/dyes
        # N_p: Number of peaks per image, this is variable and changes from image to image
        # P: Total number of peaks across batch (sum of N_p over N)
        # W: Width of peak window
        # L: Length of electropherogram (4096)
        # F_a: Dimension of autoencoder features (when flattened)
        # F_p: Dimension of peak classifier features (when flattened)

        full_image, peak_windows, marker_idxs, peak_centers = x
        # full_image: (N, C, L)
        # peak_windows: (N, N_p, C, W)
        # marker_idxs: (N, N_p)
        # peak_centers: (N, N_p, 2)

        # 0) FLATTEN TENSORS
        peaks = peak_windows.values() # (P, C, W) where P = sum of N_p over N
        markers = marker_idxs.values() # (P,)
        peak_locations = peak_centers.values() # (P, 2)

        offsets = peak_windows.offsets() # index location where a new image starts in the flattened peaks
        lengths = offsets[1:] - offsets[:-1] # (N,) number of peaks per image
        num_images = lengths.numel() # = N

        image_indices = torch.arange(num_images, dtype=torch.long, device=peaks.device)  # (N,)
        peak_to_image = torch.repeat_interleave(image_indices, lengths) # (P,) mapping each peak index to its source image


        # 1) GLOBAL BRANCH (per image)
        # full_images:  (N, C, L)
        if self.freeze_autoencoder:
            with torch.no_grad():
                autoencoder_out = self.autoencoder.encoder(full_image) # Output can be any shape
        else:
            autoencoder_out = self.autoencoder.encoder(full_image) # Output can be any shape

        # global_encoded_shape = self.autoencoder.encoded_shape() # Output can be any shape
        # print(f"Shape of autoencoder_out: {global_encoded_shape}")

        flat_encoded = torch.flatten(autoencoder_out, start_dim=1) # (N, F_a)

        # Map each peak to its image’s global features -> (P, F_a)
        autoencoder_out_per_peak = flat_encoded.index_select(dim=0, index=peak_to_image) # (P, F_a)
        # print(f"Autoencoder features per peak shape: {autoencoder_out_per_peak.shape}")

        # 2) LOCAL BRANCH (per peak)
        # peaks:   (P, C, W)
        # markers: (P,)
        local_features = self.peak_classifier.backbone((peaks, markers)) # (P, F_p)
        # print(f"Local features shape: {local_features.shape}")

        # 3) COMBINE + CLASSIFY (per peak)
        if self.combiner_strategy == 'mlp':
            # print(f"shape autoencoder_out_per_peak: {autoencoder_out_per_peak.shape}, shape local_features: {local_features.shape}")
            # Concatenate global+local features per peak -> (P, F_a + F_p)
            combined_features = torch.cat((autoencoder_out_per_peak, local_features), dim=1) # (P, F_a + F_p)
            # print(f"Combined features shape: {combined_features.shape}")
            logits = self.combiner(combined_features) # (P, num_classes)
        elif self.combiner_strategy == 'film':
            logits = self.combiner(autoencoder_out_per_peak, local_features) # (P, num_classes)
        elif self.combiner_strategy == 'attention':
            # Need to pass full autoencoder output (N, C, W) for attention mechanism, does not use the flattened features
            logits = self.combiner(autoencoder_out, local_features, peak_to_image) # (P, num_classes)
        else:
            raise ValueError(f"Unknown combiner strategy: {self.combiner_strategy}")

        # logits now contains a classification for each peak: (P, num_classes)

        # 4) MAP BACK TO IMAGE
        # create output tensor
        N, C, L = full_image.size() # N: num images, C: num dyes, L: length (4096)
        # very_negative = torch.finfo(logits.dtype).min
        # segmented = torch.full((N, C, L, self.num_classes), fill_value=very_negative, device=full_image.device) # (N, C, 4096)
        segmented = torch.zeros((N, C, L, self.num_classes), device=logits.device, dtype=logits.dtype) # (N, C, L, num_classes)
        # Set default class (background noise) logits to 8, while other classes are 0, this should make sure that non-peak regions are classified as noise
        segmented[:, :, :, self.default_class_idx] = 8.0

        # peak_locations: (P, 2) -> dye_idx and position
        # This assumes top annotation
        dye_idx = peak_locations[:, 0].to(torch.long)   # (P,)
        pos_idx = peak_locations[:, 1].to(torch.long)   # (P,)

        # Assign logits to the correct positions in the output tensor
        # segmented[peak_to_image, dye_idx, pos_idx, :] and logits are of shape (P, num_classes)
        segmented[peak_to_image, dye_idx, pos_idx, :] = logits  # (N, C, L, num_classes)

        # Move dimensions for loss function compatibility
        segmented = segmented.permute(0, 3, 1, 2) # (N, num_classes, C, L)

        return segmented # (N, num_classes, C, L)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_autoencoder:
            self.autoencoder.eval()
        return self


class PeakOnlyClassifier(nn.Module):
    def __init__(self,
                 peak_classifier: nn.Module,
                 num_classes : int = 2,
                 default_class: int = 0
                 ):
        super().__init__()
        self.peak_classifier = peak_classifier
        self.num_classes = num_classes
        self.default_class_idx = default_class

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        ### DEFINITION OF DIMENSIONS:
        # N: Number of images in batch
        # C: Number of channels/dyes
        # N_p: Number of peaks per image, this is variable and changes from image to image
        # P: Total number of peaks across batch (sum of N_p over N)
        # W: Width of peak window
        # L: Length of electropherogram (4096)
        # F_a: Dimension of autoencoder features (when flattened)
        # F_p: Dimension of peak classifier features (when flattened)
        # TODO: reduce duplicated code from CombinedClassifier
        full_image, peak_windows, marker_idxs, peak_centers = x

        # peak_windows: (N, N_p, C, W)
        # marker_idxs: (N, N_p)
        # peak_centers: (N, N_p, 2)

        # 0) FLATTEN TENSORS
        peaks = peak_windows.values() # (P, C, W) where P = sum of N_p over N
        markers = marker_idxs.values() # (P,)
        peak_locations = peak_centers.values() # (P, 2)

        offsets = peak_windows.offsets() # index location where a new image starts in the flattened peaks
        lengths = offsets[1:] - offsets[:-1] # (N,) number of peaks per image
        num_images = lengths.numel() # = N

        image_indices = torch.arange(num_images, dtype=torch.long, device=peaks.device)  # (N,)
        peak_to_image = torch.repeat_interleave(image_indices, lengths) # (P,) mapping each peak index to its source image


        logits = self.peak_classifier((peaks, markers)) # (P, num_classes)

        # 4) MAP BACK TO IMAGE
        # create output tensor

        N, C, L = full_image.size() # N: num images, C: num dyes, L: length (4096)
        # very_negative = torch.finfo(logits.dtype).min
        # segmented = torch.full((N, C, L, self.num_classes), fill_value=very_negative, device=full_image.device) # (N, C, 4096)
        segmented = torch.zeros((N, C, L, self.num_classes), device=logits.device, dtype=logits.dtype) # (N, C, L, num_classes)
        # Set default class (background noise) logits to 8, while other classes are 0, this should make sure that non-peak regions are classified as noise
        segmented[:, :, :, self.default_class_idx] = 8.0

        # peak_locations: (P, 2) -> dye_idx and position
        # This assumes top annotation
        dye_idx = peak_locations[:, 0].to(torch.long)   # (P,)
        pos_idx = peak_locations[:, 1].to(torch.long)   # (P,)

        # Assign logits to the correct positions in the output tensor
        # segmented[peak_to_image, dye_idx, pos_idx, :] and logits are of shape (P, num_classes)
        segmented[peak_to_image, dye_idx, pos_idx, :] = logits  # (N, C, L, num_classes)

        # Move dimensions for loss function compatibility
        segmented = segmented.permute(0, 3, 1, 2) # (N, num_classes, C, L)

        return segmented # (N, num_classes, C, L)


class MLPCombiner(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 out_dim: int):
        super().__init__()
        self.model = nn.Sequential()
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.model.add_module(f"fc{i}", nn.Linear(current_dim, hidden_dim))
            self.model.add_module(f"relu{i}", nn.ReLU())
            current_dim = hidden_dim
        self.model.add_module("output", nn.Linear(current_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"MLPCombiner input shape: {x.shape}")
        return self.model(x)


class FiLMCombiner(nn.Module):
    def __init__(self,
                 global_dim: int,
                 local_dim: int,
                 hidden_dims: List[int],
                 out_dim: int):
        """
        Args:
            global_dim: Dimension of autoencoder output (F_a)
            local_dim: Dimension of peak classifier backbone output (F_p)
            hidden_dims: Hidden layers for the final classifier MLP
            out_dim: Number of classes
        """
        super().__init__()

        # 1. FiLM Generator: Maps Global Features -> Scale (gamma) and Shift (beta)
        # We output 2 * local_dim to get both gamma and beta
        self.film_generator = nn.Linear(global_dim, 2 * local_dim)

        # Initialize gamma to 1 and beta to 0 to start as an identity mapping (stability trick)
        self.film_generator.weight.data.normal_(0, 0.02)
        self.film_generator.bias.data.fill_(0)
        # Bias for gamma (first half) should be 1.0 initially
        self.film_generator.bias.data[:local_dim] = 1.0

        # 2. Classifier: Takes the modulated local features
        self.classifier = nn.Sequential()
        current_dim = local_dim

        for i, hidden_dim in enumerate(hidden_dims):
            self.classifier.add_module(f"fc{i}", nn.Linear(current_dim, hidden_dim))
            self.classifier.add_module(f"relu{i}", nn.ReLU())
            current_dim = hidden_dim

        self.classifier.add_module("output", nn.Linear(current_dim, out_dim))

    def forward(self, global_features: torch.Tensor, local_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_features: (P, F_a) - The context
            local_features:  (P, F_p) - The target to modulate
        """
        # Predict modulation parameters from global context
        film_params = self.film_generator(global_features) # (P, 2 * F_p)

        # Split into gamma (scale) and beta (shift)
        gamma, beta = torch.chunk(film_params, 2, dim=1) # Each is (P, F_p)

        # Apply Affine Transformation: Scale + Shift
        modulated_features = (gamma * local_features) + beta # (P, F_p)

        # Apply optional non-linearity after modulation (common in FiLM blocks)
        modulated_features = torch.relu(modulated_features)

        # Classify
        return self.classifier(modulated_features)

class CrossAttentionCombiner1D(nn.Module):
    """
    Cross-Attention module for 1D signals (electropherograms).

    - Global: Autoencoder output (N, C_global, W) - spatial feature map
    - Local: Peak classifier features (P, D_local) - flattened per-peak features

    Uses cross-attention where local features "query" the global signal map.
    """

    def __init__(self,
                 global_channels: int,  # Number of channels in autoencoder output
                 local_dim: int,        # Dimension of local peak feature
                 embed_dim: int = 128,  # Dimension for attention projections
                 num_heads: int = 4,
                 hidden_dims: List[int] = None,
                 out_dim: int = 2):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # 1. Project both modalities to common embedding dimension
        # For global features (spatial map): use Conv1d to project channels
        self.global_proj = nn.Conv1d(global_channels, embed_dim, kernel_size=1)

        # For local features: use Linear projection
        self.local_proj = nn.Linear(local_dim, embed_dim)

        # 2. Multi-Head Cross-Attention
        # Query: local features (per-peak context)
        # Key/Value: global signal map (entire electropherogram)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        # 3. Optional positional encoding for the 1D signal
        # Learnable position embeddings for the signal length
        self.positional_encoding = nn.Parameter(
            torch.randn(1, embed_dim, 1)  # (1, embed_dim, 1) - broadcasted to (1, embed_dim, W)
        )

        # 4. Final MLP Classifier
        # Concatenate: original local features + attention output
        self.classifier = nn.Sequential()
        current_dim = local_dim + embed_dim

        for i, hidden_dim in enumerate(hidden_dims):
            self.classifier.add_module(f"fc{i}", nn.Linear(current_dim, hidden_dim))
            self.classifier.add_module(f"relu{i}", nn.ReLU())
            # self.classifier.add_module(f"dropout{i}", nn.Dropout(0.1))
            current_dim = hidden_dim

        self.classifier.add_module("output", nn.Linear(current_dim, out_dim))

    def forward(self,
                global_signal: torch.Tensor,      # (N, C_global, W)
                local_features: torch.Tensor,      # (P, D_local)
                peak_to_image_idx: torch.Tensor) -> torch.Tensor:  # (P,)
        """
        Args:
            global_signal: (N, C_global, W) - Autoencoder output, one 1D signal per image
            local_features: (P, D_local) - Flattened features per peak
            peak_to_image_idx: (P,) - Index mapping each peak to its source image

        Returns:
            logits: (P, num_classes) - Classification logits for each peak
        """

        # --- Prepare Global Keys/Values (from 1D signal map) ---
        # 1. Project global channels: (N, C_global, W) -> (N, embed_dim, W)
        global_embed = self.global_proj(global_signal)  # (N, E, W)
        N, E, W = global_embed.shape

        # 2. Add positional encoding (optional, but helpful for 1D signals)
        global_embed = global_embed + self.positional_encoding

        # 3. Transpose to sequence format for attention: (N, E, W) -> (N, W, E)
        global_embed = global_embed.transpose(1, 2)  # (N, W, E)

        # 4. Select the correct global context for each peak
        # This expands the global signal maps to match the number of peaks
        peak_global_context = global_embed[peak_to_image_idx]  # (P, W, E)

        # --- Prepare Local Queries ---
        # 1. Project local features: (P, D_local) -> (P, E)
        local_embed = self.local_proj(local_features)  # (P, E)

        # 2. Add sequence dimension for attention: (P, E) -> (P, 1, E)
        # The query is a single vector per peak (asking about that specific peak)
        query = local_embed.unsqueeze(1)  # (P, 1, E)

        # --- Cross-Attention ---
        # Query: "What does this peak look like?" (P, 1, E)
        # Key/Value: "What's in the rest of the signal?" (P, W, E)
        # Output: Weighted combination of signal regions relevant to this peak
        with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
        ):
            attn_out, attn_weights = self.mha(query, peak_global_context, peak_global_context)

        # attn_out shape: (P, 1, E) - the attended global context for each peak
        attn_out = attn_out.squeeze(1)  # (P, E)
        attn_out = self.norm(attn_out)

        # --- Combine and Classify ---
        # Concatenate original local signal with the retrieved global context
        # This preserves the original peak shape while adding surrounding context
        combined = torch.cat([local_features, attn_out], dim=1)  # (P, D_local + E)

        logits = self.classifier(combined)  # (P, num_classes)

        return logits

