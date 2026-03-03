import abc
from typing import Tuple, List

import torch
from torch import nn


class BackboneModule(nn.Module, abc.ABC):
    """
    Base class for models with a backbone + head architecture.
    """
    def __init__(self):
        super().__init__()

    def head(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def backbone(self, x) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def backbone_out_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)



class PeakClassificationModel(BackboneModule):
    def __init__(self,
                 num_classes: int,
                 width: int = 120,
                 n_markers: int = 27,
                 embedding_dim: int = 8,
                 include_max_pool_dyes: bool = False,
                 hidden_channels: List[int] = [16, 32],
                 kernel_size: int = 3,
                 pooling: str = "flat",                 # "flat" (original), "avg", or "attn"
                 activation: str = "relu",              # "relu" (original) or "tanh" or "gelu"

                 # normalization / dropout / downsampling controls
                 use_batchnorm: bool = False,
                 bn_momentum: float = 0.1,
                 conv_dropout_p: float = 0.0,
                 head_dropout_p: float = 0.0,
                 downsample: str = "maxpool",           # "maxpool" (original) or "conv"
                 ):
        super().__init__()
        assert pooling in {"avg", "attn", "flat"}, "pooling must be 'flat', 'avg', or 'attn'"
        assert downsample in {"maxpool", "conv"}, "downsample must be 'maxpool' or 'conv'"
        assert activation in {"relu", "tanh", "gelu"}, "activation must be 'relu', 'tanh', or 'gelu'"
        self.pooling = pooling
        self.use_batchnorm = use_batchnorm
        self.conv_dropout_p = conv_dropout_p
        self.include_max_pool_dyes = include_max_pool_dyes

        in_channels = 2 if include_max_pool_dyes else 1

        # ----------------
        # Conv/downsample stack
        # ----------------
        convs = []
        prev_channels = in_channels
        for out_channels in hidden_channels:
            block = []

            # main conv (keep length with padding)
            block.append(nn.Conv1d(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1
            ))
            if use_batchnorm:
                block.append(nn.BatchNorm1d(out_channels, momentum=bn_momentum))

            if activation == "relu":
                block.append(nn.ReLU())
            elif activation == "tanh":
                block.append(nn.Tanh())
            elif activation == "gelu":
                block.append(nn.GELU())

            if conv_dropout_p and conv_dropout_p > 0.0:
                block.append(nn.Dropout(p=conv_dropout_p))

            # downsampling: MaxPool (original) or strided conv
            if downsample == "maxpool":
                block.append(nn.MaxPool1d(kernel_size=2, stride=2))
            else:  # "conv"
                pad = (3 // 2)
                block.append(nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=pad
                ))


            convs.append(nn.Sequential(*block))
            prev_channels = out_channels

        self.conv = nn.Sequential(*convs)
        self._out_channels = prev_channels

        # ----------------
        # Pooling head selector
        # ----------------
        if self.pooling == "attn":
            self.attn_proj = nn.Linear(self._out_channels, 1)

        # ----------------
        # Optional marker embedding
        # ----------------
        self.use_embedding = embedding_dim > 0
        if self.use_embedding:
            self.embed = nn.Embedding(num_embeddings=n_markers, embedding_dim=embedding_dim)


        # ----------------
        # Dynamically infer head input size using a dummy forward through conv + pooling
        # ----------------
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, width)
            out = self.conv(dummy)  # (1, C_out, T_out)
            if self.pooling == "flat":
                feat_dim = out.flatten(1).shape[1]
            elif self.pooling == "avg":
                feat_dim = out.mean(dim=-1).shape[1]
            else:  # "attn"
                # compute attention pooling with the same projection implemented in forward
                scores = self.attn_proj(out.permute(0, 2, 1)).squeeze(-1)  # (1, T)
                weights = torch.softmax(scores, dim=-1)
                pooled = (out * weights.unsqueeze(1)).sum(dim=-1)  # (1, C)
                feat_dim = pooled.shape[1]

        in_features = feat_dim + (embedding_dim if self.use_embedding else 0)
        self._backbone_out_shape = (in_features,)

        # ----------------
        # Classification head
        # ----------------

        head_layers = [
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU()
        ]
        if head_dropout_p and head_dropout_p > 0.0:
            head_layers.append(nn.Dropout(p=head_dropout_p))
        head_layers.append(nn.Linear(in_features=64, out_features=num_classes))
        self._head = nn.Sequential(*head_layers)

    # ---- pooling implementations ----
    def _global_avg_pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, C)
        return x.mean(dim=-1)

    def _attention_pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, C)
        scores = self.attn_proj(x.permute(0, 2, 1)).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)                  # (B, T)
        return (x * weights.unsqueeze(1)).sum(dim=-1)            # (B, C)

    # ---- forward ----
    def backbone(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, marker_idx = x
        # x shape : (B, C_in, W)
        # marker_idx shape : (B,)


        x = self.conv(x)  # (B, C_out, T_out)

        if self.pooling == "flat":
            x = torch.flatten(x, start_dim=1)        # (B, C_out * T_out) — matches original
        elif self.pooling == "avg":
            x = self._global_avg_pool(x)             # (B, C_out)
        else:  # "attn"
            x = self._attention_pool(x)              # (B, C_out)

        if self.use_embedding:
            assert marker_idx[0] != -1, f"Marker indices must be provided when using embeddings. Got {marker_idx}"
            marker_embeddings = self.embed(marker_idx)  # (B, embedding_dim)
            x = torch.cat((x, marker_embeddings), dim=1)

        return x

    def head(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._head(x)  # (B, num_classes)

    def backbone_out_shape(self) -> Tuple[int, ...]:
        return self._backbone_out_shape
