from typing import List, Optional, Tuple, Sequence

import torch
import torchmetrics
import torchvision
from torch import nn
from torchmetrics import Metric

from DNAnet.data.data_models.extracted_peak import ExtractedPeak
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.preprocessing.peak_utils import MARKER_TO_IDX
from DNAnet.models.HIDImageBaseModel import HIDImageBaseModel
from DNAnet.models.classification.peak_classification_torch import PeakClassificationModel
from DNAnet.models.prediction import Prediction


class PeakClassification(HIDImageBaseModel):
    def __init__(self,
                 labels: List[str],
                 device: Optional[str] = None,
                 window_size: int = 120,
                 include_marker: bool = True,
                 channels: List[int] = [32, 64],
                 kernel_size: int = 3,
                 pooling: str = 'avg',     # 'avg' or 'attn'
                 include_max_pool_dyes: bool = False,
                 loss_fn: str = "cross_entropy",
                 use_batchnorm: bool = False,
                 bn_momentum: float = 0.1,
                 conv_dropout_p: float = 0.0,
                 head_dropout_p: float = 0.0,
                 downsample: str = "maxpool",           # "maxpool" (original) or "conv",
                 activation: str = "relu",
                 ):

        self.num_classes = len(labels)
        embedding_dim = 8 if include_marker else 0

        model = PeakClassificationModel(
            num_classes=self.num_classes,
            width=window_size,
            n_markers=len(MARKER_TO_IDX) + 1,
            embedding_dim=embedding_dim,
            include_max_pool_dyes=include_max_pool_dyes,
            hidden_channels=channels,
            kernel_size=kernel_size,
            pooling=pooling,
            use_batchnorm=use_batchnorm,
            bn_momentum=bn_momentum,
            conv_dropout_p=conv_dropout_p,
            head_dropout_p=head_dropout_p,
            downsample=downsample,
            activation=activation
        )


        self.window_size = window_size
        self.label_to_idx = dict(zip(labels, range(self.num_classes)))
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()} # invert the mapping
        self.labels = labels
        self.include_marker = include_marker
        self.include_max_pool_dyes = include_max_pool_dyes
        if loss_fn == "cross_entropy":
            loss = nn.CrossEntropyLoss()
        elif loss_fn == "focal":
            loss = torchvision.ops.sigmoid_focal_loss
        elif loss_fn == "KL_divergence":
            loss = self._kl_div_loss
            self._kl_div = nn.KLDivLoss(reduction="batchmean")
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        super().__init__(model, loss, device, apply_allele_caller=False)

    def _kl_div_loss(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=1)
        targets = nn.functional.one_hot(y_true, num_classes=self.num_classes).to(dtype=log_probs.dtype)
        return self._kl_div(log_probs, targets)

    def get_input(self, peak: ExtractedPeak) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the input for a single image as a tensor on the correct
        device.
        Output shape: ((C, W), (1,))


        Returns: tensor of shape (C, W), type torch.float32 and marker tensor of shape (1,), type torch.long

        """
        x = torch.as_tensor(peak.data, dtype=torch.float32)
        # Ensure shape (C, W)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, W)
        elif x.dim() == 2:
            pass  # already (C, W)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)  # (C, W, 1) -> (C, W)
        else:
            raise ValueError(f"peak.data must be 1D or 2D, got shape {tuple(x.shape)}")

        if self.include_marker:
            marker_idx = MARKER_TO_IDX.get(peak.get_marker_name(), len(MARKER_TO_IDX))
            marker_tensor = torch.tensor([marker_idx], dtype=torch.long)  # shape (1,)
        else:
            marker_tensor = torch.full((1,), -1, dtype=torch.long)

        return x, marker_tensor

    def get_inputs(self, peaks: Sequence[ExtractedPeak]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the inputs for multiple images as a tensor on the correct device.
        Input shape: (N,)

        Args:
            peaks: the peaks to get the inputs for

        Returns: tensor of shape (N, C, W), type torch.float32 and marker tensor of shape (N,), type torch.long

        """
        peak_data, marker_indices = zip(*(self.get_input(peak) for peak in peaks))

        peak_data_tensor = torch.stack(peak_data).to(self._device)
        marker_indices_tensor = torch.stack(marker_indices).squeeze(1).to(self._device)

        assert peak_data_tensor.dim() == 3, f"Expected peak_data_tensor to have 3 dimensions, got {peak_data_tensor.dim()}"
        assert marker_indices_tensor.dim() == 1, f"Expected marker_indices_tensor to have 1 dimension, got {marker_indices_tensor.dim()}"

        return peak_data_tensor, marker_indices_tensor


    def get_targets(self,
                    peaks: Sequence[HIDImage]) -> torch.Tensor:
        """
        Get the targets for multiple images as a tensor on the correct device.
        target shape: (N,)
        Args:
            peaks: the peaks to get the targets for

        Returns: tensor of shape (N,)
        """
        targets = []
        for peak in peaks:
            targets.append(self.label_to_idx[peak.annotation.label])

        return torch.tensor(targets).to(self._device)

    def predict_class(self, probs: torch.Tensor) -> str:
        """
        From the predicted probabilities of a single image, find the class index with the highest
        probability and translate this to the actual class label.
        """
        return self.idx_to_label[torch.argmax(probs).item()]

    def set_up_metrics(self, use_evaluation_metric: bool) -> List[Optional[torchmetrics.Metric]]:
        """
        Use multiclass accuracy as evaluation metric if desired.
        """
        if not use_evaluation_metric:
            return []
        else:
            metrics = [torchmetrics.classification.Accuracy(task="multiclass",
                                                            num_classes=len(self.labels),
                                                            average="micro"),
                       torchmetrics.classification.Precision(task="multiclass",
                                                             num_classes=len(self.labels),
                                                             average="macro",),
                       torchmetrics.classification.Recall(task="multiclass",
                                                          num_classes=len(self.labels),
                                                          average="macro",),
                       torchmetrics.classification.F1Score(task="multiclass",
                                                           num_classes=len(self.labels),
                                                           average="macro",)
                       ]
        return [metric.to(self._device) for metric in metrics]

    def update_metric(self,
                      metric: Metric,
                      logits: torch.Tensor,
                      y_true: torch.Tensor) -> None:
        probs = self.compute_probabilities(logits)
        y_pred = torch.argmax(probs, dim=1)
        metric.update(y_pred, y_true)

    @staticmethod
    def compute_probabilities(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the probabilities from the logits by taking a softmax
        """
        return torch.softmax(logits, dim=1)

    def create_predictions(self, logits: torch.Tensor, batch: Sequence[HIDImage]) -> List[Prediction]:
        return [Prediction(classification=dict(zip(self.labels, map(float, probs))))
                           # meta={"predicted_label": self.predict_class(probs)})
                    for probs in self.compute_probabilities(logits)]

