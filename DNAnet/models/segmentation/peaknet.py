import logging
import os
from typing import Optional, List, Mapping, Any, Tuple, Sequence

import torch
import torchmetrics
from torch import nn
from torchmetrics import Metric

from DNAnet.data.data_models.base import Image
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.preprocessing.peak_extraction import extract_peaks_torch, extract_peak_windows
from DNAnet.data.preprocessing.peak_utils import get_peak_centers
from DNAnet.models.HIDImageBaseModel import HIDImageBaseModel
from DNAnet.models.classification.peak_classification import PeakClassification
from DNAnet.models.prediction import Prediction
from DNAnet.models.reconstruction.autoencoder import HIDAutoencoder
from DNAnet.models.segmentation.peaknet_architecture import CombinedClassifier, PeakOnlyClassifier
from DNAnet.typing import PathLike
from config_io import load_model

LOGGER = logging.getLogger('dnanet')

class PeakNet(HIDImageBaseModel):


    def __init__(self,
                 peak_classifier_checkpoint: Optional[PathLike] = None,
                 autoencoder_checkpoint: Optional[PathLike] = None,
                 threshold: int = 40,
                 hidden_dims: List[int] = [256, 128],
                 default_label: str = "noise",
                 freeze_autoencoder: bool = True,
                 combiner_strategy: str = "mlp",
                 autoencoder: Mapping[str, Any] = None,
                 peak_classifier: Mapping[str, Any] = None,
                 ):

        torch.set_float32_matmul_precision("high")

        self.hidden_dims = hidden_dims
        self.threshold = threshold

        self.freeze_autoencoder = freeze_autoencoder
        self.combiner_strategy = combiner_strategy

        print(f" peak classfier config: {peak_classifier}")

        # self.peak_classifier: DNANet_PeakClassification = load_model(peak_classifier_config) if not context \
        #     else load_model(peak_classifier_config, builder=context.MODELS)

        if peak_classifier is not None:
            self.peak_classifier: PeakClassification = load_model(peak_classifier)
        if peak_classifier_checkpoint is not None:
            self.peak_classifier.load(peak_classifier_checkpoint)
        if peak_classifier_checkpoint is None and peak_classifier is None:
            raise ValueError("Either peak_classifier or peak_classifier_checkpoint must be provided.")


        self.window_size = self.peak_classifier.window_size
        self.noise_label_idx = self.peak_classifier.label_to_idx[default_label]
        self.num_classes = self.peak_classifier.num_classes

        if autoencoder is not None or autoencoder_checkpoint is not None:
            if autoencoder is not None and autoencoder_checkpoint is not None:
                self.autoencoder: HIDAutoencoder = load_model(autoencoder)
                self.autoencoder.load(autoencoder_checkpoint)
                LOGGER.info(f"Loaded autoencoder from {autoencoder_checkpoint}")
            elif autoencoder is not None:
                self.autoencoder: HIDAutoencoder = load_model(autoencoder)
                LOGGER.info(f"Loaded autoencoder from config")
            elif autoencoder_checkpoint is not None:
                self.autoencoder: HIDAutoencoder = load_model(autoencoder_checkpoint)
                LOGGER.info(f"Loaded autoencoder from {autoencoder_checkpoint}")

            autoencoder_out_shape = self.autoencoder.encoded_shape

            model = CombinedClassifier(
                autoencoder=self.autoencoder.model,
                autoencoder_out_shape=autoencoder_out_shape,
                peak_classifier=self.peak_classifier.model,
                peak_classifier_out_shape=self.peak_classifier.model.backbone_out_shape(),
                hidden_dims=self.hidden_dims,
                num_classes=self.peak_classifier.num_classes,
                default_class=self.noise_label_idx,
                freeze_autoencoder=self.freeze_autoencoder,
                combiner_strategy=self.combiner_strategy,
            )
        else:
            self.autoencoder = None
            model = PeakOnlyClassifier(
                peak_classifier=self.peak_classifier.model,
                num_classes=self.peak_classifier.num_classes,
                default_class=self.noise_label_idx,
                out_channels=6, # assume output size when no autoencoder is used
                out_length=4096,
            )


        loss = self.peak_classifier.loss_fn

        super().__init__(model, loss)


    def get_input(self, image: HIDImage, use_torch: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the input tensors for the combined model from the image.

        image tensors has shape (C, 4096) and type torch.float32
        peak tensors has shape  ((N_p, C, W), (N_p,)) and type float32
        marker idx has shape (N_p,) and type torch.long
        peak centers has shape (N_p, 2) and type torch.long

        All tensors are moved to self._device.

        Args:
            image: Image object.
            use_torch: whether to use the torch implementation of peak extraction. If False, uses the numpy implementation.

        Returns: A tuple of (image_tensor, peak_tensors, marker_idx, peak_centers)
        """
        if use_torch:
            peak_tensors, marker_idx, peak_centers = extract_peaks_torch(image,
                                                                         self._device,
                                                                         self.threshold,
                                                                         self.window_size,
                                                                         self.peak_classifier.include_max_pool_dyes)
        else:
            peaks = extract_peak_windows(image,
                                         threshold=self.threshold,
                                         window_size=self.window_size,
                                         use_ground_truth_labels=False,
                                         include_raw_annotation=True,
                                         include_max_pool_dyes=self.peak_classifier.include_max_pool_dyes) # (N_p,)

            peak_tensors, marker_idx = self.peak_classifier.get_inputs(peaks) # peak_tensors: ((N_p, C, W), marker_idx: (N_p,))
            peak_centers = get_peak_centers(peaks).to(self._device) # (N_p, 2); index 0: dye index, index 1: center position


        if self.autoencoder is not None:
            image_tensor = self.autoencoder.get_input(image) # (C, 4096, 1)
            image_tensor = image_tensor.squeeze(-1)  # (C, 4096)
        else:
            image_tensor = None

        return image_tensor, peak_tensors, marker_idx, peak_centers


    def get_inputs(self, images: Sequence[HIDImage]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the input tensors for multiple images as a tuple of tensors.

        peak_tensors, marker_idx, peak_centers are nested tensors because the number of peaks per image can vary.
        The N_p dimension is ragged-shaped, it varies for every N.

        Args:
            images: Sequence of HIDImage objects.

        Returns: A tuple of (image_tensors, peak_tensors_list, marker_idx_list, peak_centers_list)
            - image_tensors: Tensor of shape (N, C, 4096) and type torch.float32
            - peak_tensors_list: NestedTensor of shape (N, N_p, C, W) and type torch.float32
            - marker_idx_list: NestedTensor of shape (N, N_p) and type torch.long
            - peak_centers_list: NestedTensor of shape (N, N_p, 2) and type torch.long

        """
        image_tensors = []
        peak_tensors_list = []
        marker_idx_list = []
        peak_centers_list = []
        for image in images:
            image_tensor, peak_window, marker_idx, peak_center = self.get_input(image)
            # image tensor: (C, 4096)
            # peak_window: (N_p, C, W),
            # marker_idx: (N_p,)
            # peak_centers: (N_p, 2)
            if image_tensor is not None: # image tensor is not required when no autoencoder is used
                image_tensors.append(image_tensor)
            peak_tensors_list.append(peak_window)
            marker_idx_list.append(marker_idx)
            peak_centers_list.append(peak_center)

        if self.autoencoder is not None:
            image_tensors = torch.stack(image_tensors)  # (N, C, 4096)

        # We use nested tensors because N_p is ragged-shaped: the number of peaks per image can vary
        peak_tensors_list = torch.nested.nested_tensor(peak_tensors_list, layout=torch.jagged, device=self._device) # (N, N_p, C, W)
        # marker_idx_list = torch.nested.nested_tensor_from_jagged(values=marker_idx_list, offsets=peak_tensors_list.offsets()) # (N, N_p)

        marker_idx_list = torch.nested.nested_tensor(marker_idx_list, layout=torch.jagged, device=self._device) # (N, N_p)
        peak_centers_list = torch.nested.nested_tensor(peak_centers_list, layout=torch.jagged, device=self._device) # (N, N_p, 2)

        return image_tensors, peak_tensors_list, marker_idx_list, peak_centers_list


    def get_targets(self,
                    images: Sequence[Image]) -> torch.Tensor:

        targets = []
        for image in images:
            img = torch.from_numpy(image.annotation.image) # (C, 4096, 1)
            targets.append(img.squeeze(-1)) # (C, 4096)
        return torch.stack(targets).to(device=self._device, dtype=torch.long) # (N, C, 4096)

    def set_up_metrics(self, use_evaluation_metric: bool) -> List[Optional[Metric]]:
        if not use_evaluation_metric:
            return []

        acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average="micro").to(self._device)
        f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro").to(self._device)

        return [f1, acc]

    def update_metric(self,
                      metric: Metric,
                      logits: torch.Tensor,
                      y_true: torch.Tensor):

        # logits: (N, K, C, L)
        # y_true: (N, C, L)
        if logits.ndim != 4:
            raise ValueError(f"logits must be (N, K, C, L), got shape {tuple(logits.shape)}")
        if y_true.ndim != 3:
            raise ValueError(f"y_true must be (N, C, L), got shape {tuple(y_true.shape)}")
        if logits.shape[0] != y_true.shape[0] or logits.shape[2:] != y_true.shape[1:]:
            raise ValueError(f"Batch/space dims mismatch: logits {tuple(logits.shape)} vs targets {tuple(y_true.shape)}")

        # ----- Argmax over class dimension to get predicted labels -----
        # preds: (N, C, L) integer class indices
        preds = logits.argmax(dim=1)  # (N, K, C, L) -> (N, C, L)

        # ----- Flatten for torchmetrics -----
        # preds_flat, targets_flat: (N*C*L,)
        preds_flat = preds.reshape(-1)
        targets_flat = y_true.reshape(-1)

        metric.update(preds_flat, targets_flat)


    def create_predictions(self, logits: torch.Tensor, batch: Sequence[HIDImage]) -> List[Prediction]:
        ## TODO: support for multiclass classification
        probs = torch.softmax(logits, dim=1) # (N, num_classes, C, 4096)
        y_pred = probs[:, 1, :, :]  # Take the probabilities for the 'allele' class (index 1) (N, C, 4096)
        y_pred = y_pred.unsqueeze(-1)  # (N, C, 4096, 1)

        predictions = []
        for pred_image in y_pred:
            pred_image_np = pred_image.cpu().numpy()
            predictions.append(Prediction(image=pred_image_np))

        return predictions

    def save(self, model_dir: PathLike):
        os.makedirs(model_dir, exist_ok=True)

        if self.autoencoder is not None:
            # 2) Save submodels
            ae_dir = os.path.join(model_dir, "autoencoder")
            self.autoencoder.save(ae_dir)

            combiner_path = os.path.join(model_dir, "combiner.pt")
            torch.save(self._model.combiner.state_dict(), combiner_path)

        pc_dir = os.path.join(model_dir, "peak_classifier")
        self.peak_classifier.save(pc_dir)


    def load(self, model_dir: PathLike):
        # 1) Load submodels first (so their modules are in place)
        if self.autoencoder is not None:
            ae_dir = os.path.join(model_dir, "autoencoder")
            self.autoencoder.load(ae_dir)


            combiner_path = os.path.join(model_dir, "combiner.pt")
            state = torch.load(combiner_path, map_location=self._device)
            self._model.combiner.load_state_dict(state)

        pc_dir = os.path.join(model_dir, "peak_classifier")

        if os.path.isdir(pc_dir):
            self.peak_classifier.load(pc_dir)




