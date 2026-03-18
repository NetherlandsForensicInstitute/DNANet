import logging
from typing import Optional, Tuple, Sequence, List

import numpy as np
import torch
import torchmetrics
from torch import Tensor

from torchmetrics import Metric

from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.preprocessing.preprocess_item import RFU_MAX_VALUE, inverse_scale_data, preprocess_profile_torch
from DNAnet.models.base_model import BaseModel
from DNAnet.models.prediction import Prediction
from DNAnet.models.reconstruction.autoencoder_architectures import Conv1dAutoencoder, PerDyeConv1dAutoencoder, \
    SharedWeightPerDyeConv1dAutoencoder
from DNAnet.models.reconstruction.autoencoder_baselines import FourierAutoEncoder

LOGGER = logging.getLogger('dnanet')


class Autoencoder(BaseModel):


    def __init__(self,
                 device: Optional[str] = None,
                 input_dyes: int = 6,
                 signal_length: int = 4096,
                 preprocessing_max_rfu_scale_value: Optional[int] = RFU_MAX_VALUE,
                 preprocessing_smooth_keep_factor: Optional[float] = None,
                 preprocessing_log_scale: bool = True,
                 architecture: str = "cnn_per_dye",
                 compression: int = 8,
                 hidden_dims: int = 32,
                 kernel_size: int = 7,
                 use_sigmoid: bool = True,
                 use_batchnorm: bool = False,
                 depth: int = 5,
                 ):

        loss = torch.nn.MSELoss()

        self.preprocessing_log_scale = preprocessing_log_scale
        self.preprocessing_max_rfu_scale_value = preprocessing_max_rfu_scale_value
        self.preprocessing_num_dyes_included = input_dyes
        self.preprocessing_smooth_keep_factor = preprocessing_smooth_keep_factor
        self.architecture = architecture
        # depth = np.sqrt(compression).astype(int)


        if architecture == "cnn":
            model = Conv1dAutoencoder(device,
                                      in_channels=input_dyes,
                                      hidden_channels=hidden_dims,
                                      kernel_size=kernel_size,
                                      depth=depth,
                                      input_length=signal_length,
                                      compression=compression)
        elif architecture == "cnn_per_dye":
            model = PerDyeConv1dAutoencoder(device=device,
                                            in_channels=input_dyes,
                                            hidden_channels=hidden_dims,
                                            depth=depth,
                                            compression=compression,
                                            input_length=signal_length,
                                            kernel_size=kernel_size,
                                            use_sigmoid=use_sigmoid,
                                            use_batchnorm=use_batchnorm)
        elif architecture == "cnn_shared_per_dye":
            model = SharedWeightPerDyeConv1dAutoencoder(device=device,
                                                        in_channels=input_dyes,
                                                        hidden_channels=hidden_dims,
                                                        depth=depth,
                                                        compression=compression,
                                                        input_length=signal_length,
                                                        kernel_size=kernel_size,
                                                        use_sigmoid=use_sigmoid,
                                                        use_batchnorm=use_batchnorm)
        elif architecture == "fourier":
            latent_coeffs = signal_length // (2 * compression)
            model = FourierAutoEncoder(device, input_dyes, signal_length, latent_coeffs)
            self.fit = self._not_trainable
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        super().__init__(model, loss, device, apply_allele_caller=False)

        self.encoded_shape = self._model.encoded_shape()


    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return inverse_scale_data(
            tensor,
            log_scale=self.preprocessing_log_scale,
            max_rfu_scale_value=self.preprocessing_max_rfu_scale_value,
        )

    def _not_trainable(self, *args, **kwargs):
        raise ValueError(f"This model ({self.architecture}) is not trainable. Please use the epoch function to compute loss and metrics.")

    def get_input(self, image: HIDImage) -> torch.Tensor:
        data = preprocess_profile_torch(image,
                                        log_scale=self.preprocessing_log_scale,
                                        max_rfu_scale_value=self.preprocessing_max_rfu_scale_value,
                                        num_dyes_included=self.preprocessing_num_dyes_included)
        return data.to(self._device)

    def get_targets(self, images: Sequence[HIDImage]) -> torch.Tensor:
        """
        Returns the targets for the autoencoder, which are the same as the input images.
        """
        return (
        torch.stack(
            [torch.tensor(img.data, dtype=torch.float32, device=self._device) for img in images],
            dim=0
        ).squeeze(-1))


    def step(self,
             batch,
             metrics=None) -> torch.Tensor:
        # 1. Get preprocessed inputs and "targets" (autoencoder targets = inputs)
        inputs = self.get_inputs(batch)           # preprocessed
        y_true = self.get_targets(batch)

        # print(f"Step: inputs shape: {inputs.shape}, y_true shape: {y_true.shape}")

        # 2. Forward pass in preprocessed space
        pred_pre = self._model(inputs)

        # 3. De-normalize to original RFU space
        pred_orig = self.denormalize(pred_pre)


        # 4. Make sure shapes match what the loss expects
        if pred_orig.dim() == 4:
            pred_orig = pred_orig.squeeze(-1)


        # 5. Update metrics (in original space)
        if metrics:
            for metric in metrics:
                # For regression metrics we pass flattened vectors
                metric.update(pred_orig.view(-1), y_true.view(-1))

        # print(f"Step: pred_orig shape: {pred_orig.shape}, y_true shape: {y_true.shape}")

        # 6. Compute loss in original space
        loss = self.loss_fn(pred_orig, y_true)

        # 7. NaN guard (after sanitization, this should normally not trigger)
        if torch.isnan(pred_orig).any() or torch.isnan(y_true).any():
            raise ValueError("NaN values found in original-space tensors.")

        return loss

    def set_up_metrics(self, use_evaluation_metric: bool) -> List[Optional[Metric]]:
        metrics = [torchmetrics.regression.MeanSquaredError()]
        return [metric.to(self._device) for metric in metrics]

    def update_metric(self, metric: Metric, logits: Tensor, y_true: Tensor):
        reconstruction = self.denormalize(logits)
        metric.update(reconstruction, y_true)


    def create_predictions(self, logits: torch.Tensor, batch: Sequence[HIDImage]) -> List[Prediction]:
        predictions = []
        reconstructions = self.denormalize(logits)
        for profile in reconstructions:
            pred_image = profile.cpu().numpy()

            prediction = Prediction(image=pred_image)
            predictions.append([prediction])

        return predictions
