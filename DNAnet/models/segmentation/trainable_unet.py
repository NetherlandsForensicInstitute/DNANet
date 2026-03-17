from typing import Sequence, Tuple, Optional, List

import torch
import torchmetrics
from torchmetrics import Metric

from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.utils import process_image
from DNAnet.models.HIDImageBaseModel import HIDImageBaseModel
from DNAnet.models.loss import DiceLoss
from DNAnet.models.prediction import Prediction
from DNAnet.models.segmentation.unet_architecture import UNet


class DNANet_UNet(HIDImageBaseModel):
    """
    A setup for a U-Net model geared towards analysing dna profiles using PyTorch.
    """

    def __init__(self,
                 depth: int,
                 kernel_size: Tuple[int, int],
                 num_filters: int = 64,
                 device: Optional[str] = None,
                 apply_allele_caller: Optional[bool] = True):
        """
        Initialize the DNAnet UNet.

        :param depth: depth the unet should have
        :param kernel_size: (height, width) of the kernel to use in the convolutional layers
        :param num_filters: the number of initial filters in the first conv layer
        :param device: The device on which the model should run. Should be either "cpu" or
        "cuda" for CPU or GPU respectively.
        :param apply_allele_caller: Whether to call actual alleles from the predicted segmentation
        """
        model = UNet(depth, kernel_size, num_filters, device)
        loss = DiceLoss()
        super().__init__(model, loss, device, apply_allele_caller)


    def get_input(self, image: HIDImage) -> torch.Tensor:
        """
        Returns the input tensor corresponding to the ``image`` for the
        underlying PyTorch model. For a torch model, the channels of the image should be
        first.

        :param image: The image to turn into a tensor
        :return: A 3D tensor of shape `(3, height, width)`.
        """
        return torch.tensor(
            data=process_image(image.data, channels_first=True),
            dtype=torch.float32)


    def get_targets(self,
                    images: Sequence[HIDImage]) -> torch.Tensor:
        """
        Get the target for an image in the correct format
        """
        return torch.stack([
            torch.tensor(image.annotation.image).movedim(2, 0)
            for image in images
        ]).to(self._device)

    def update_metric(self,
                      metric: Metric,
                      logits: torch.Tensor,
                      y_true: torch.Tensor):
        metric.update(torch.flatten(torch.sigmoid(logits)), torch.flatten(y_true))


    def set_up_metrics(self, use_evaluation_metric: bool) -> List[Optional[Metric]]:
        """
        Use binary accuracy as evaluation metric if desired.
        """
        if not use_evaluation_metric:
            return []
        else:
            metrics = [torchmetrics.classification.BinaryAccuracy()]
        return [metric.to(self._device) for metric in metrics]


    def create_predictions(self, logits: torch.Tensor, batch: Sequence[HIDImage]) -> List[Prediction]:
        return [Prediction(
            image=torch.sigmoid(pred_im).movedim(0, -1).cpu().detach().numpy(),
            original_image_path=image.path)
            for image, pred_im in zip(batch, logits)]
