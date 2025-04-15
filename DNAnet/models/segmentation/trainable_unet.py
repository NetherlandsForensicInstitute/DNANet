import contextlib
import logging
import os
import shutil
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import mlflow
import torch
import torchmetrics
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric
from tqdm import tqdm

from DNAnet.allele_callers import NearestBasePairCaller
from DNAnet.data.data_models.hid_dataset import HIDDataset
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.utils import process_image
from DNAnet.models.base import TORCH_DEFAULT_DEVICE, TrainableModel
from DNAnet.models.loss import DiceLoss
from DNAnet.models.prediction import Prediction
from DNAnet.models.segmentation.unet_architecture import UNet
from DNAnet.typing import PathLike
from DNAnet.utils import chunks


LOGGER = logging.getLogger('dnanet')


class DNANet_UNet(TrainableModel):
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
        self._device = device or TORCH_DEFAULT_DEVICE
        model = UNet(depth, kernel_size, num_filters, self._device)
        self._model = model.to(self._device)
        self.loss_fn = DiceLoss()
        self.allele_caller = NearestBasePairCaller() if apply_allele_caller else None

    @staticmethod
    def get_input(image: HIDImage) -> torch.Tensor:
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

    def get_inputs(self, images: Sequence[HIDImage]) -> torch.Tensor:
        """
        Get the input for multiple images as a tensor on the correct device
        """
        return torch.stack([self.get_input(image) for image in images]).to(
            self._device)

    def get_targets(self,
                    images: Sequence[HIDImage]) -> torch.Tensor:
        """
        Get the target for an image in the correct format
        """
        return torch.stack([
            torch.tensor(image.annotation.image).movedim(2, 0)
            for image in images
        ]).to(self._device)

    def fit(self,
            dataset: HIDDataset,
            batch_size: int = 4,
            num_epochs: int = 10,
            learning_rate: float = 0.005,
            weight_decay: float = 0.0005,
            tensorboard: bool = False,
            validation_set: Optional[HIDDataset] = None,
            min_delta: Optional[float] = 0.01,
            steps_per_epoch: Optional[int] = None,
            use_evaluation_metric: bool = True,
            checkpoint_dir: PathLike = None,
            save_best: bool = False,
            use_scheduler: bool = False,
            **kwargs):
        """
        Fits the model on training data.
        :param dataset: The training dataset
        :param batch_size: The number of images to include in a single
                training step
        :param num_epochs: The number of times to loop over the entire
                training set.
        :param learning_rate: The initial learning rate for the optimizer.
        :param weight_decay: A regularization technique adding a small
                penalty to the weights. A higher weight decay rate means more
                regularization and less overfitting, but also less flexibility
                and more underfitting.
        :param tensorboard: Whether to write logs containing the training and
                validation loss (if a `validation_set` was specified) that can
                afterwards be visualized using TensorBoard. The logs will be
                written to the `<checkpoint_dir>/tensorboard/` directory and
                can be visualized by running the following from your terminal:
                `tensorboard --logdir=<checkpoint_dir>/tensorboard`
        :param validation_set: An optional separate dataset that we use to
                evaluate the model after each epoch.
        :param min_delta: The minimum difference in validation loss we want to see
                in five training steps. If not, we apply early stopping
                (large negative numbers will result in no early stopping).
        :param steps_per_epoch: The number of batches in each epoch.
        :param use_evaluation_metric: If True,
                `torchmetrics.classification.BinaryAccuracy` is
                computed for both the training and validation set (if any).
        :param checkpoint_dir: An optional directory where we can save model
                checkpoints and other files generated during training.
        :param save_best: Whether to keep track of the model with the best
                performance of the validation set during training. Once
                training has completed, the weights of the best model will be
                restored. This helps greatly in preventing the model from over-
                fitting. If True, both a `validation_set` and `checkpoint_dir`
                must be specified.
        :param use_scheduler: Whether to use an exponential scheduler to make the
                learning rate decrease during training.
        """
        if save_best:
            if validation_set is None:
                raise TypeError(
                    "`validation_set` is required when `save_best` is True")
            if checkpoint_dir is None:
                raise TypeError(
                    "`checkpoint_dir` is required when `save_best` is True")

        for parameter in self._model.parameters():
            parameter.requires_grad = True

        optimizer = torch.optim.Adam(self._model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        # Setup scheduler to decrease the learning rate exponentially after every epoch
        if use_scheduler:
            LOGGER.info(f"Setting up exponential scheduler, starting with learning "
                        f"rate {learning_rate}")
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        metrics = self.set_up_metrics(use_evaluation_metric)

        # Ensure `checkpoint_dir` is a `Path` for future typing.
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        # Handle `tensorboard`.
        writer = None
        if tensorboard:
            if checkpoint_dir is None:
                raise TypeError(
                    "`checkpoint_dir` is required when `tensorboard` is True")
            tensorboard_dir = checkpoint_dir / "tensorboard"
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tensorboard_dir))
            LOGGER.info(f"Tensorboard logs are written to {tensorboard_dir}")
            LOGGER.info(f"Run `tensorboard --logdir={tensorboard_dir}`")

        # Determine the number of steps per epoch if it was not specified.
        if not steps_per_epoch:
            steps_per_epoch = max(1, len(dataset) // batch_size)

        if validation_set:
            steps_per_epoch_val = len(validation_set) // batch_size
        else:
            steps_per_epoch_val = None

        prev_best = (1., 0)  # what was the previous best loss and at what epoch?
        # Used only if `save_best` is True.
        best_validation_loss = float('inf')

        for epoch in range(num_epochs):
            # Clear the previous summary and start a new one for this epoch.
            summary: Dict[str, Dict[str, float]] = defaultdict(dict)

            descr = f"Epoch {epoch + 1}/{num_epochs}"

            # Slice the training data in a number of batches
            train_batches = chunks(dataset, batch_size)
            train_batches = islice(train_batches, steps_per_epoch)
            # Train the model for a single epoch and compute the loss
            training_loss = self.epoch(batches=train_batches,
                                       # Todo add possibility for balancer
                                       steps_per_epoch=steps_per_epoch,
                                       description=descr,
                                       optimizer=optimizer,
                                       train=True,
                                       metrics=metrics)
            mlflow.log_metric(key="training_loss", value=training_loss,
                              step=epoch)

            # Update the logs we write to tensorboard.
            summary["Loss"]["training"] = training_loss
            for metric in metrics:
                result = metric.compute()
                metric_name = metric.__class__.__name__
                summary[metric_name]['training'] = result
                LOGGER.info(f"{descr} - Training {metric_name.lower()}: {result:.6f}")

            LOGGER.info(f"{descr} - Training loss: {training_loss:.6f}")

            if validation_set:
                # If a validation set is specified, apply the model to it after
                # each training epoch to keep track of a metric and the loss.
                # TODO: now only first metric is taken as validation metric
                validation_metric = self.compute_metric(metrics[0],
                                                        validation_set,
                                                        batch_size)
                validation_loss = self.compute_validation_loss(
                    validation_set, batch_size, descr,
                    steps_per_epoch_val, metrics)
                LOGGER.info(f"{descr} - Validation metric ({metrics[0]}): "
                            f"{validation_metric:.6f} | Validation loss: "
                            f"{validation_loss:.6f}")

                mlflow.log_metrics({"validation_loss": validation_loss,
                                    "validation_metric": validation_metric},
                                   step=epoch)

                # Update the logs we write to tensorboard.
                summary["Loss"]["validation"] = validation_loss
                for metric in metrics:
                    result = metric.compute()
                    metric_name = metric.__class__.__name__
                    summary[metric_name]["validation"] = result
                    LOGGER.info(
                        f"{descr} - Validation {metric_name.lower()}: {result:.6f}")

                mlflow.log_metrics({"validation_loss": validation_loss,
                                    "validation_metric": validation_metric},
                                   step=epoch)

                # If `save_best` is True, keep track of the model with the best
                # performance on the validation set so far.
                if save_best and validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    self.save(checkpoint_dir / "best_model")
                    LOGGER.info(f"Saved best model with loss "
                                f"{best_validation_loss:.6f} at epoch "
                                f"{epoch + 1}.")

                # Check for early stopping. We want to see a min_delta improvement in the last
                # 5 steps, otherwise we stop
                if validation_loss < prev_best[0] - min_delta:
                    prev_best = (validation_loss, epoch)
                elif epoch - prev_best[1] > 5:
                    LOGGER.info(f"Early stopping reached at epoch {epoch + 1}.")
                    break

            if use_scheduler:
                self.update_scheduler(descr, optimizer, scheduler)

            # Write any logs to tensorboard if possible
            if writer and summary:
                for tag, scalars in summary.items():
                    writer.add_scalars(tag, scalars, global_step=epoch + 1)

            # Close tensorboard writer if it exists
            if writer:
                LOGGER.info(f"Tensorboard logs written to {writer.log_dir}")
                LOGGER.info(f"Run `tensorboard --logdir={writer.log_dir}` to view")
                writer.close()

        # If we kept track of the model with the best performance on the
        # validation set, we want to reload the best performing model's
        # weights at the end of training.
        if save_best:
            self.load(checkpoint_dir / 'best_model')
            shutil.rmtree(checkpoint_dir / 'best_model')
            LOGGER.info("Restored previous best model")

    def set_up_metrics(self, use_evaluation_metric: bool) -> List[Optional[Metric]]:
        """
        Use binary accuracy as evaluation metric if desired.
        """
        if not use_evaluation_metric:
            return []
        else:
            metrics = [torchmetrics.classification.BinaryAccuracy()]
        return [metric.to(self._device) for metric in metrics]

    @staticmethod
    def update_scheduler(descr: str,
                         optimizer: torch.optim.Optimizer,
                         scheduler: torch.optim.lr_scheduler.LRScheduler):
        scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]
        LOGGER.info(f"{descr}: scheduler decreased learning rate to {new_lr}")

    def epoch(self,
              batches: Iterator[Sequence[HIDImage]],
              steps_per_epoch: int,
              description: str,
              optimizer: Optional[torch.optim.Optimizer] = None,
              train: bool = True,
              metrics: Sequence[Metric] = None) -> float:
        batches = tqdm(batches, desc=description, total=steps_per_epoch)

        # Depending on whether this is a training or validation epoch, put the
        # underlying PyTorch model in the proper mode (`train` or `eval`).
        if train:
            self._model.train()
            context = contextlib.nullcontext()  # type: ignore
            optimizer.zero_grad()
        else:
            self._model.eval()
            # Temporarily set all `requires_grad` flags to False
            context = torch.no_grad()  # type: ignore

        if metrics:
            for metric in metrics:
                metric.reset()

        epoch_loss = 0.
        with context:
            for step, batch in enumerate(batches):
                loss = self.step(batch, metrics)
                if train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Keep track of a running average of the loss.
                epoch_loss = (epoch_loss * step + float(loss)) / (step + 1)

                # If `batches` is actually a `tqdm` progress bar, update the
                # loss after each batch.
                if isinstance(batches, tqdm):
                    name = 'training' if train else 'validation'
                    batches.set_postfix({name + '_loss': epoch_loss})

        return epoch_loss

    def step(self,
             batch: Sequence[HIDImage],
             metrics: Optional[Sequence[Metric]]) -> torch.Tensor:
        inputs = self.get_inputs(batch)
        y_true = self.get_targets(batch)

        logits = self._model(inputs)

        if metrics:
            for metric in metrics:
                self.update_metric(metric, logits, y_true)

        return self.loss_fn(logits, y_true)

    def predict(self, image: HIDImage) -> Prediction:
        return self.predict_batch([image])[0]

    def predict_batch(self, batch: Sequence[HIDImage]) -> List[Prediction]:
        self._model.eval()

        with torch.no_grad():
            y_pred = self.predict_raw(batch)

        predictions = [Prediction(
            image=self.sigmoid(pred_im).movedim(0, -1).cpu().detach().numpy(),
            meta={'original_image_path': image.path}) for image, pred_im in zip(batch, y_pred)]

        if self.allele_caller:
            LOGGER.info("Calling alleles from predicted segmentation...")
            predictions = [self.allele_caller.call_alleles(im, pred) for im, pred in
                           zip(batch, predictions)]
        return predictions

    def predict_raw(self, images: Sequence[HIDImage]) -> torch.Tensor:
        """
        Takes a batch of `images` and returns the raw predictions as output
        directly by the underlying model. These have not yet been
        converted to `fire.models.prediction.Prediction` instances.

        :param images: Sequence[HIDImage]
        :return: Union[np.ndarray, Sequence[np.ndarray]]
        """
        return self._model(self.get_inputs(images))

    @staticmethod
    def sigmoid(logits: torch.Tensor) -> torch.Tensor:
        """
        Applies a sigmoid function to `logits`

        :param logits: The logits to apply the proper output activation to.
        :return: The confidence scores after applying the output activation.
        """
        return torch.sigmoid(logits)

    def update_metric(self,
                      metric: Metric,
                      logits: Tensor,
                      y_true: Tensor):
        metric.update(torch.flatten(self.sigmoid(logits)), torch.flatten(y_true))

    def compute_metric(self,
                       metric: Metric,
                       dataset: HIDDataset,
                       batch_size: int) -> float:
        """
        Compute a value for a provided TorchMetric and dataset.
        """
        metric.reset()
        batches = chunks(dataset, batch_size)

        for batch in batches:
            y_true = self.get_targets(batch)
            inputs = self.get_inputs(batch)
            logits = self._model(inputs)
            self.update_metric(metric, logits, y_true)
        return metric.compute()

    def compute_validation_loss(self,
                                validation_set: HIDDataset,
                                batch_size: int, description: str,
                                steps_per_epoch: int,
                                metrics: Sequence[Metric]) -> float:
        """
        Compute the loss for the validation set.
        """
        validation_batches = chunks(validation_set, batch_size)
        return self.epoch(validation_batches, steps_per_epoch=steps_per_epoch,
                          description=description, train=False,
                          metrics=metrics)

    def save(self, model_dir: PathLike):
        """
        Saves the current model's state dictionary to a specified directory.

        :param model_dir: The directory where the model's state dictionary
            should be saved
        """
        os.makedirs(model_dir, exist_ok=True)
        path = self._get_weights_path(model_dir)
        torch.save(self._model.state_dict(), path)

    def load(self, model_dir: PathLike):
        """
        Loads the model's state dictionary from a specified directory.

        :param model_dir: The directory where the model's state dictionary
            should be loaded from
        """
        path = self._get_weights_path(model_dir)
        self._model.load_state_dict(torch.load(path, self._device))

    def _get_weights_path(self, model_dir: PathLike) -> str:
        return os.path.join(model_dir,
                            f'{self.__class__.__name__}_checkpoint.pt')
