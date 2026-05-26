# Lightning Modules

```{eval-rst}
.. automodule:: dnanet.modules
```

Lightning modules wrap model architectures with training logic. Each module
handles loss computation, metric logging, optimizer configuration, and
learning rate scheduling.

## SegmentationModule

```{eval-rst}
.. autoclass:: dnanet.modules.segmentation.SegmentationModule
   :members:
   :show-inheritance:
```

Binary segmentation of EPG signals. Uses Dice loss by default and logs
Accuracy, Precision, Recall, F1, and IoU metrics.

**Constructor args:**
- `model` — Any `nn.Module` producing `(B, 1, D, L)` output
- `loss_fn` — Loss function (default: DiceLoss)
- `learning_rate` — Initial learning rate
- `weight_decay` — L2 regularization
- `scheduler_gamma` — LR decay per epoch
- `threshold` — Prediction threshold (default: 0.5)

## ClassificationModule

```{eval-rst}
.. autoclass:: dnanet.modules.classification.ClassificationModule
   :members:
   :show-inheritance:
```

Multi-class peak classification. Uses CrossEntropy or Focal loss.

## ReconstructionModule

```{eval-rst}
.. autoclass:: dnanet.modules.reconstruction.ReconstructionModule
   :members:
   :show-inheritance:
```

Autoencoder reconstruction. Uses MSE loss and logs reconstruction error.
