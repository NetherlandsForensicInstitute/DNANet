# Lightning Modules

Lightning modules wrap model architectures with training logic. Each module
handles loss computation, metric logging, optimizer configuration, and
learning rate scheduling.

## SegmentationModule

Binary segmentation of EPG signals. Uses Dice loss by default and logs
Accuracy, Precision, Recall, F1, and IoU metrics.

```python
from dnanet.modules.segmentation import SegmentationModule
```

**Constructor args:**
- `model` — Any `nn.Module` producing `(B, 1, D, L)` output
- `loss_fn` — Loss function (default: DiceLoss)
- `learning_rate` — Initial learning rate
- `weight_decay` — L2 regularization
- `scheduler_gamma` — LR decay per epoch
- `threshold` — Prediction threshold (default: 0.5)

## ClassificationModule

Multi-class peak classification. Uses CrossEntropy or Focal loss.

```python
from dnanet.modules.classification import ClassificationModule
```

## ReconstructionModule

Autoencoder reconstruction. Uses MSE loss and logs reconstruction error.

```python
from dnanet.modules.reconstruction import ReconstructionModule
```
