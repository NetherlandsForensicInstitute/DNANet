# Model Architectures

All architectures are stateless `nn.Module` classes. Training logic lives in
`dnanet.modules`.

## U-Net

```python
from dnanet.models.unet import UNet, DoubleConv, EncoderBlock, DecoderBlock
```

## Autoencoders

```python
from dnanet.models.autoencoder import (
    Conv1dAutoencoder,
    PerDyeConv1dAutoencoder,
    SharedWeightPerDyeConv1dAutoencoder,
    UNet2DAutoEncoder,
    FourierAutoencoder,
)
```

## Peak Classifier

```python
from dnanet.models.peak_classifier import PeakClassificationModel
```

## Combined Classifiers (PeakNet)

```python
from dnanet.models.peaknet import (
    CombinedClassifier,
    PeakOnlyClassifier,
    MLPCombiner,
    FiLMCombiner,
    CrossAttentionCombiner,
)
```

## Loss Functions

```python
from dnanet.models.loss import DiceLoss, FocalLoss
```
