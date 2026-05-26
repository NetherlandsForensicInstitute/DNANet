# Model Architectures

```{eval-rst}
.. automodule:: dnanet.models
```

All architectures are stateless `nn.Module` classes. Training logic lives in
`dnanet.modules`.

## U-Net

```{eval-rst}
.. autoclass:: dnanet.models.unet.UNet
   :members:
.. autoclass:: dnanet.models.unet.DoubleConv
   :members:
.. autoclass:: dnanet.models.unet.EncoderBlock
   :members:
.. autoclass:: dnanet.models.unet.DecoderBlock
   :members:
```

## Autoencoders

```{eval-rst}
.. autoclass:: dnanet.models.autoencoder.Conv1dAutoencoder
   :members:
.. autoclass:: dnanet.models.autoencoder.PerDyeConv1dAutoencoder
   :members:
.. autoclass:: dnanet.models.autoencoder.SharedWeightPerDyeConv1dAutoencoder
   :members:
.. autoclass:: dnanet.models.autoencoder.FourierAutoencoder
   :members:
```

## Peak Classifier

```{eval-rst}
.. autoclass:: dnanet.models.peak_classifier.PeakClassificationModel
   :members:
```

## Combined Classifiers (PeakNet)

```{eval-rst}
.. autoclass:: dnanet.models.peaknet.CombinedClassifier
   :members:
.. autoclass:: dnanet.models.peaknet.PeakOnlyClassifier
   :members:
.. autoclass:: dnanet.models.peaknet.MLPCombiner
   :members:
.. autoclass:: dnanet.models.peaknet.FiLMCombiner
   :members:
.. autoclass:: dnanet.models.peaknet.CrossAttentionCombiner
   :members:
```

## Loss Functions

```{eval-rst}
.. autoclass:: dnanet.models.loss.DiceLoss
   :members:
.. autoclass:: dnanet.models.loss.FocalLoss
   :members:
```
