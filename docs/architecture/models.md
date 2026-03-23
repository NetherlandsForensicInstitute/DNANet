# Model Architectures

All model architectures are pure `nn.Module` classes in `dnanet/models/`.
They contain no training logic — that lives in Lightning modules
(`dnanet/modules/`).

## U-Net (`dnanet.models.unet.UNet`)

The primary segmentation model. Produces a binary mask predicting which
scan points belong to allele peaks.

**Input:** `(batch, 1, num_dyes, signal_length)` = `(B, 1, 5, 4096)`
**Output:** `(batch, 1, num_dyes, signal_length)` — sigmoid probabilities

**Key design choice:** Pooling and upsampling operate only along the
signal-length axis (width), not along the dye axis (height). This preserves
the dye-channel dimension while capturing multi-scale signal patterns.

### Architecture

```
Input (1, 5, 4096)
  │
  ├── Encoder Block 1: DoubleConv(1→32) + MaxPool(1,2)    → (32, 5, 2048)
  ├── Encoder Block 2: DoubleConv(32→64) + MaxPool(1,2)   → (64, 5, 1024)
  ├── Encoder Block 3: DoubleConv(64→128) + MaxPool(1,2)  → (128, 5, 512)
  ├── Encoder Block 4: DoubleConv(128→256) + MaxPool(1,2) → (256, 5, 256)
  │
  ├── Bottleneck: DoubleConv(256→512)                      → (512, 5, 256)
  │
  ├── Decoder Block 4: Upsample + Cat + DoubleConv(512→256) → (256, 5, 512)
  ├── Decoder Block 3: Upsample + Cat + DoubleConv(256→128) → (128, 5, 1024)
  ├── Decoder Block 2: Upsample + Cat + DoubleConv(128→64)  → (64, 5, 2048)
  ├── Decoder Block 1: Upsample + Cat + DoubleConv(64→32)   → (32, 5, 4096)
  │
  └── Final Conv: 1x1 conv(32→1) + Sigmoid                → (1, 5, 4096)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth` | 4 | Number of encoder/decoder levels |
| `kernel_size` | `(3, 5)` | Convolution kernel (height, width) |
| `num_filters` | 32 | Base filter count (doubles per level) |

## Autoencoders (`dnanet.models.autoencoder`)

Reconstruction models that learn compressed representations of EPG profiles.

### Conv1dAutoencoder

Standard 1D convolutional autoencoder. Flattens the dye×signal input and
processes as a single 1D sequence.

**Input/Output:** `(batch, 1, num_dyes, signal_length)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_dyes` | 5 | Number of dye channels |
| `signal_length` | 4096 | Scan points per dye |
| `depth` | 5 | Encoder/decoder depth |
| `compression` | 16 | Bottleneck compression ratio |

### PerDyeConv1dAutoencoder

Processes each dye channel independently with separate encoder/decoder pairs,
then concatenates.

### SharedWeightPerDyeConv1dAutoencoder

Like PerDyeConv1dAutoencoder but shares weights across all dye channels.

### FourierAutoencoder

Operates in the frequency domain. Applies FFT before encoding and IFFT after
decoding.

## Peak Classifier (`dnanet.models.peak_classifier`)

Multi-class classifier for individual peaks.

**Input:** Fixed-size peak windows
**Output:** Class probabilities per peak

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | 100 | Peak window size |
| `hidden_size` | 64 | Hidden layer size |
| `num_classes` | 5 | Number of allele classes |
| `num_layers` | 3 | MLP depth |

## Combined Classifier (`dnanet.models.peaknet`)

Combines segmentation and classification outputs using a learned combiner.

### Combiner Types

- **MLP** — Simple concatenation + feedforward network
- **FiLM** — Feature-wise Linear Modulation (classification features
  modulate segmentation features via learned scale/shift)
- **CrossAttention** — Multi-head cross-attention between the two feature
  streams

## Loss Functions (`dnanet.models.loss`)

### DiceLoss

Dice coefficient loss for binary segmentation. Handles class imbalance
naturally (allele peaks are sparse in the signal).

```python
loss = 1 - (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)
```

### FocalLoss

Focuses training on hard examples by down-weighting well-classified samples.
Used for classification tasks.

```python
loss = -α * (1 - p_t)^γ * log(p_t)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.25 | Balancing factor |
| `gamma` | 2.0 | Focusing parameter |
