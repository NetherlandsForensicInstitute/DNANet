# Evaluation

Evaluation computes metrics on a trained model's predictions against
ground-truth annotations. DNANet supports both pixel-level and allele-level
metrics.

## Evaluation Pipeline

The `evaluate.run(cfg)` function:

1. Load the model from a checkpoint
2. Run predictions on the dataset (validation split or full dataset)
3. Compute configured metrics
4. Save results to `metrics.json`
5. Optionally save raw predictions as `.npy` files

## Pixel-Level Metrics

These metrics compare the predicted binary mask against the ground-truth
mask at every scan point.

### Pixel Precision
Fraction of predicted positive pixels that are truly positive.
```
precision = TP / (TP + FP)
```

### Pixel Recall
Fraction of truly positive pixels that are correctly predicted.
```
recall = TP / (TP + FN)
```

### Pixel F1 Score
Harmonic mean of precision and recall.
```
F1 = 2 × precision × recall / (precision + recall)
```

### Average Binary IoU
Intersection over Union, averaged over all samples.
```
IoU = TP / (TP + FP + FN)
```

## Allele-Level Metrics

These metrics evaluate whether the model correctly identifies individual
alleles, not just pixel positions. They require an **allele caller** to
translate pixel masks into discrete allele calls.

### Allele Calling

The `AlleleCaller` interface defines how pixel predictions are converted to
allele calls:

```python
class AlleleCaller(ABC):
    @abstractmethod
    def call_alleles(self, prediction: np.ndarray, panel: Panel, scaler: np.ndarray) -> list[Marker]:
        ...
```

**`NearestBasePairCaller`** — The default implementation:
1. Find contiguous regions above threshold in the prediction mask
2. Compute the center base-pair position of each region
3. Assign each region to the nearest allele in the panel
4. Return the set of called alleles per marker

### Allele Precision
Fraction of called alleles that are in the ground truth.

### Allele Recall
Fraction of ground-truth alleles that are correctly called.

### Allele F1 Score
Harmonic mean of allele precision and recall.

## Configuration

```yaml
# conf/evaluation/segmentation.yaml
metrics:
  - pixel_precision
  - pixel_recall
  - pixel_f1_score
  - average_binary_iou
save_predictions: false
predictions_dir: predictions
```

## Visualisation

The `evaluation.visualization` module provides EPG plotting utilities:

```python
from dnanet.evaluation.visualization import plot_profile

fig = plot_profile(
    signal,                   # (num_dyes, signal_length) EPG signal
    annotation=ann_mask,      # Optional: ground-truth annotation
    prediction=pred_mask,     # Optional: model prediction overlay
    title="Sample 1A2",
)
fig.savefig("epg.png")
```

This produces a multi-panel plot with one subplot per dye channel, showing
the fluorescence signal, ground-truth annotations, and model predictions.

## Running Evaluation

```bash
# Evaluate a checkpoint
dnanet task=evaluate \
    data=dnanet_rd model=unet \
    checkpoint=outputs/.../checkpoints/best.ckpt \
    evaluation=segmentation

# Save predictions for later analysis
dnanet task=evaluate \
    data=dnanet_rd model=unet \
    checkpoint=best.ckpt \
    evaluation.save_predictions=true
```

## Programmatic API

```python
from dnanet.tasks.evaluate import run

results = run(cfg, dataset)
# {'pixel_precision': 0.92, 'pixel_recall': 0.88, ...}
```
