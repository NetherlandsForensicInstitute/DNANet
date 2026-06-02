# PeakNet Module

The `dnanet.modules.peaknet` module provides the Lightning module for combined
PeakNet training.

## PeakNetModule

PyTorch Lightning module for combined PeakNet. Handles forward pass with
dual-input (full images + peak windows), per-position cross-entropy loss,
multi-class metric tracking, and optimizer + LR scheduler configuration.

```python
from dnanet.modules.peaknet import PeakNetModule
```

**Design pattern: Mediator** — Coordinates model, loss, optimizer, scheduler,
and metrics for the combined PeakNet training scenario.

**Key difference from `ClassificationModule`:** PeakNet operates on
**per-scan-point** classification of full profiles, not on individual peak
windows. The model receives both full images and extracted peak windows, and
produces logits of shape `(N, K, C, L)`.

**Constructor args:**
- `model` — Combined classifier (`CombinedClassifier` or `PeakOnlyClassifier`)
- `loss_fn` — Loss function (e.g. `nn.CrossEntropyLoss`)
- `optimizer` — Optimizer instance
- `num_classes` — Number of output classes (default: 2)
- `learning_rate` — Initial learning rate (default: 1e-4)
- `weight_decay` — L2 regularization (default: 5e-4)
- `allele_class_index` — Index of the allele output class (default: 1)
- `lr_scheduler` — Optional learning-rate scheduler
- `metrics` — Metric collection for train/validation logging
- `batch_size` — Batch size for logging

**Batch format:** Accepts either:
- Nested: `((full_images, peak_windows, marker_idxs, peak_centers, peak_counts), targets)`
- Flat 6-item tuple: `(full_images, peak_windows, marker_idxs, peak_centers, peak_counts, targets)`
- 7-item metadata-augmented batch
- 5-item input-only batch (for inference)

**Key methods:**
- `compute_step_outputs(batch)` — Compute per-position loss and metric inputs
- `compute_test_step_outputs(batch)` — Also returns allele probabilities
- `compute_validation_step_outputs(batch)` — Alias for test step outputs
- `predict_step(batch, batch_idx)` — Return softmax probabilities
- `_split_batch(batch, require_targets)` — Parse various batch formats
- `_allele_probabilities(logits)` — Extract allele class probabilities
