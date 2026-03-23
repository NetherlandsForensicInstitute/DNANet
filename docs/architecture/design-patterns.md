# Design Patterns

DNANet makes deliberate use of classic software design patterns. This page
documents where and why each pattern is applied.

## Strategy Pattern

**Where:** Scaling strategies, dataset strategies, allele callers, baseline
estimation, combiner types.

**Why:** Forensic DNA analysis varies significantly between kits and datasets.
The Strategy pattern encapsulates this variance behind a common interface,
eliminating conditional branching throughout the codebase.

```python
# Each kit provides its own size-standard parsing
class ScalingStrategy(ABC):
    @abstractmethod
    def parse_size_standard(self, lane: np.ndarray) -> SizeStandardParseResult: ...

class PowerPlexFusion6CStrategy(ScalingStrategy):
    def parse_size_standard(self, lane):
        # WEN ILS: find 19 peaks, validate pixel/bp ratio
        ...

class GlobalFilerStrategy(ScalingStrategy):
    def parse_size_standard(self, lane):
        # GeneScan 600 LIZ: iterative shrinking polynomial fit
        ...
```

**Instances:**
- `ScalingStrategy` → `PowerPlexFusion6CStrategy`, `GlobalFilerStrategy`
- `DatasetStrategy` → `NFIRnDStrategy`, `ProvedItStrategy`
- `AlleleCaller` → `NearestBasePairCaller`
- Baseline estimation → `superior_baseline`, `classic_baseline`, `enhanced_baseline`
- PeakNet combiner → `MLPCombiner`, `FiLMCombiner`, `CrossAttentionCombiner`

## Service Locator (Registry)

**Where:** `StrategyRegistry`

**Why:** Some deep call chains (e.g., inside `Panel.fill_allele_bins`) need
access to the active scaling strategy. Threading a context parameter through
every function is impractical. The registry is a pragmatic compromise:
explicit configuration at startup, global read-only access at runtime.

```python
# Configured once at startup
StrategyRegistry.configure_kit("PPF6C")
StrategyRegistry.configure_dataset("NFI_RND")

# Read by any component that needs kit-specific behavior
scaling = StrategyRegistry.get_scaling_strategy()
dataset = StrategyRegistry.get_dataset_strategy()
```

## Lazy Loading (Virtual Proxy)

**Where:** `HIDImage.data` property

**Why:** Creating an `HIDImage` object is cheap (just stores a path). The
expensive HID parsing, size-standard validation, and rescaling only happen
when `.data` is first accessed. This allows scanning hundreds of files without
loading any data.

```python
image = HIDImage(path="sample.hid")  # Instant: no I/O
shape = image.data.shape              # First access: triggers full load
shape2 = image.data.shape             # Cached: no re-load
```

## Template Method

**Where:** Lightning modules (`training_step`, `validation_step`, etc.),
`InMemoryDataset.split()`, `ScalingStrategy.interpolate()`

**Why:** The *skeleton* of the algorithm is fixed (e.g., Lightning's training
loop calls `training_step` → compute loss → backprop). Subclasses override
the *variable parts* (what loss to use, how to log metrics) while inheriting
the invariant structure.

```python
class SegmentationModule(L.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss)
        return loss
```

## Facade

**Where:** `tasks/train.py`, `tasks/evaluate.py`, `tasks/cross_validate.py`,
`HIDDataset`

**Why:** Each task function (`run()`) is a single entry point that wires
together models, data, callbacks, loggers, and the trainer. Callers don't
need to know the internal wiring.

```python
# One function call does everything
from dnanet.tasks.train import run
trainer, module = run(cfg)
```

## Adapter / Bridge

**Where:** `DNANetDataModule`, `HIDTorchDataset`

**Why:** DNANet's domain model (`HIDImage`, `InMemoryDataset`) and
PyTorch/Lightning (`Dataset`, `DataModule`, `DataLoader`) are independent
hierarchies. The adapter bridges them:

```python
class HIDTorchDataset(Dataset):
    """Adapts list[HIDImage] → PyTorch Dataset returning (x, y) tensors."""

    def __getitem__(self, idx):
        image = self._images[idx]
        x = np.transpose(image.data, (2, 0, 1))  # (1, dyes, signal)
        y = np.transpose(image.annotation.image, (2, 0, 1))
        return torch.from_numpy(x), torch.from_numpy(y)
```

## Composite

**Where:** `UNet` (built from `DoubleConv`, `EncoderBlock`, `DecoderBlock`)

**Why:** The U-Net architecture is composed of reusable building blocks.
Each block is self-contained and independently testable. The composite
structure makes it easy to adjust depth or filter counts.

## Command

**Where:** `cli.py` task dispatch

**Why:** The CLI maps `task=train` to `train.run(cfg)`, `task=evaluate` to
`evaluate.run(cfg)`, etc. The Hydra config object encapsulates all
parameters needed to execute the command.

## Null Object

**Where:** `HIDImage.annotation` returns `None` (not an exception) when
no annotation is available.

**Why:** Consumers check `if image.annotation is not None:` rather than
wrapping every access in try/except. This simplifies code in datasets that
don't have annotations (e.g., ProvedIt without XLSX parsing).

## Factory Method

**Where:** `Panel.from_xml()`, `Ladder.from_hid_data()`,
`Prediction.for_segmentation()`

**Why:** Construction logic is complex (XML parsing, allele bin filling,
ladder peak matching). Factory methods encapsulate this complexity and
provide a clean, self-documenting API.

```python
panel = Panel.from_xml("SGPanel_PPF6C.xml", hid_dye_mapping={1:0, 2:1, ...})
```

## Composition over Inheritance (Configuration)

**Where:** Hydra config groups

**Why:** Rather than creating a class hierarchy of experiment configs, Hydra
composes independent groups (data × model × training × logging). Any
combination can be specified on the command line without creating new config
files.

```bash
# Compose any combination
dnanet data=provedit model=autoencoder training=reconstruction logging=tensorboard
```
