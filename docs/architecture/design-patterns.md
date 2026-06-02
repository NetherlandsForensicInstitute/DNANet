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
- Baseline estimation → `baseline_superior`, `baseline_classic`, `baseline_enhanced`
- PeakNet combiner → `MLPCombiner`, `FiLMCombiner`, `CrossAttentionCombiner`

## Dependency Injection

**Where:** `HIDDataset`, `HIDImage`, data transformers

**Why:** Shared helpers need kit-specific and dataset-specific behavior such
as marker mappings, annotation class names, and split rules. These strategies
are passed through constructors and function arguments so the active dataset is
explicit at each call site.

```python
from dnanet.data.strategies import NFIRnDStrategy, PowerPlexFusion6CStrategy

dataset_strategy = NFIRnDStrategy()
scaling_strategy = PowerPlexFusion6CStrategy()
```

## Lazy Loading (Virtual Proxy)

**Where:** `HIDImage.data` property

**Why:** Creating an `HIDImage` object is cheap (just stores a path). The
expensive HID parsing, size-standard validation, and rescaling only happen
when `.data` is first accessed. This allows scanning hundreds of files without
loading any data.

```python
image = HIDImage(
    path="sample.hid",
    scaling_strategy=scaling,
    dataset_strategy=dataset_strategy,
)  # Instant: no I/O
shape = image.data.shape              # First access: triggers full load
shape2 = image.data.shape             # Cached: no re-load
```

## Template Method

**Where:** Lightning modules (`training_step`, `validation_step`, etc.),
`DatasetStrategy.split()`, `ScalingStrategy.interpolate()`

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

**Where:** `DNANetDataModule`

**Why:** DNANet's domain model (`HIDImage`, `TransformableDataset`) and
PyTorch/Lightning (`Dataset`, `DataModule`, `DataLoader`) are independent
hierarchies. The data module bridges them by applying the dataset split,
transformer, collate function, and `DataLoader` construction:

```python
datamodule = DNANetDataModule(dataset, batch_size=16)
datamodule.setup("fit")
train_loader = datamodule.train_dataloader()
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

**Where:** `Panel.from_xml()`, `Ladder.from_hid_data()`

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
