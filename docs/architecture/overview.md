# Architecture Overview

DNANet is organized into six packages, each with a clear responsibility:

```
dnanet/
├── core/          Domain models (alleles, markers, panels, annotations)
├── data/          Data loading, parsing, preprocessing, strategies
├── models/        Neural network architectures (stateless nn.Module)
├── modules/       PyTorch Lightning modules (training logic)
├── evaluation/    Metrics, allele calling, visualization
└── tasks/         CLI task runners (train, evaluate, cross-validate)
```

## Package Dependencies

The dependency graph flows strictly downward — no circular imports:

```
tasks/
  ↓
modules/     evaluation/
  ↓              ↓
models/      data/
  ↓           ↓
core/       core/
```

- **`core`** depends on nothing (pure domain models + stdlib)
- **`data`** depends on `core` + NumPy/SciPy
- **`models`** depends on `core` + PyTorch
- **`modules`** depends on `models` + Lightning
- **`evaluation`** depends on `core` + `data` + NumPy
- **`tasks`** depends on everything (orchestration layer)

## Key Design Decisions

### 1. Separation of Architecture from Training

Neural network architectures (`models/`) are pure `nn.Module` classes with
no training logic. Training loops, optimizers, metrics, and logging live
in Lightning modules (`modules/`). This allows:

- Reusing the same UNet architecture for segmentation or reconstruction
- Unit-testing architectures without Lightning overhead
- Swapping training strategies without touching model code

### 2. Strategy Pattern for Kit & Dataset Variance

Forensic DNA kits (PPF6C, GlobalFiler) and datasets (NFI R&D, ProvedIt) have
different conventions for size standards, file naming, and annotations. Rather
than scattering `if kit == "PPF6C"` conditions, each variant is a Strategy
class registered in the `StrategyRegistry`.

### 3. Hydra Composition over Monolithic Config

Configuration is split into independent groups (data, model, training,
evaluation, logging) that are composed at runtime. This avoids the
combinatorial explosion of maintaining separate config files for every
experiment combination.

### 4. Lazy Loading for HIDImage

HID files are parsed on first access to the `.data` property, not at
construction time. This means scanning a directory of 800 files is instant;
the I/O cost is paid only when data is actually needed.

## Module Map

| Module | Lines | Key Classes | Design Patterns |
|--------|-------|-------------|-----------------|
| `core/allele.py` | ~50 | `Allele` | Value Object |
| `core/marker.py` | ~80 | `Marker` | Value Object |
| `core/panel.py` | ~200 | `Panel` | Factory Method (`from_xml`) |
| `core/annotation.py` | ~60 | `Annotation` | Null Object |
| `data/image.py` | ~250 | `HIDImage` | Lazy Loading (Virtual Proxy) |
| `data/dataset.py` | ~150 | `InMemoryDataset`, `SimpleDataset` | Template Method |
| `data/hid_dataset.py` | ~440 | `HIDDataset` | Facade |
| `data/datamodule.py` | ~150 | `DNANetDataModule`, `HIDTorchDataset` | Adapter, Bridge |
| `data/strategies/scaling.py` | ~460 | `ScalingStrategy`, `PPF6CStrategy`, `GlobalFilerStrategy` | Strategy, Template Method |
| `data/strategies/registry.py` | ~100 | `StrategyRegistry` | Service Locator |
| `models/unet.py` | ~160 | `UNet`, `EncoderBlock`, `DecoderBlock` | Composite |
| `models/autoencoder.py` | ~300 | `Conv1dAutoencoder`, `FourierAutoencoder` | — |
| `models/peaknet.py` | ~350 | `CombinedClassifier`, `FiLMCombiner` | Strategy (combiner) |
| `modules/segmentation.py` | ~120 | `SegmentationModule` | Mediator, Template Method |
| `evaluation/allele_caller.py` | ~150 | `AlleleCaller`, `NearestBasePairCaller` | Strategy |
| `tasks/train.py` | ~280 | `run()`, `run_with_data()` | Facade, Command |
| `cli.py` | ~60 | `main()` | Command |
