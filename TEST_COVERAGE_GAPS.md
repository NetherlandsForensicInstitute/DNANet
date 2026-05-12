# DNANet Test Coverage Gap Analysis

**Overall coverage: 78%** (677 passed, 5 skipped)

Generated: 2026-05-12

---

## Executive Summary

The codebase has **30 source files** in `src/dnanet/`. Of these:

| Category | Files | Avg Coverage | Status |
|---|---|---|---|
| **100% covered** | 10 | 100% | Healthy |
| **90-99%** | 14 | 94% | Good |
| **50-89%** | 8 | 61% | Needs work |
| **<50%** | 6 | 21% | Critical gaps |

**Total untested lines: 915 of 5,018** (22% uncovered)

---

## 100% Coverage (Healthy - No Action Needed)

| File | Lines | Notes |
|---|---|---|
| `core/allele.py` | 16 | Core allele logic fully tested |
| `core/types.py` | 4 | Type aliases |
| `data/cache/fingerprint.py` | 43 | Cache fingerprinting |
| `data/cache/layout.py` | 23 | Cache layout |
| `data/dataset.py` | 6 | Simple dataset wrapper |
| `data/strategies/scaling/size_standard.py` | 9 | Size standard constants |
| `data/preprocessing/peaks.py` | 43 | Peak utilities |
| `evaluation/metrics/allele.py` | 50 | Allele metrics |
| `evaluation/metrics/per_RFU.py` | 119 | Per-RFU metrics |
| `models/loss.py` | 24 | Loss functions |

---

## 2. Critical Gaps (0-30% coverage, core functionality)

### 2.1 `dnanet/tasks/evaluate.py` — 15% (11 untested lines)

**Test file:** None. Only a bare import check in `test_cli.py`.

This is a **core entry point** for model evaluation, parallel to `train.py`.

| Missing | Lines | Description |
|---|---|---|
| `_as_2d_array()` | 36-39 | Squeeze last dim from 3D arrays |
| `_save_results()` | 42-52 | Write results to JSON |
| `_build_callbacks()` | 55-70 | Build callback list from config |
| `run()` | 73-185 | Full evaluation pipeline |

**Recommended tests:**
1. Unit tests for `_as_2d_array()` with 3D and 2D inputs
2. Unit test for `_save_results()` with sample results dict
3. Unit test for `_build_callbacks()` with DictConfig, ListConfig, empty, and invalid types
4. Integration test for `run()` with mocked checkpoint + config file

**Priority: HIGH** — Core entry point with zero tests.

---

### 2.2 `dnanet/logging.py` — 23% (28 untested lines)

**Test file:** None.

Centralized logging configuration used at application startup.

| Missing | Lines | Description |
|---|---|---|
| `_InterceptHandler.emit()` | 30-43 | Stdlib → loguru forwarding |
| `configure()` | 55-110 | Full config with console + file sinks |
| `_intercept_third_party_loggers()` | 113-124 | Third-party logger interception |

**Recommended tests:**
1. Test `_InterceptHandler.emit()` with mock `LogRecord`
2. Test `configure()` with various verbosity levels, log_file paths, and serialize option
3. Test `_intercept_third_party_loggers()` verifies handler assignment

**Priority: MEDIUM** — Infrastructure code, but important for debugging.

---

### 2.3 `dnanet/tools/labeltool/tool.py` — 19% (169 untested lines)

**Test file:** `tests/tools/test_labeltool.py` exists but only tests CLI entry point and annotations.

Interactive GUI tool for manual annotation. Low priority for unit tests due to GUI nature.

**Recommended:** Consider skipping GUI component tests. Focus on non-GUI utility functions if any exist.

**Priority: LOW** — GUI tool, impractical to fully test.

---

### 2.4 `dnanet/tools/labeltool/visualization.py` — 28% (91 untested lines)

**Test file:** None.

Visualization logic for the label tool. GUI-dependent.

**Priority: LOW** — GUI-dependent visualization.

---

### 2.5 `dnanet/tools/labeltool/cli.py` — 27% (29 untested lines)

**Test file:** `tests/tools/test_labeltool.py` has some tests but not the CLI module.

CLI commands for the label tool.

**Priority: LOW** — UI tooling, limited automation value.

---

### 2.6 `dnanet/cli.py` — 28% (18 untested lines)

**Test file:** `tests/tasks/test_cli.py` exists but tests the main CLI entry point, not this module.

CLI argument parsing and dispatch.

**Recommended tests:**
1. Test CLI command registration and dispatch
2. Test help output
3. Test error handling for unknown commands

**Priority: MEDIUM** — Entry point for all CLI commands.

---

## 3. High Gaps (30-60% coverage, important functionality)

### 3.1 `dnanet/data/strategies/datasets/provedit.py` — 34% (97 untested lines)

**Test file:** `tests/data/test_dataset_strategies.py` — only tests `categorize_file`, `get_sample_id`, `get_number_of_contributors`.

ProvedIT data strategy for forensic DNA analysis. Large file with extensive annotation parsing.

| Untested Method | Lines | Description |
|---|---|---|
| `collect_dataset_files()` | 49-83 | Full file collection pipeline |
| `cache_signature()` | 46-47 | Cache identification |
| `get_number_of_contributors()` | 106-131 | NoC extraction with ValueError |
| `get_sample_id()` | 133-146 | Sample ID extraction |
| `_extract_contributors()` | 148-154 | Regex extraction |
| `_find_annotation_file()` | 156-169 | Annotation file discovery |
| `parse_annotations()` | 171-223 | XLSX parsing pipeline |
| `find_ladder_for_sample()` | 225-252 | Well prefix matching |
| `_combine_contributors_into_annotation()` | 254-261 | Contributor merging |
| `get_annotation_classes()` | 263-265 | Returns `['noise', 'allele']` |
| `_split()` | 267-320 | Fractional + k-fold dispatch |
| `_fractional_split()` | 322-362 | 3-way data split |
| `_kfold_split()` | 364-387 | StratifiedKFold/KFold |

**Recommended tests:**
1. `cache_signature()` — verify return dict
2. `_extract_contributors()` — valid stem, invalid stem (ValueError)
3. `_find_annotation_file()` — 0 files (FileNotFoundError), 1 file, multiple (RuntimeError)
4. `parse_annotations()` — non-XLSX (ValueError), XLSX with N/A markers, multiple_research_ids=True
5. `find_ladder_for_sample()` — with/without ladder_mapping, well prefix match, fallback
6. `_combine_contributors_into_annotation()` — valid, no contributors (ValueError)
7. `get_annotation_classes()` — verify `['noise', 'allele']`
8. `_split()` — fractional, k-fold, invalid args
9. `_fractional_split()` — 3-way with test_fraction, 2-way without
10. `_kfold_split()` — stratified and non-stratified paths

**Priority: HIGH** — Complex data pipeline with 97 untested lines.

---

### 3.2 `dnanet/modules/base.py` — 56% (35 untested lines)

**Test file:** None.

Base class for all task modules (segmentation, classification, reconstruction, peaknet).

| Untested Method | Lines | Description |
|---|---|---|
| `compute_test_step_outputs()` | 59-65 | Default returns None callback_preds |
| `_metrics_for_stage()` | 67-74 | "test" stage + ValueError |
| `_shared_step()` | 76-79 | Calls compute_step_outputs |
| `_log_step_outputs()` | 81-109 | Empty metrics skip + logging path |
| `test_step()` | 119-125 | None callback_preds path |
| `transfer_batch_to_device()` | 127-135 | Metadata batch handling |
| `_is_metadata_batch()` | 137-143 | Batch type detection |
| `_is_metadata_sequence()` | 145-151 | Sequence vs string/bytes |
| `configure_optimizers()` | 155-168 | None optimizer error + scheduler |
| `EpochConsoleLogger` | 172-211 | Callback for console logging |

**Recommended tests:**
1. Create a `MockBaseModule` subclass testing all lifecycle methods
2. `_metrics_for_stage()` with "test" and invalid stage
3. `_log_step_outputs()` with empty and populated metrics
4. `test_step()` with None and populated callback_preds
5. `transfer_batch_to_device()` with metadata batch (3-tuple) and regular batch
6. `_is_metadata_batch()` with valid/invalid batches
7. `configure_optimizers()` with None (ValueError) and with scheduler
8. `EpochConsoleLogger` — `_format()`, `on_train_epoch_end()`, `on_validation_epoch_end()`

**Priority: HIGH** — Base class for all modules, zero tests.

---

### 3.3 `dnanet/modules/peaknet.py` — 57% (23 untested lines)

**Test file:** `tests/models/test_peaknet_integration.py` — only 1 test (`test_training_step`).

| Untested Method | Lines | Description |
|---|---|---|
| `compute_test_step_outputs()` | 97-103 | Returns allele probabilities |
| `_compute_logits_and_targets()` | 105-125 | Full model forward pass |
| `_compute_loss_and_metric_inputs()` | 127-137 | Loss + argmax preds |
| `_allele_probabilities()` | 139-145 | Softmax extraction |
| `_split_batch()` | 147-190 | All 6 batch format branches |
| `predict_step()` | 192-208 | Softmax output |

**Recommended tests:**
1. `compute_test_step_outputs()` — verify 4-tuple with allele probabilities
2. `_allele_probabilities()` — valid case, invalid index (ValueError)
3. `_split_batch()` — nested 2-tuple, flat 6-tuple, 7-tuple, 5-input, 3-input, invalid len
4. `predict_step()` — verify softmax output shape and range [0,1]

**Priority: HIGH** — Core model module, 6 of 10+ methods untested.

---

### 3.4 `dnanet/modules/segmentation.py` — 65% (18 untested lines)

**Test file:** `tests/modules/test_segmentation.py`

| Missing | Lines | Description |
|---|---|---|
| `compute_test_step_outputs()` | 103-108 | Returns preds as callback output |
| `_split_batch()` | 110-116 | 3-tuple with metadata path |
| `_compute_loss_and_probabilities()` | 118-125 | Full forward + loss + sigmoid |
| `MultiClassSegmentationModule` | 135-163 | Entire subclass |

**Recommended tests:**
1. `MultiClassSegmentationModule` — test `_compute_loss_and_probabilities`, `predict_step`
2. `compute_test_step_outputs()` — verify 4-tuple with preds as callback output
3. `_split_batch()` with 3-tuple containing metadata

**Priority: MEDIUM** — Mostly covered, minor gaps.

---

### 3.5 `dnanet/tasks/train.py` — 63% (28 untested lines)

**Test file:** `tests/tasks/test_train.py`

| Missing | Lines | Description |
|---|---|---|
| `_load_state_dict()` | 113-131 | Lightning checkpoint with `model.` prefix |
| `_load_pretrained_weights()` | 134-150 | AE + PC checkpoint loading |
| `_save_config()` | 153-160 | Config file writing |
| `run()` | 163-254 | Full training pipeline |

**Recommended tests:**
1. `_load_state_dict()` — plain PyTorch dict, Lightning checkpoint with `model.` prefix
2. `_load_pretrained_weights()` — network with AE + PC attrs, network without them
3. `_save_config()` — write config.yaml, verify content
4. `run()` — integration test with mocked instantiate + trainer.fit

**Priority: HIGH** — Core entry point, key functions untested.

---

## 4. Medium Gaps (65-85% coverage, notable missing paths)

### 4.1 `dnanet/data/hid_dataset.py` — 79% (44 untested lines, 260 total)

**Test file:** `tests/data/test_hid_dataset.py`

| Missing | Lines | Description |
|---|---|---|
| `_load_cache_into_ram()` | 227-243 | RAM budget exceeded path |
| `_total_ram_bytes()` | 245-251 | OSError fallback |
| `_resolve_cache()` | 255-281 | Cache hit, stale fingerprint |
| `_build_cache()` | 283-303 | Resume from partial cache |
| `_load_images()` | 307-416 | Ladder handling, allele→scanpoint, span adjustment |
| `_translate_allele_to_scanpoint_annotation()` | 418-505 | Full conversion pipeline |
| `_adjust_and_flatten_span_annotation()` | 507-589 | Per-class span adjustment |
| `images` property | 599-604 | Stub images |
| `dataset_strategy` property | 606-609 | Strategy accessor |
| `__len__` | 613-614 | Length |
| `__repr__` | 616-617 | String repr |
| `__getitem__` | 619-624 | With/without transform |
| `_stub_image()` | 628-636 | Stub image creation |
| `get_stub_image()` | 638-640 | Delegates to _stub_image |
| `get_image()` | 642-647 | Materialize from cache |
| `_materialize()` | 649-687 | Full materialization pipeline |

**Recommended tests:**
1. `_load_cache_into_ram()` — mock to exceed RAM budget, verify RuntimeError
2. `_resolve_cache()` — cache hit (fingerprint valid), stale fingerprint rebuild
3. `_build_cache()` — with resume paths
4. `_load_images()` — AlleleAnnotation, SpanAnnotation, ScanpointAnnotation, missing data
5. `_translate_allele_to_scanpoint_annotation()` — with/without adjustment_type
6. `_adjust_and_flatten_span_annotation()` — with/without profile data
7. `__getitem__` with transform applied
8. `_materialize()` — with allele_json, panel_json, meta_json present

**Priority: MEDIUM** — Core dataset class, 44 untested lines.

---

### 4.2 `dnanet/data/strategies/datasets/nfi_rnd.py` — 79% (47 untested lines, 274 total)

**Test file:** `tests/data/test_nfi_rnd_splitting.py`, `tests/data/test_dataset_strategies.py`

Splitting is well tested. File parsing and annotation helpers are the gap.

| Missing | Lines | Description |
|---|---|---|
| `__init__()` | 59-77 | Invalid annotation_type assertion |
| `cache_signature()` | 79-80 | Returns dict |
| `collect_dataset_files()` | 82-138 | Full collection pipeline |
| `_parse_analyst_annotation()` | 140-178 | DTH/DTL annotation parsing |
| `_parse_ground_truth_annotations()` | 180-211 | Ground truth with caching |
| `_reference_file_stems_for_prefix()` | 213-223 | Donor reference lookup |
| `_build_ground_truth_annotation()` | 225-253 | Mixture annotation building |
| `_read_reference_profile()` | 255-286 | CSV donor reference parsing |
| `find_annotation_file()` | 326-337 | Annotation file discovery |
| `find_ladder_for_sample()` | 340-355 | Ladder fallback logic |
| `load_ladder_mapping()` | 357-369 | CSV ladder mapping |
| `parse_annotations()` | 371-405 | Empty file, multi-sample |
| `_parse_sample_annotations()` | 407-435 | Per-sample marker parsing |
| `_parse_csv_header()` | 437-449 | Delimiter detection |
| `_read_csv_file()` | 451-465 | CSV reading |
| `get_annotation_classes()` | 467-471 | span vs non-span |
| `_fractional_split()` | 514-614 | 3-way split, genotype-aware |
| `_kfold_split()` | 618-662 | StratifiedKFold path |
| `_get_mixture_dataset_groups()` | 665-684 | Group key extraction |

**Recommended tests:**
1. `cache_signature()` — verify return dict
2. `collect_dataset_files()` — each annotation type (DTH, DTL, ground_truth, span)
3. `_parse_analyst_annotation()` — mock AlleleReport files
4. `_reference_file_stems_for_prefix()` — prefix "1A2" → ["1A", "1B"]
5. `_read_reference_profile()` — mock CSV with donor data
6. `find_annotation_file()` — with/without txt files
7. `parse_annotations()` — empty file (RuntimeError), multi-sample
8. `_parse_csv_header()` — comma/semicolon/tab, no match
9. `_fractional_split()` — 3-way with test_fraction, genotype_aware=True
10. `_kfold_split()` — StratifiedKFold path with noc, Subset input path

**Priority: MEDIUM** — Splitting well tested, file parsing needs coverage.

---

### 4.3 `dnanet/data/strategies/datasets/dataset.py` — 82% (18 untested lines)

**Test file:** `tests/data/test_dataset_strategies.py`

| Missing | Lines | Description |
|---|---|---|
| `split()` | 147-167 | PeakWindowDataset conversion path |
| `annotation_to_idx` | 169-171 | Annotation index mapping |
| `_parse_span_annotation()` | 173-291 | CSV parsing, validation, merge |
| `_df_to_span_annotation()` | 293-332 | Row-to-tensor conversion |
| `_merge_span_annotations()` | 334-351 | Multiple annotator merge |
| `_span_to_scanpoint_annotation()` | 353-375 | argmax flattening, overlap logging |

**Recommended tests:**
1. `split()` — with PeakWindowDataset, with regular dataset
2. `annotation_to_idx` — verify mapping
3. `_parse_span_annotation()` — no CSV, missing columns, unknown dye, valid parsing
4. `_df_to_span_annotation()` — valid rows, out-of-bounds dye (skip), out-of-bounds category (raise)
5. `_merge_span_annotations()` — multiple annotators, single annotator
6. `_span_to_scanpoint_annotation()` — overlapping annotations, non-overlapping

**Priority: MEDIUM** — Core strategy base class, span annotation parsing untested.

---

### 4.4 `dnanet/data/datamodule.py` — 78% (7 untested lines)

**Test file:** `tests/data/test_datamodules.py`

| Missing | Lines | Description |
|---|---|---|
| `setup()` | 47-73 | ConcatDataset collate extraction, k_folds error |
| `train_dataloader()` | 75-85 | persistent_workers, pin_memory |
| `val_dataloader()` | 87-97 | persistent_workers, pin_memory |
| `test_dataloader()` | 99-112 | No test split error |

**Recommended tests:**
1. `setup()` — ConcatDataset with multiple collate functions (ValueError), matching collate
2. `setup()` — k_folds in split_kwargs (ValueError)
3. `train_dataloader()` — verify persistent_workers=True when num_workers > 0
4. `test_dataloader()` — without test split (RuntimeError)

**Priority: LOW** — Small file, mostly edge cases.

---

### 4.5 `dnanet/data/extracted_peak.py` — 82% (4 untested lines)

**Test file:** `tests/data/test_extracted_peak.py`

| Missing | Lines | Description |
|---|---|---|
| `annotation` property | 87-90 | Returns ClassAnnotation or None |
| `__eq__` | 98-105 | Non-ExtractedPeak (NotImplemented) |
| `__hash__` | 107-108 | Hash consistency |

**Priority: LOW** — Minor edge cases.

---

### 4.6 `dnanet/data/strategies/scaling/powerplex_y23.py` — 67% (3 untested lines)

**Test file:** None (only indirectly via `test_scaling_strategy.py`)

| Missing | Lines | Description |
|---|---|---|
| `__init__()` | 7-9 | Kit initialization |
| `marker_name_to_dye_idx()` | 11-17 | Dye index mapping |

**Priority: LOW** — Simple init + dict return.

---

### 4.7 `dnanet/data/strategies/scaling/kit.py` — 82% (4 untested lines)

**Test file:** `tests/data/test_kit.py`

| Missing | Lines | Description |
|---|---|---|
| `panel` cached_property | 57-62 | panel_path=None → None |
| `ladder_alleles` cached_property | 64-70 | panel=None → None |
| `dye_row_from_hid_index()` | 74-86 | Valid index conversion |

**Priority: LOW** — Cached property edge cases.

---

### 4.8 `dnanet/data/preprocessing/peak_extraction.py` — 85% (25 untested lines)

**Test file:** `tests/data/test_peak_extraction.py`

| Missing | Lines | Description |
|---|---|---|
| Various edge cases | 149, 155, 231, 238, 381, 397, 432, 436, 443, 472, 497, 504, 532, 534, 536, 544, 562-574 | Peak extraction edge cases |

**Recommended tests:**
1. Test each untested line with specific input scenarios
2. Focus on: invalid peak heights, edge window positions, boundary conditions

**Priority: MEDIUM** — Core preprocessing pipeline.

---

### 4.9 `dnanet/core/panel.py` — 87% (12 untested lines)

**Test file:** `tests/core/test_panel.py`

| Missing | Lines | Description |
|---|---|---|
| Various methods | 96, 158, 176-182, 192, 199, 209-210, 244 | Panel XML parsing edge cases |

**Priority: LOW** — Mostly covered, minor XML parsing edge cases.

---

### 4.10 `dnanet/evaluation/callbacks.py` — 80% (17 untested lines)

**Test file:** `tests/evaluation/test_allele_metrics_callback.py`, `tests/evaluation/test_per_rfu_callback.py`

| Missing | Lines | Description |
|---|---|---|
| Various callback methods | 63, 71, 79-81, 83, 90, 109-111, 128, 130, 178, 189, 218, 225, 238 | Callback lifecycle edge cases |

**Priority: LOW** — Mostly covered, lifecycle edge cases.

---

## 5. Branch Coverage Gaps

Beyond line coverage, several important **branch conditions** are not tested:

| File | Branch | Condition |
|---|---|---|
| `data/hid_dataset.py:332` | `if cache_valid:` | Cache hit path (always rebuilds) |
| `data/hid_dataset.py:342-344` | `if len(stub_image_data) == 0:` | Empty stub image handling |
| `data/hid_dataset.py:473` | `if not self._panel:` | Missing panel error path |
| `data/hid_dataset.py:604` | `images` property | Stub image path |
| `data/hid_dataset.py:662,668` | `if not allele_json/panel_json` | Missing metadata paths |
| `data/strategies/datasets/nfi_rnd.py:157` | `if not is_genotype:` | Non-genotype skip |
| `data/strategies/datasets/nfi_rnd.py:166` | `if not ground_truth:` | No ground truth path |
| `data/strategies/datasets/nfi_rnd.py:252` | `if not donor_profiles:` | No donors found |
| `data/strategies/datasets/nfi_rnd.py:304,307` | `if not annotation_data:` | Empty annotation |
| `data/strategies/datasets/nfi_rnd.py:333-337` | `if not annotation_data:` | Empty file error |
| `data/strategies/datasets/nfi_rnd.py:348-355` | `if not ladder_mapping:` | No ladder mapping |
| `data/strategies/datasets/nfi_rnd.py:364-369` | `if not os.path.exists(...)` | Missing ladder CSV |
| `data/strategies/datasets/nfi_rnd.py:388-389` | `if not samples:` | No samples to parse |
| `data/strategies/datasets/nfi_rnd.py:395` | `if not sample_data:` | Empty sample |
| `data/strategies/datasets/nfi_rnd.py:423` | `if marker not in data:` | Missing marker |
| `data/strategies/datasets/nfi_rnd.py:448-449` | `if not fields:` | Empty CSV fields |
| `data/strategies/datasets/nfi_rnd.py:470` | `if self._annotation_type == "span"` | Span vs non-span |
| `data/strategies/datasets/nfi_rnd.py:502` | `if not self._reference_profiles:` | No reference profiles |
| `data/strategies/datasets/nfi_rnd.py:526` | `if not profiles:` | Empty profiles |
| `data/strategies/datasets/nfi_rnd.py:580-599` | `_fractional_split()` | 3-way split, genotype-aware, stratify_noc |
| `data/strategies/datasets/nfi_rnd.py:647` | `if not nocs:` | No NoC for stratification |
| `data/strategies/datasets/nfi_rnd.py:677` | `if not groups:` | No mixture groups |
| `data/preprocessing/peak_extraction.py:149` | `if peak_height < min_height:` | Skip low peaks |
| `data/preprocessing/peak_extraction.py:155` | `if peak_height > max_height:` | Skip high peaks |
| `data/preprocessing/peak_extraction.py:231,238` | Window boundary checks | Edge windows |
| `data/preprocessing/peak_extraction.py:381` | `if not is_dye:` | Non-dye peak |
| `data/preprocessing/peak_extraction.py:397` | `if peak_center < window_start` | Peak outside window |
| `data/preprocessing/peak_extraction.py:432,436,443` | Threshold checks | Peak detection thresholds |
| `data/preprocessing/peak_extraction.py:472` | `if peak_height < min_height` | Min height filter |
| `data/preprocessing/peak_extraction.py:497,504` | Dye assignment | Dye index lookup |
| `data/preprocessing/peak_extraction.py:532,534,536,544` | Peak merging | Merge conditions |
| `data/preprocessing/peak_extraction.py:562-574` | Final peak validation | Validation pipeline |

---

## 6. Summary of Recommendations by Priority

### Phase 1: Critical (Core entry points, zero tests)
| # | File | Est. Tests | Effort |
|---|---|---|---|
| 1 | `tasks/evaluate.py` | 4-5 tests | 2-3 hours |
| 2 | `modules/base.py` | 6-8 tests | 3-4 hours |
| 3 | `cli.py` | 3-4 tests | 1-2 hours |

### Phase 2: High (Important functionality, major gaps)
| # | File | Est. Tests | Effort |
|---|---|---|---|
| 4 | `data/strategies/datasets/provedit.py` | 10-12 tests | 4-6 hours |
| 5 | `modules/peaknet.py` | 4-5 tests | 2-3 hours |
| 6 | `tasks/train.py` | 4 tests | 2-3 hours |

### Phase 3: Medium (Core data pipeline, notable gaps)
| # | File | Est. Tests | Effort |
|---|---|---|---|
| 7 | `data/hid_dataset.py` | 8-10 tests | 4-5 hours |
| 8 | `data/strategies/datasets/nfi_rnd.py` | 10-12 tests | 4-6 hours |
| 9 | `data/strategies/datasets/dataset.py` | 6 tests | 2-3 hours |
| 10 | `data/preprocessing/peak_extraction.py` | 8-10 tests | 3-4 hours |

### Phase 4: Low (Minor gaps, edge cases)
| # | File | Est. Tests | Effort |
|---|---|---|---|
| 11 | `logging.py` | 3 tests | 1 hour |
| 12 | `data/datamodule.py` | 3-4 tests | 1-2 hours |
| 13 | `data/extracted_peak.py` | 2-3 tests | 30 min |
| 14 | `data/strategies/scaling/powerplex_y23.py` | 1-2 tests | 30 min |
| 15 | `data/strategies/scaling/kit.py` | 2 tests | 30 min |

**Total estimated effort: 30-40 hours**

---

## 7. Files Not Worth Testing (Intentionally low coverage)

| File | Coverage | Reason |
|---|---|---|
| `tools/labeltool/tool.py` | 19% | GUI application, not automatable |
| `tools/labeltool/visualization.py` | 28% | GUI visualization, not automatable |
| `tools/labeltool/cli.py` | 27% | UI tooling, limited automation value |
| `__main__.py` | 0% | Entry point, covered by CLI tests |

---

## 8. Target Coverage Goals

| Category | Current | Target |
|---|---|---|
| Core (`core/`) | 95% | 100% |
| Data pipeline (`data/`) | 80% | 90% |
| Models (`models/`) | 95% | 98% |
| Modules (`modules/`) | 65% | 85% |
| Tasks (`tasks/`) | 55% | 85% |
| Evaluation (`evaluation/`) | 90% | 95% |
| Tools (`tools/`) | 30% | 50% (GUI excluded) |
| **Overall** | **78%** | **88%** |
