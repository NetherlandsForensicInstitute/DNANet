# Synthetic DNA Profile Generation

This folder hosts the R-based pipeline for generating synthetic DNA profiles with `simDNAmixtures`. Use the bundled `renv` workflow so everyone shares the same R package versions.

## Dependencies
- R (renv will install the required packages)
- Panel file already in the repo at `resources/data/SGPanel_Globalfiler_Panel.xml`

## Setup (renv)
From the repository root:
```bash
cd synthetic_profiles
Rscript -e "renv::restore()" --vanilla
```
This restores the isolated `renv` environment using the committed `renv.lock` and installs the required packages (`simDNAmixtures`, `dplyr`, `xml2`).

## Run
From the repository root (after setup/restore):
```bash
cd synthetic_profiles
Rscript simulateMassProduceRandomParamsFixedTemplate.R
# Optional: pass a custom suffix for the output directory (defaults to timestamp)
Rscript simulateMassProduceRandomParamsFixedTemplate.R my_experiment_name
```

Outputs land under `generated/generated_alleles_<suffix>` with EPG CSVs, reference genotypes, and a mapping file. If no suffix is provided, the script uses a timestamp (e.g., `generated_alleles_20240606_153012`). Paths are resolved relative to the repo root, so the command works from any current working directory.
Each run also writes `run_parameters.json` into the output folder with the key parameters and RFU threshold (uses `jsonlite` if installed; otherwise falls back to a simple R dump).

## Tuning the generator
Common knobs live in `synthetic_profiles/sim_helpers.R` (template ratio functions, degradation settings, default template amounts, output naming, panel lookup helpers). Adjust them there; the main script stays focused on orchestration.

- **Detection threshold (RFU)**: `configure_global_filer()` sets `threshold_rfu`. We default to `15` as a middle ground: higher (e.g., ~80) misses many low-template peaks (allelic and artefactual), while very low (near 0) floods you with undetectable peaks and extra compute. Raise to be stricter, lower to keep more weak signals.
- **Template ratios**: The four rule-based ratio generators in `get_template_ratio_functions()` were hand-picked to mimic ProvedIt-like patterns. Feel free to add/replace functions to explore other mixture ratios.
- **Simulation settings**: `get_simulation_params()` holds the defaults for `base_template_amounts` (total DNA per sample), `degradation_settings` (shape/scale), `contributors_list`, `replicates` (same settings/genotypes, different randomness), and `n_per_config` (samples per parameter set). Tweak these lists to change the sweep.
- **Panel/kit support**: The script is wired to GlobalFiler. To switch kits, adjust the panel path in `build_panel_lookup()` and the dye map (`kits$<KitName>`). Supported kits are available via `simDNAmixtures::kits`; you can inspect them with:
  ```r
  local_env <- environment()
  utils::data(kits, envir = local_env)
  kits <- local_env$kits
  names(kits)  # list of kits
  ```
