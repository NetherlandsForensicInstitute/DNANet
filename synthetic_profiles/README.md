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
```

Outputs land under `generated/generated_alleles_fixed_ratios_all_loci_thresh_15_8k` with EPG CSVs, reference genotypes, and a mapping file. Paths are resolved relative to the repo root, so the command works from any current working directory.
