# Synthetic DNA Profile and Electropherogram Generation

# Part 1: Simulating DNA profiles
This folder hosts the R-based pipeline for generating synthetic DNA profiles with `simDNAmixtures`. Use the bundled `renv` workflow so everyone shares the same R package versions.

## Dependencies
- R (renv will install the required packages)
- Panel file already in the repo at `resources/data/SGPanel_Globalfiler_Panel.xml`

## Setup (renv)
From the repository root:
```bash
cd synthetic_profiles/simulateDNA
Rscript -e "renv::restore()" --vanilla
```
This restores the isolated `renv` environment using the committed `renv.lock` and installs the required packages (`simDNAmixtures`, `dplyr`, `xml2`).

## Run
From the repository root (after setup/restore):
```bash
cd synthetic_profiles/simulateDNA

# Optional: pass a custom suffix
Rscript simulateMassProduceRandomParamsFixedTemplate.R example
```

Outputs land under `generated/generated_alleles_<suffix>` with EPG CSVs, reference genotypes, and a mapping file. If no suffix is provided, the script uses a timestamp (e.g., `generated_alleles_20240606_153012`). Paths are not resolved relative to the repo root, so the command only works from the above working directory.
Each run also writes `run_parameters.json` into the output folder with the key parameters and RFU threshold.

## Tuning the generator (relevant section if you want to experiment with different settings)
Common knobs live in `synthetic_profiles/simulateDNA/sim_helpers.R` (template ratio functions, degradation settings, default template amounts, output naming, panel lookup helpers). Adjust them there; the main script stays focused on orchestration.

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


# Part 2: Generating synthetic Electropherograms
## Generating synthetic EPGs
You can turn simulated DNA profiles into realistic-looking electropherograms (EPGs) with `synthetic_profiles/generateEPG/generate.py`. The script takes peak heights and positions from the simulated CSVs, builds an idealized EPG with Gaussian-shaped peaks, and then passes it through the trained generator to add noise and realism.

Example:
```bash
cd synthetic_profiles/generateEPG

python generate.py --csv_dir='../generated/generated_alleles_example/epgs' --output_dir='../generated_epgs/1' --batch_size=64 --epg_shape 4000 5000 --std_dev 4
```
The paths are evaluated in their relation from the location of the file, which is currently in [synthetic_profiles/generateEPG/](synthetic_profiles/generateEPG/). The script reads the CSV profiles in `--csv_dir`, writes `.npy` EPGs to `--output_dir`, processes in batches of 64, shapes each EPG to 6x5000, and uses a Gaussian std dev of 4 for the idealized peaks.

`--epg_shape` takes two ints: `scan_min` and `epg_length`. With `--epg_shape 4000 5000`, the code keeps scans from 4000 to 4000+5000 and maps them to a 5000-wide array. Let's relate this to a real EPG. Scan point 5000 of a real EPG would be mapped to position 1000 in the synthetic EPG, and points 3999 and 9001 in a normal EPG would fall out of bounds. The default values were chosen because no allelic peaks are expected to be found outside the range [4000, 9000] of a real EPG.

### Using tensorflow
Environment note: `generate.py` has been verified with TensorFlow 2.14.0. TensorFlow 2.18.x fails here, and the main project dependencies conflict with installing TF directly. Use a separate venv/conda env just for EPG generation, install `tensorflow==2.14.0` there, and run `python generate.py ...` from that env.


## Visualizing synthetic EPGs
After you generate two sets of EPGs—one with the generator (default) and one without (`--no_generator`)—you can compare them side by side:
1. Keep the outputs under [synthetic_profiles/generated_epgs/with_generator/epgs](synthetic_profiles/generated_epgs/with_generator/epgs) and [synthetic_profiles/generated_epgs/without_generator/epgs](synthetic_profiles/generated_epgs/without_generator/epgs) and don't change the path names!
2. Run:
   ```bash
   cd synthetic_profiles

   python generateEPG/generate.py --csv_dir='../generated/generated_alleles_example/epgs' \
    --output_dir='../generated_epgs/with_generator'

   python generateEPG/generate.py --csv_dir='../generated/generated_alleles_example/epgs' \
    --output_dir='../generated_epgs/without_generator' --no_generator


   python visualization.py --file <epg_filename.npy>
   ```
   If you omit `--file`, the script lists the available files (like `ls ... | head`) and uses the first one it finds. It plots the without-generator and with-generator EPGs together for easy comparison.



## Credits
The GAN-based generator was designed by Duncan Taylor and Melissa Humphries and is described in:
```bibtex
@article{taylor_simulating_2025,
  title = {Simulating realistic short tandem repeat capillary electrophoretic signal using a generative adversarial network},
  volume = {280},
  issn = {09574174},
  urldate = {2025-04-23},
  journal = {Expert Systems with Applications},
  author = {Taylor, Duncan Alexander and Humphries, Melissa},
  month = jun,
  year = {2025},
  pages = {127536},
}
```
