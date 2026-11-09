# Synthetic Data

The synthetic data component generates artificial STR profiles and
electropherograms (EPGs). It lives in `src/synthetic-profile-generation/` and is
**separate from the `dnanet` Python package**: it has its own R and TensorFlow
environments and is not installed by `pip install`.

The pipeline has two steps:

```
simulateDNA (R)                        generateEPG (Python + TensorFlow)
simDNAmixtures  ->  per-profile CSVs  ->  Gaussian peaks + GAN  ->  .npy EPGs
```

The detailed reference is the
[component README](https://github.com/NetherlandsForensicInstitute/DNANet/blob/main/src/synthetic-profile-generation/README.md).

## Step 1: Simulate DNA profiles (R)

Uses [`simDNAmixtures`](https://cran.r-project.org/package=simDNAmixtures) with
a pinned `renv` environment. The pipeline is wired to the GlobalFiler kit.

```bash
cd src/synthetic-profile-generation/simulateDNA
Rscript -e "renv::restore()" --vanilla

# Optional argument: output suffix (default: timestamp)
Rscript simulateDNA.R example
```

Run it from this directory, paths are resolved relative to it.
Output goes to `src/synthetic-profile-generation/generated/generated_alleles_<suffix>/`: EPG CSVs, reference
genotypes, a mapping file and `run_parameters.json`.

Settings (detection threshold, template ratios, DNA amounts, degradation,
number of contributors and replicates) are in `sim_helpers.R`.

## Step 2: Generate EPGs (Python + TensorFlow)

`generateEPG/generate.py` builds an idealized EPG with Gaussian peaks from the
simulated peak heights and positions, then passes it through a trained GAN
generator that adds noise and realism.

- Needs `tensorflow==2.14.0` (Python <= 3.11), so use a **separate
  environment** from the main project.
- The `.h5` models in `generateEPG/TGAN_model/` are stored with
  [Git LFS](https://git-lfs.com/). Pull them first.

```bash
cd src/synthetic-profile-generation/generateEPG

python generate.py \
    --csv_dir='../generated/generated_alleles_example/epgs' \
    --output_dir='../generated_epgs/1' \
    --batch_size=64 --epg_shape 4000 5000 --std_dev 4
```

| Argument | Description |
|----------|-------------|
| `--csv_dir` | Folder of simulated profile CSVs |
| `--output_dir` | Where the `.npy` EPGs are written |
| `--batch_size` | Profiles per batch |
| `--epg_shape SCAN_MIN LENGTH` | Scan window kept from the real EPG, mapped to a `LENGTH`-wide array. Default `4000 5000` since no allelic peaks lie outside 4000-9000 |
| `--std_dev` | Standard deviation of the idealized Gaussian peaks |
| `--no_generator` | Skip the GAN and write the idealized EPG only |

To compare idealized and GAN output side by side, generate both
(`.../with_generator` and `.../without_generator`) and run
`python visualization.py --file <epg>.npy`.

## Credits

The GAN-based generator was designed by Duncan Taylor and Melissa Humphries
(*Expert Systems with Applications*, 2025, 280:127536). The pipeline is
described in the master thesis of Amar Mesic (TU Delft, 2025). Full citations
are in the component README and the main README.
