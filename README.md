# DNANet
[![pdm-managed](https://img.shields.io/endpoint?url=https%3A%2F%2Fcdn.jsdelivr.net%2Fgh%2Fpdm-project%2F.github%2Fbadge.json)](https://pdm-project.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python Version](https://img.shields.io/badge/Python-%3E%3D3.10-brightgreen?logo=python)

# Welcome!
This a Python repository that can be used to analyze DNA profiles using deep learning. It contains functionality to parse .hid files and train and 
evaluate models. The pre-trained U-Net provided can be used to call alleles in a DNA profile.

If you find this repository useful, please cite
```bibtex
@ARTICLE{Benschop2019,
      title     = "An assessment of the performance of the probabilistic genotyping
                   software {EuroForMix}: Trends in likelihood ratios and analysis
                   of Type {I} \& {II} errors",
      author    = "Benschop, Corina C G and Nijveld, Alwart and Duijs, Francisca E
                   and Sijen, Titia",
      journal   = "Forensic Sci. Int. Genet.",
      volume    =  42,
      pages     = "31--38",
      year      =  2019,
    }
```
for the data, and 
```bibtex
@ARTICLE{de-Wit2025,
    title = {Making AI accessible for forensic DNA profile analysis},
    journal = {Forensic Science International: Genetics},
    volume = {81},
    pages = {103345},
    year = {2026},
    issn = {1872-4973},
    doi = {https://doi.org/10.1016/j.fsigen.2025.103345},
    url = {https://www.sciencedirect.com/science/article/pii/S1872497325001255},
    author = {
        Abel K.J.G. de Wit and Claire D. Wagenaar and Nathalie A.C. Janssen and Brechtje Hoegen 
        and Judith van de Wetering and Huub Hoofs and Simone Ariëns and Corina C.G. Benschop 
        and Rolf J.F. Ypma
        }
}
```
for the code and model.

## Requirements
Python >= 3.10, <=3.12

## Cloning
Currently the repo has exceeded its git lfs quota. This is a current issue, which causes problems in the cloning process since there are files making use of git lfs. To be able to clone the repo without issues, temporarily disable git lfs when cloning. The commands are:
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <REPO_URL> DNANet

cd DNANet
git lfs install --skip-smudge
```

## Setup
Create a virtual environment. We have used `pdm` and a `pyproject.toml` file to manage environment dependencies. Ensure you
have pdm installed:
```bash
$ pip install pdm
```
Then run the following command to install the dependencies:
```bash
$ pdm install
```

Git LFS is used to track `.pt` (model) files. Make sure to [install Git LFS](https://git-lfs.com/) on your system. In order to retrieve the files from the remote run the following command:
```bash
$ git lfs pull
```

HugginFace datasets is used to download the research data. This is done automatically whenever the data is missing from your config's provided root directory.
When this is triggered, data is pulled from the ["NetherlandsForensicInstitute/DNANet_2p5pMixture_PPF6C_2024"](https://huggingface.co/datasets/NetherlandsForensicInstitute/DNANet_2p5pMixture_PPF6C_2024) HuggingFace repository.

## Code overview
The repository is roughly organized into four sections:
* Data
* Models
* Evaluation
* Labeltool

Additionally, you can run a training script, an evaluation script and a cross validation script from the command line. 

To load datasets and models or to load settings for the scripts, the code relies on config files that are
read via the package `confidence` (see https://github.com/NetherlandsForensicInstitute/confidence). The config files are located in the `config` folder,
and can be adjusted as desired.

# Data
This directory contains all logic to parse a .hid DNA profile into an `HIDImage` object. Multiple `HIDImage`'s 
are stored in a `HIDDataset`. The `HIDDataset` class is specifically implemented to load the 2p-5p NFI dataset,
containing 350 raw hid files and the annotations in .txt files. This raw data is stored in the `resources/data` folder.

The `HIDDataset` inherits from the `InMemoryDataset` and ensures the `HIDDataset` can be considered as a list of `HIDImage`'s,
as the `InMemoryDataset` is iterable over instances in the `._data` attribute. The `InMemoryDataset` class also contains functionality for shuffling and splitting the
dataset. 

In a similar fashion, the `HIDImage` inherits from the `Image` base class, enforcing the presence of raw data, an
annotation and meta information in respectively the `data`,  `annotation` and `meta` properties.

To load an `HIDImage`, you can provide the direct path to the .hid file (and optionally information to load annotations):

```bash
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.data_models import Panel


panel = Panel("resources/data/SGPanel_PPF6C.xml")
image = HIDImage(
  path="resources/data/2p_5p_Dataset_NFI/Raw data .HID files/Mixture dataset 1/Inj5 2017-05-01-09-45-24-128/1A2_A01_01.hid",
  annotations_file="resources/data/2p_5p_Dataset_NFI/txt_annotations_2024/Dataset 1 DTL_AlleleReport.txt",
  panel=panel,
  meta={'annotations_name': '1L_11148_1A2'}
)
```
The image has a `.data` attribute containing the numpy array of peak heights and a `.annotation` attribute containing the binary
segmentation of the ground truth location of peaks. These are based on the called alleles present in the annotations file, those 
can be found in the `called_alleles` of the `.meta` attribute.

An `HIDDataset` can be easily loaded using a config file:

```bash
from config_io import load_dataset

hid_dataset = load_dataset("config/data/dnanet_rd.yaml")
```

or directly by providing arguments:

```bash
from DNAnet.data.data_models.hid_dataset import HIDDataset

hid_dataset = HIDDataset(
  root="resources/data/2p_5p_Dataset_NFI/Raw data .HID files",
  panel="resources/data/SGPanel_PPF6C.xml",
  annotations_path="resources/data/2p_5p_Dataset_NFI/txt_annotations_2024",
  hid_to_annotations_path="resources/data/2p_5p_Dataset_NFI/2p_5p_hid_to_annotation.csv",
  limit=10
)
```

The list of `HIDImage`'s is stored in the `._data` attribute of the class. 

Note that when loading the 2p-5p R&D dataset without limit, two hid files do not pass data validation, leaving the dataset with 348 images instead of 350.

## Annotations
There are three types of 'annotations' we can have for a hid image:

* Ground truth. These are the alleles of the donors to a sample. Will generally not be available for case data.
* Called alleles. This is what analysts generally create during case work, thus will be available for all case data. 
There can be multiple sets of called alleles, for instance when alleles are called using low or standard thresholds.
* Scan point annotations. For each profile, it is possible to annotate each scan point of each peak separately as belonging to a given
category, e.g. Allele, Stutter, Bleedthrough, Spike, Dye blob, etc. The current repository has a labeltool to help
in creating these annotations. All scan points not annotated are assumed to be baseline.

The scan point annotations give much detailed information on what can be seen in a profile, but are harder to create
at scale than the called alleles.

Models can potentially be trained or evaluated on any of the above. Note that the prediction problem will be binary
for the first to annotations, and multiclass for the third.


# Models

## U-Net
We have implemented a U-Net model to identify peaks in a DNA profile. The U-Net architecture can be found in `models.segmentation.unet_architecture.py`.
To load a trainable version of this exact model and make predictions, we can use:
```bash
from DNAnet.data.data_models.hid_dataset import HIDDataset
from config_io import load_dataset, load_model

hid_dataset = load_dataset("config/data/dnanet_rd.yaml")
unet_model = load_model("config/models/unet.yaml")
predictions = unet_model.predict_batch(hid_dataset)
```
For binary annotations, this model creates a binary segmentation, where `1` indicates the presence of 
an allele and `0` otherwise. 
For multiclass annotations, the different classes are predicted.

We have also implemented an `AlleleCaller` (see `DNAnet/allele_callers.py`) to translate the binary segmentation
into called alleles. This step is part of the `predict_batch()` function of the U-Net and will be applied when
`apply_allele_caller` is set to `True` in the `unet.yaml`. The called alleles are stored in the 
`meta` attribute of a `Prediction` object.

A trained U-Net is located in `resources/model/current_best_unet`. To load the model's weights:

```bash
unet_model.load("resources/model/current_best_unet")
```

Note that allele metrics (`DNAnet/evaluation/segmentation/allele_metrics.py`) cannot be used on predictions of the U-Net model if no `AlleleCaller` 
is applied. 

## HumanAnalysis
The `HumanAnalysis` model can be used to analyze the analyst's allele calls. It is interesting to compare those with the 
ground truth donor alleles. For the 2p-5p R&D Dataset, the actual donor alleles are known. By setting `ground_truth_as_annotations: True` in the `dnanet_rd.yaml` file, those ground truth donor alleles will be stored in 
`meta['called_alleles]` and the analyst allele calls in `meta['called_alleles_manual']` of the `HIDImage` when loading the dataset.

When applying the `HumanAnalysis` model to the dataset, the values in `meta['called_alleles_manual']` of the `HIDImage` will be stored in the
`meta['called_alleles']` of a `Prediction` object. This way, the analyst allele calls can be compared to the ground truth alleles. 

Note that pixel metrics (`DNAnet/evaluation/segmentation/pixel_metrics.py`) cannot be used on predictions of the `HumanAnalysis` model this does
not predict an image, so the `.image` attribute of a `Prediction` will remain `None`.

# Evaluation
To evaluate the U-Net we have implemented a couple of metrics. Metrics to analyse the performance
on pixel level and allele level, are located in `DNAnet/evaluation/segmentation/pixel_metrics.py` and
`DNAnet/evaluation/segmentation/allele_metrics.py` respectively. 

To visualize the DNA profiles, their annotations (if present) and/or predictions (if present), you can use the `plot_profile()` 
function from `visualizations.py`. This will plot the profiles one by one.  

For example:

```bash
from DNAnet.data.data_models.hid_dataset import HIDDataset
from DNAnet.evaluation import pixel_f1_score
from DNAnet.evaluation.visualizations import plot_profile

hid_dataset = HIDDataset(
  root="resources/data/2p_5p_Dataset_NFI/Raw data .HID files",
  panel="resources/data/SGPanel_PPF6C.xml",
  annotations_path="resources/data/2p_5p_Dataset_NFI/txt_annotations_2024",
  hid_to_annotations_path="resources/data/2p_5p_Dataset_NFI/2p_5p_hid_to_annotation.csv",
  limit=10
)
unet_model = load_model("config/models/unet.yaml")
unet_model.load("resources/model/current_best_unet/")
predictions = unet_model.predict_batch(hid_dataset)

print(pixel_f1_score(hid_dataset, predictions))
plot_profile(hid_dataset, predictions)
```

It is also possible to plot a DNA profile per marker, or to plot a single marker of a DNA profile:
```bash
from DNAnet.data.data_models.hid_dataset import HIDDataset
from DNAnet.evaluation.visualizations import plot_profile_markers

hid_dataset = HIDDataset(
  root="resources/data/2p_5p_Dataset_NFI/Raw data .HID files",
  panel="resources/data/SGPanel_PPF6C.xml",
  annotations_path="resources/data/2p_5p_Dataset_NFI/txt_annotations_2024",
  hid_to_annotations_path="resources/data/2p_5p_Dataset_NFI/2p_5p_hid_to_annotation.csv",
  limit=10
)

plot_profile_marker(hid_dataset)
plot_profile_markers(hid_dataset, marker_name='TPOX')
```

# Scripts
We have three scripts than can be run from the command line. To view the arguments of those
scripts, run: `python <script.py> --help`. 

## train.py
This script can be used to train models with specified settings. The user can provide training parameters
in the training config file, see for instance `config/training/segmentation.yaml`. 

Run for example:

```bash
python train.py \
  -m unet \  # load a model
  -d dnanet_rd \  # load a dataset
  -t segmentation \  # load training arguments
  -s 0.9 \  # apply a split to leave part for testing/evaluation
  -v 0.1 \  # apply a split for validation during training
  -o output/example_run_train  # write results and the trained model to this folder
```

## evaluate.py
This script can evaluate (trained) models by computing metrics. It is also possible to store the 
predictions of the model as .json file. Metrics can be provided using an evaluation config file, 
see for instance: `config/evaluation/segmentation.yaml`. Metrics will be written to a .txt file.

Run for example: 

```bash
python evaluate.py \
  -m unet \  # load a model
  -c resources/model/current_best_unet \  # load a checkpoint
  -d dnanet_rd \  # load a dataset
  -e segmentation \  # load evaluation metrics
  -s 0.1 \  # apply splitting
  -o output/example_run_eval \  # output folder to write results to
  -p  # also save the actual predictions
```

## cross_validate.py
This script can be used to apply k-fold cross validation. The dataset will be split into `k` folds, then
`k` train/test loops will be performed. Metrics will be averaged over those loops.

Run for example:

```bash
python evaluate.py \
  -m unet \  # load a model
  -c resources/model/current_best_unet \  # load a checkpoint
  -d dnanet_rd \  # load a dataset
  -t segmentation \  # load training arguments
  -e segmentation \  # load evaluation metrics
  -k 5 \  # number of folds to use
  -o output/example_run_cross_val  # write results to this folder
```



## New: ProvedIt / GlobalFiler support
- Load ProvedIt GlobalFiler mixtures directly (backwards-compatible with the NFI R&D workflow).
- Reusable strategies for dataset-specific parsing (`DatasetStrategy`), kit/panel metadata (`Kit`), and EPG scaling (`EPGScalingStrategy`).
- Utilities to split genotype workbooks into per-contributor CSVs and to train from an already-instantiated dataset.

### Quick start: working with ProvedIt
1. Download a GlobalFiler ProvedIt zip (e.g. `PROVEDIt_2-5-Person Profiles_3500 5sec_GF29cycles.zip`) from https://lftdi.camden.rutgers.edu/provedit/files/ and extract it. The root should contain both the `.hid` files and a genotype Excel file.
2. Extract contributor genotypes into per-sample CSVs (semicolon-separated) that the dataset strategy can load:
```python
from pathlib import Path
from DNAnet.data.dataset_compatibility.format_conversion import find_genotype_file, individualize_genotypes

# Path to the extracted ProvedIt dataset (contains .hid files and the genotype Excel file)
dataset_root_path = "/Users/amarmesic/Documents/tudelft/thesis/datasets/USE THIS - PROVEDIt_2-5-Person Profiles_3500 5sec_GF29cycles"

# Locate the genotype Excel file inside the root directory
genotype_file_path = find_genotype_file(dataset_root_path)

# Extract the genotype data into individual CSV files that downstream loading uses
# `genotypes_dir` is where we choose to store the genotypes of the contributors of the dataset.
genotypes_dir = Path("resources/data/ProvedIt/individual_genotypes")
individualize_genotypes(
    input_path=genotype_file_path,
    output_dir=genotypes_dir,
)
```
3. Instantiate the reusable kit/strategy objects and load the dataset:
```python
from pathlib import Path
from DNAnet.data.data_models import Panel
from DNAnet.data.data_models.hid_dataset import HIDDataset
from DNAnet.data.dataset_compatibility.dataset_strategy import ProvedItDatasetStrategy
from DNAnet.data.kit_compatibility.kit import GLOBALFILER_KIT
from DNAnet.data.kit_compatibility.scaling_strategy import ProvedItEPGScalingStrategy

panel = Panel(GLOBALFILER_KIT.panel_path)
dataset_strategy = ProvedItDatasetStrategy(
    panel=panel,
    genotypes_path=Path("resources/data/ProvedIt/individual_genotypes"),
)
scaling_strategy = ProvedItEPGScalingStrategy(kit=GLOBALFILER_KIT)

provedit = HIDDataset(
    root="/path/to/PROVEDIt_2-5-Person Profiles_3500 5sec_GF29cycles",
    panel=GLOBALFILER_KIT.panel_path,
    ground_truth_as_annotations=True,  # use contributor genotypes as annotations
    dataset_strategy=dataset_strategy,
    scaling_strategy=scaling_strategy,
    kit=GLOBALFILER_KIT,
)
```
This keeps legacy behavior intact: if you omit the strategies, the dataset/image loaders fall back to the original NFI R&D logic.

### Strategy & kit reference (extensible)
- [Kit](DNAnet/data/kit_compatibility/kit.py), see `GLOBALFILER_KIT` and `POWER_PLEX_FUSION_6C_KIT` objects): captures the size standard and panel for a multiplex. Create a new kit with a name, `InternalSizeStandard`, and a panel XML path.
- [DatasetStrategy](DNAnet/data/dataset_compatibility/dataset_strategy.py) defines how to categorize files, parse contributor IDs, and build `Marker` objects from contributor genotype CSVs. `ProvedItDatasetStrategy` handles filenames like `F07_RD14-0003-30_31-...` and reads per-contributor CSVs stored under `genotypes_path`.
- [EPGScalingStrategy](DNAnet/data/kit_compatibility/scaling_strategy.py) parses the size-standard dye and rescales EPGs. `ProvedItEPGScalingStrategy` mirrors the ProvedIt parsing pipeline; `NfiEPGScalingStrategy` preserves the legacy GlobalFiler flow.

To extend to a new kit/dataset, subclass the relevant strategy and wire it up when constructing `HIDDataset`/`HIDImage`:
```python
from DNAnet.data.dataset_compatibility.dataset_strategy import DatasetStrategy

class MyDatasetStrategy(DatasetStrategy):
    def categorize_file(self, file_name: str): ...
    def get_contributors(self, file_name: str): ...
    def build_marker(self, marker_name, allele_names): ...
```
and pass `dataset_strategy=MyDatasetStrategy(panel, genotypes_path=Path(...))` plus a matching `EPGScalingStrategy`.

### Training directly from an instantiated dataset
When working with custom datasets/strategies, you can skip data configs and train with an in-memory dataset:
```python
from train_with_dataset import run_with_dataset

run_with_dataset(
    dataset=provedit,
    model_config="config/models/unet.yaml",
    training_config="config/training/segmentation.yaml",
    output_dir="output/provedit_unet",
)
```




## scripts/select_ladder_for_images.py
This script is used to select the best ladder for every `HIDImage` in a dataset. The best
ladder is determined by counting how often an annotated peak falls within the allele bin. To do this,
we translate the allele name (e.g. 'AMEL-X') to a base pair location via a candidate ladder. Then we 
translate the base pair locations to pixels (using the image's scaler), so we can retrieve the peak 
height data for that allele and detect the presence of a peak. The candidate ladder for which 
most peaks are found, is declared to be the best ladder.

The output of this script is a csv file containing a mapping between the .hid filename stem 
(e.g. '1A2_A01_01') and the path to  the ladder file that resulted in the 'best' ladder. For
the high threshold (`DTH`) 2p-5p NFI data, the results can be found in 
`resources/data/2p_5p_Dataset_NFI/best_ladder_paths_DTH.csv`.

Note that for this algorithm, annotated images (having called alleles) are necessary.

## scripts/scan_point_annotation_statistics.py
This script combines annotation CSV files and publishes basic statistics (counts per label, sample type, and annotation width). For datasets with known ground truth, it also reports annotation-derived allele-calling performance.

Run for example:
```bash
python scripts/scan_point_annotation_statistics.py \
  -a <annotation_folder> \
  -o output/annotations_all.csv
```

Arguments:
- `-a`, `--annotation-folder-path`: folder containing annotation CSV files.
- `-o`, `--output-file`: path of the merged output CSV.

## scripts/annotator_agreement.py
This script compares repeated annotations from multiple annotators on the same profiles. It computes pairwise overlap/label agreement and produces visual outputs such as heatmaps, confusion plots, and Sankey diagrams.

Run for example:
```bash
python scripts/annotator_agreement.py \
  -a <annotation_folder> \
  -d dnanet_rd_annotation \
  -o output/annotations_repeated.csv
```

Arguments:
- `-a`, `--annotation-folder-path`: folder containing annotation CSV files.
- `-d`, `--data-config`: dataset config used to load profiles.
- `-o`, `--output-file`: path of the merged output CSV.

## scripts/create_paper_figures.py
This script generates publication-style figures from a trained model and selected profiles. It creates full-profile figures, optional marker-level figures, and a bin-type illustration figure.

Run:
```bash
python scripts/create_paper_figures.py
```

Note: input model/data paths and selected profile-marker pairs are defined in the script `__main__` block.

## labeltool.py
This interactive tool is used to create and edit scan-point annotations, and to compare annotations between users.

Run for annotation mode:
```bash
python labeltool.py \
  -u <user_name> \
  -f <annotations_csv> \
  -d <data_config>
```

Run for compare mode:
```bash
python labeltool.py \
  -c \
  -f <folder_with_annotation_csvs> \
  -d <data_config>
```

Arguments:
- `-u`, `--user`: user name used while writing/filtering annotations.
- `-f`, `--filepath`: annotation CSV file (annotation mode) or annotation folder (compare mode).
- `-d`, `--data-config`: dataset config to load profiles.
- `-c`, `--compare`: load all annotators and show stacked annotations in non-interactive compare mode.
