# DNANet
[![pdm-managed](https://img.shields.io/endpoint?url=https%3A%2F%2Fcdn.jsdelivr.net%2Fgh%2Fpdm-project%2F.github%2Fbadge.json)](https://pdm-project.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python Version](https://img.shields.io/badge/Python-%3E%3D3.10-brightgreen?logo=python)

# Welcome!
This a Python repository that can be used to analyze DNA profiles using deep learning. It contains functionality to parse .hid files and train and 
evaluate models. The pre-trained U-Net provided can be used to call alleles in a DNA profile.

If you find this repository useful, please cite
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
for the data. Our publication describing the code and model is forthcoming.

## Requirements
Python >= 3.10, <=3.12

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

Git LFS is used to track `.hid` and `.pt` files. 

## Code overview
The repository is roughly organized into three sections:
* Data
* Models
* Evaluation

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


panel = Panel("resources/data/SGPanel_PPF6C_SPOOR.xml")
image = HIDImage(
  path="resources/data/2p_5p_Dataset_NFI/Raw data .HID files/Mixture dataset 1/Inj5 2017-05-01-09-45-24-128/1A2_A01_01.hid",
  annotations_file="resources/data/2p_5p_Dataset_NFI/.txt bestanden 2024 naam/Dataset 1 DTL_AlleleReport.txt",
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
  panel="resources/data/SGPanel_PPF6C_SPOOR.xml",
  annotations_path="resources/data/2p_5p_Dataset_NFI/.txt bestanden 2024 naam",
  hid_to_annotations_path="resources/data/2p_5p_Dataset_NFI/2p_5p_hid_to_annotation.csv",
  limit=10
)
```

The list of `HIDImage`'s is stored in the `._data` attribute of the class. 

Note that when loading the 2p-5p R&D dataset without limit, two hid files do not pass data validation, leaving the dataset with 348 images instead of 350.

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
This model creates a binary segmentation, where `1` indicates the presence of a peak and `0` otherwise. 

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
The `HumanAnalysis` model can be used to analyze the analyst's annotations. It is interesting to compare those with the 
ground truth donor alleles. For the 2p-5p R&D Dataset, the actual donor alleles are known. By setting `ground_truth_as_annotations: True` in the `dnanet_rd.yaml` file, those ground truth donor alleles will be stored in 
`meta['called_alleles]` and the analyst annotations in `meta['called_alleles_manual']` of the `HIDImage` when loading the dataset.

When applying the `HumanAnalysis` model to the dataset, the values in `meta['called_alleles_manual']` of the `HIDImage` will be stored in the
`meta['called_alleles']` of a `Prediction` object. This way, the analyst annotations can be compared to the ground truth alleles. 

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
  panel="resources/data/SGPanel_PPF6C_SPOOR.xml",
  annotations_path="resources/data/2p_5p_Dataset_NFI/.txt bestanden 2024 naam",
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
  panel="resources/data/SGPanel_PPF6C_SPOOR.xml",
  annotations_path="resources/data/2p_5p_Dataset_NFI/.txt bestanden 2024 naam",
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
