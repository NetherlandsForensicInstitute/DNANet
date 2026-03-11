import glob
import os
from functools import partial
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Union

import confidence
from confidence import Configuration, dumpf

from DNAnet.data.data_models.base import InMemoryDataset
from DNAnet.data.data_models.hid_dataset import HIDDataset
from DNAnet.data.data_models.peak_dataset import PeakWindowDataset
from DNAnet.evaluation import (
    allele_f1_score,
    allele_precision,
    allele_recall,
    average_binary_iou,
    pixel_f1_score,
    pixel_precision,
    pixel_recall,
)
from DNAnet.evaluation.reconstruction import reconstruction_mse
from DNAnet.models.base import Model, TrainableModel
from DNAnet.typing import PathLike
from utils import get_defaults
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError


DATASETS = {
    'dataset': {
        'hid': HIDDataset,
        'peak_dataset': PeakWindowDataset
    }
}


def get_models() -> Dict[str, Dict[str, type]]:
    # Local import: avoids circular imports and only imports the relevant models when needed.
    from DNAnet.models.classification.peak_classification import PeakClassification
    from DNAnet.models.reconstruction.autoencoder import HIDAutoencoder
    from DNAnet.models.segmentation.human_analysis import HumanAnalysis
    from DNAnet.models.segmentation.peaknet import PeakNet
    from DNAnet.models.segmentation.trainable_unet import DNANet_UNet

    return {
        "model": {
            "unet": DNANet_UNet,
            "human_analysis": HumanAnalysis,
            "autoencoder": HIDAutoencoder,
            "peak_classification": PeakClassification,
        }
    }

METRICS = {'pixel_precision': pixel_precision,
           'pixel_recall': pixel_recall,
           'average_binary_iou': average_binary_iou,
           'pixel_f1_score': pixel_f1_score,
           'allele_precision': allele_precision,
           'allele_recall': allele_recall,
           'allele_f1_score': allele_f1_score,
           'reconstruction_mse': reconstruction_mse}
HF_DATASET = "NetherlandsForensicInstitute/DNANet_2p5pMixture_PPF6C_2024"


def load_config(path: PathLike, kind: Optional[str] = None) -> Configuration:
    """
    Load the data config file for the provided `path`. This can either be
    the name of the yaml itself or the full path to a config file.
    """
    # first check whether the path refers to an existing file. If not, we extract the
    # full path and then load the file.
    if not os.path.isfile(path):
        path = get_config_path(path, kind)
    return confidence.loadf(path)


def get_config_path(
        name: str,
        kind: Optional[str] = None,
        root: PathLike = "."
) -> Path:
    """
    Return the expected path to a local config file given the specified `name`
    and `kind`. Optionally, you can specify a `root` directory from where to
    locate the config file. If omitted, the current working directory is used.

    :param name: The base name of the config file either with or without its
        ``.yaml`` extension. If the `.yaml` extension is omitted, it is
        automatically appended.
    :param kind: The type of config file to load. This will typically be a
        subdirectory in the ``<root>/config/`` directory, such as ``data``,
        ``models``, ``training`` or ``evaluation``. Ignored if ``path`` is an
        absolute path.
    :param root: The parent directory of the ``config`` directory that
        contains all the (nested) config files. If omitted, the current
        working directory is used. Ignored if ``path`` is an absolute path.
    :return: Path object representing the full path to the correct config file.
    """
    if not name.endswith(".yaml"):
        name += ".yaml"
    elements = [root, "config"]
    if kind:
        elements.append(kind)
    elements.append(name)
    return Path(os.path.join(*elements))


def load_evaluation_config(source: PathLike) -> Dict[str, Any]:
    """
    Load the metrics found in the evaluation config.
    """
    evaluation_config = load_config(source, kind='evaluation')
    metrics = {}
    for metric in evaluation_config.evaluation.metrics:
        metric = dict(metric)
        func_name = metric.pop('name')
        if metric:
            func_name_dict = func_name + [f"_{k}_{v}" for k, v in metric.items()][0]
        else:
            func_name_dict = func_name
        metrics[func_name_dict] = partial(METRICS[func_name], **metric)
    return metrics


def load_training_config(source: PathLike) -> Dict[str, Any]:
    """
    Load the training parameters from a config file.
    """
    training_config = load_config(source, kind='training')
    return parse_config(training_config)['training']


def load_dataset(source: PathLike) -> InMemoryDataset:
    """
    Load a dataset from a config file.
    """
    data_config = dict(load_config(source, kind='data'))
    data_config["dataset"] = dict(data_config["dataset"])
    root_dir = Path(data_config["dataset"]["root"])
    files_in_root: bool = any(root_dir.glob("[!.]*"))  # Check for non-hidden files (not starting with `.`)

    # If the root_dir does not exist, or it exists but there's no (hidden) files, we download from HF
    if not (root_dir.exists() and files_in_root):
        print(f"Downloading dataset {HF_DATASET}...")
        if (data_root := data_config["dataset"].get("data_root", None)) is None:
            raise ValueError("No `data_root` provided, needed for HuggingFace download")
        download_online_dataset(data_root)

    data_config["dataset"].pop("data_root", None)
    dataset = parse_config(data_config, DATASETS)['dataset']
    return dataset


def download_online_dataset(data_root: PathLike):
    downloading, retry_count = True, 1
    while downloading:
        try:
            local_path = snapshot_download(
                repo_id=HF_DATASET,
                repo_type="dataset",
                local_dir_use_symlinks=False,
                max_workers=1
            )
            break
        except HfHubHTTPError as e:
            if "429" in str(e) and retry_count < 3:
                print(
                    "Hit HuggingFace rate limit (HTTP 429).\n"
                    "Retrying in 5s..."
                )
                retry_count += 1
                time.sleep(5)
            else:
                raise e

    # Read in data root
    shutil.copytree(local_path, data_root)

    # Clean up the download folder of remaining files
    shutil.rmtree(local_path)
    pass


def load_model(source: Union[PathLike, Mapping[str, Any]]) -> Model:
    """
    Load a trainable model from a config file or mapping.
    """

    if isinstance(source, PathLike):
        # The source is a path to a config file or checkpoint directory. We check if it's a
        # directory first, and if so, we look for the config file and checkpoint inside it. If
        # it's not a directory, we assume it's a config file.
        if os.path.isdir(source):
            # search for the yaml config file and load the checkpoint
            yaml_files = glob.glob(os.path.join(source, '*.yaml'))
            if len(yaml_files) == 0:
                raise FileNotFoundError(f"No config file found in {source}.")
            if len(yaml_files) > 1:
                raise ValueError(f"More than one config file found in {source}.")
            if not glob.glob(os.path.join(source, '*.pt')):
                raise FileNotFoundError(f"No checkpoint (.pt) file found in {source}.")
            config_file = load_config(yaml_files[0])
            # take only the model-part of the config as there might be other configurations
            # present in the yaml
            model_config = Configuration({'model': config_file['model']})
            model = parse_config(model_config, get_models())['model']
            model.load(source)  # load the checkpoint file
            return model
        else:
            # assume source is a config file
            model_config = load_config(source, kind='models')
            return parse_config(model_config, get_models())['model']
    else:
        # The source is a mapping. We assume it's already parsed and just need to be instantiated.
        source: Mapping[str, Any]
        # print(f"Loading model from mapping: {source}")
        model_config = Configuration({'model': source})
        return parse_config(model_config, get_models())['model']


def parse_config(config: Union[str, Path, Mapping[str, Any]],
                 config_options: Optional[dict] = None) \
        -> MutableMapping[str, Any]:
    """
    Recursively parse a `config` mapping consisting of serialized values of
    built-in types (e.g. `str`, `int`, `list`, etc.) and deserialize them by
    applying the appropriate callbacks in `config_options`.

    This function iterates over each `(key, value)` pair in `config`. If the
    `key` matches a `key` in `config_options` and `value` is itself a `Mapping`, the
    corresponding callback in `config_options` (i.e. `config_options[key]`) is used to
    instantiate a Python object from the serialized items in `value`.
    """

    def parse_item(key: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            value = dict(value)
            name = value.pop('name', None)
            value = parse_config(value, config_options)
            if config_options and key in config_options and name in config_options[key]:
                return config_options[key][name](**value)
            value['name'] = name # Add the name back to the value dict, in case it is needed for parsing nested items.
            return value

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return [parse_item(key, v) for v in value]

        return value

    return {k: parse_item(k, v) for k, v in config.items()}


def dump_config(
        path: PathLike,
        data_config_path: str,
        model_config_path: str,
        training_config_path: Optional[str] = None,
        validation_config: Optional[Union[float, str]] = None,
        dataset: Optional[InMemoryDataset] = None,
        model: Optional[Model] = None,
):
    """
    Extracts the config entries that can safely be dumped and appends any
    default arguments used to instantiate the relevant objects.

    Some default values for special arguments in the training configuration
    will automatically be filtered out, because those arguments are typically
    set dynamically by a script rather than originating from a training
    configuration file. This means you typically do not want to see default
    values for these arguments in the dumped config file, because you normally
    wouldn't specify these arguments in a training config file to begin with.

    The special training defaults that will be filtered out are as follows:

        - ``checkpoint_dir``
        - ``validation_set``
    """
    # Get defaults for the dataset class if we can.
    try:
        data_defaults = get_defaults(dataset.__class__) if dataset else {}
    except TypeError:
        data_defaults = {}

    # Get defaults for the models class if we can.
    try:
        model_defaults = get_defaults(model.__class__) if model else {}
    except TypeError:
        model_defaults = {}

    # Get defaults for the model's `.fit()` method.
    training_defaults = {}
    if model and isinstance(model, TrainableModel):
        training_defaults = get_defaults(model.fit)

    # Remove special training arguments that are not normally part of a
    # training configuration (because these are specified via the CLI).
    if "validation_set" in training_defaults:
        del training_defaults["validation_set"]
    if "checkpoint_dir" in training_defaults:
        del training_defaults["checkpoint_dir"]

    # Dataset serialization: prefer dataset.serialize() when provided; otherwise load config.
    data_config = {}
    if dataset and hasattr(dataset, "serialize"):
        try:
            data_config = dataset.serialize()
        except Exception:
            data_config = {}
    if len(data_config) == 0 and data_config_path:
        data_config = load_config(data_config_path, kind='data')
        data_config = {**data_defaults, **dict(data_config.dataset)}
    model_config = load_config(model_config_path, kind='models')
    model_config = {**model_defaults, **dict(model_config.model)}

    if training_config_path:
        training_config = load_config(training_config_path, kind='training')
        training_config = {**training_defaults, **dict(training_config.training)}
    else:
        training_config = {}

    validation_data_config = {}
    if validation_config:
        if isinstance(validation_config, str):
            validation_data_config = load_config(validation_config, kind='data')
            validation_data_config = {**data_defaults, **dict(validation_data_config.dataset)}
        elif isinstance(validation_config, InMemoryDataset) and hasattr(validation_config, "serialize"):
            validation_data_config = validation_config.serialize()
        else:
            validation_data_config = {'split': validation_config}

    output_config = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "validation": validation_data_config
    }
    dumpf(Configuration(output_config), path)
    return output_config
