import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from DNAnet.models.prediction import Prediction
from DNAnet.typing import PathLike


def add_file_handler_to_logger(
        logger: logging.Logger,
        path: Optional[str] = None):
    """
    Adds a file handler to an existing `logging.Logger` object, so that logs
    will be written to the `path`.
    """
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(path, 'a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt, datefmt)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def prepare_output_file(path: str) -> str:
    """
    Create a new directory with all subdirectories included.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def save_predictions(predictions: Sequence[Prediction], filename: PathLike):
    """
    Serialize a list of Prediction objects to JSON file
    """
    predictions_dicts = [prediction.to_dict() for prediction in predictions]
    with open(filename, 'w') as f:
        json.dump(predictions_dicts, f)


def load_predictions(filename: PathLike) -> Sequence[Prediction]:
    """
    Deserialize a list of Prediction objects from JSON file
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return [Prediction.from_dict(item) for item in data]


def get_defaults(func: Callable) -> Dict[str, Any]:
    """
    Returns a dict of `func` arguments and their default values. Arguments
    without a default value are excluded.
    """
    argspec = inspect.getfullargspec(func)
    if argspec.defaults:
        return dict(zip(
            argspec.args[-len(argspec.defaults):],
            argspec.defaults
        ))
    return dict()
