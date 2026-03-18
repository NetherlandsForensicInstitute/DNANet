import itertools
import json
import logging
import math
import random
import shutil
from pathlib import Path
from typing import Optional, Callable, Mapping, Any, Dict, ChainMap, List, Tuple

import confidence
import mlflow
import numpy as np
from confidence import Configuration
from tqdm import tqdm

from DNAnet.data.data_models.base import InMemoryDataset, Metric
from DNAnet.models.base_model import TrainableModel
from DNAnet.typing import PathLike
from config_io import parse_config, load_model

_DIST_OPS = {"uniform", "loguniform", "choice", "randint", "int"}


def _unwrap_singleton_lists(x):
    """
    Recursively convert list values of length 1 into scalars.
    Keeps real list-valued parameters (len != 1) intact.
    """
    if isinstance(x, dict):
        return {k: _unwrap_singleton_lists(v) for k, v in x.items()}
    if isinstance(x, list):
        if len(x) == 1:
            return _unwrap_singleton_lists(x[0])
        return [_unwrap_singleton_lists(v) for v in x]
    return x


def reconstruct_mapping(
        flat_mapping: Mapping[str, Any],
        key_separator: str = '.'
) -> Dict[str, Any]:
    """
    Reconstruct a nested mapping from a flat mapping with compound keys.
    """
    config: Dict[str, Any] = {}
    for k, v in flat_mapping.items():
        *tree, leaf = k.split(key_separator)
        ref = config
        for t in tree:
            if t not in ref:
                ref[t] = {}
            ref = ref[t]
        ref[leaf] = v
    return config

def flatten_params_config(
        config: Mapping[str, Any],
        base: Optional[List[str]] = None
) -> Dict[str, List[Any]]:
    if not base:
        base = []
    res: Dict[str, List[Any]] = {}
    for k, v in config.items():
        keys = base + [k]
        if isinstance(v, Mapping):
            res = {**res, **flatten_params_config(v, base=keys)}
        else:
            if hasattr(v, '_source'):
                v = v._source
            res[".".join(keys)] = v if isinstance(v, list) else [v]
    return res

def create_temporary_checkpoint_dir(
        output_dir: PathLike,
        config: Mapping[str, Any]
) -> Path:

    experiment_hash = str(hash(confidence.dumps(Configuration(config))))
    checkpoint_dir = Path(output_dir) / experiment_hash
    if checkpoint_dir.exists():
        raise FileExistsError(
            f"Temporary `checkpoint_dir` {checkpoint_dir} exists")
    checkpoint_dir.mkdir()
    return checkpoint_dir

# -------------------------
# Training function factory
# -------------------------

def get_train_func_single_dataset(
        default_config: Configuration,
        logger: Optional[logging.Logger],
        metric: Metric,
        output_dir: Optional[PathLike] = None,
        training_set: InMemoryDataset = None,
        test_set: InMemoryDataset = None,
        validation_set: InMemoryDataset = None,
) -> Callable[..., float]:
    """
    Returns a train function to be used in search loops.

    NOTE: The returned train_func accepts optional overrides:
          training_set=..., validation_set=..., test_set=...
          If provided, these override the defaults captured in this closure.
    """

    def train_func(**kwargs) -> float:
        # Allow per-call dataset overrides (used by cross-validation)
        train_ds: InMemoryDataset = kwargs.pop("training_set", training_set)
        val_ds: Optional[InMemoryDataset] = kwargs.pop("validation_set", validation_set)
        test_ds: Optional[InMemoryDataset] = kwargs.pop("test_set", test_set)

        config = Configuration(default_config, reconstruct_mapping(kwargs))

        logger.info("Loading model...")
        model = load_model(config.model)
        if not isinstance(model, TrainableModel):
            raise TypeError(f"`model` is not trainable: {type(model)}")

        # Parse training arguments for the current experiment.
        training_kwargs = parse_config(config.training)

        # Create a temporary checkpoint dir if needed by the model.
        checkpoint_dir = output_dir and create_temporary_checkpoint_dir(
            output_dir, default_config)

        # MLflow (optional)
        if "mlflow" in config:
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
            parent = mlflow.active_run()
            mlflow.start_run(run_name="train",
                             experiment_id=config.mlflow.experiment_id,
                             nested=parent is not None,
                             )
            mlflow.autolog()
            configs = dict(ChainMap(*[
                config.data._source,
                config.model._source,
                config.training._source,
                {"checkpoint_dir": checkpoint_dir},
            ]))

            # configs = _unwrap_singleton_lists(configs)

            reconstructed = reconstruct_mapping(configs)
            flat_for_mlflow = flatten_params_config(reconstructed)

            # MLflow requires params to be stringifiable
            mlflow.log_params({k: str(_unwrap_singleton_lists(v)) for k, v in flat_for_mlflow.items()})

        # Pass validation_set / checkpoint_dir only if supported by the model.
        # training_kwargs.update(**filter_kwargs(
        #     model.fit,
        #     validation_set=val_ds,
        #     checkpoint_dir=checkpoint_dir,
        # ))

        logger.info("Training model...")
        model.fit(train_ds, **training_kwargs)

        # Clean up checkpoints
        if checkpoint_dir:
            shutil.rmtree(str(checkpoint_dir))

        # Choose evaluation set: prefer test_ds, otherwise val_ds
        eval_ds = test_ds if test_ds is not None else val_ds
        if eval_ds is None:
            raise ValueError("No evaluation dataset available for scoring.")

        logger.info("Applying model on evaluation set...")
        eval_items = list(eval_ds)
        predictions = list(map(model.predict, tqdm(eval_items)))

        logger.info("Evaluating model...")
        result = metric(eval_items, predictions)

        if "mlflow" in config:
            mlflow.log_metric(key="eval_metric", value=result)
            mlflow.end_run()

        if not isinstance(result, (float, int)):
            raise TypeError(
                f"Invalid return type for metric {metric.__name__}: "
                f"{type(result)} (should be float)")
        return result

    return train_func



def sample_random_trial(search_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts either a flattened or nested search space. Resolves distributions
    (e.g., {loguniform: [lo, hi]}) and returns a flat dict suitable for kwargs.
    """
    def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flatten_dict(v, key))
            else:
                out[key] = v
        return out

    # Detect flattened space and reconstruct if needed.
    is_flat = isinstance(search_space, dict) and any(
        isinstance(k, str) and "." in k for k in search_space.keys()
    )
    nested_space = reconstruct_mapping(search_space) if is_flat else search_space

    # Resolve distributions/categoricals to concrete values.
    sampled_nested = _resolve_value(nested_space)

    # Return flat params without distribution operators in the keys.
    sampled_flat = _flatten_dict(sampled_nested)

    return sampled_flat


# -------------------------
# Cross-validation helpers
# -------------------------




def cross_val_mean_score(
        train_func: Callable[..., float],
        base_training_set: InMemoryDataset,
        k: int,
        seed: int,
        logger: Optional[logging.Logger],
        **params: Any,
) -> Tuple[float, float, List[float]]:
    """
    Run K-fold cross-validation.
    For each fold i: train on CombinedDataset(other folds), validate on fold i.
    Returns (mean, std) of the metric across folds.
    """
    folds = base_training_set.split_k_fold(k, seed=seed)
    scores: List[float] = []

    for i in range(k):
        train_set, val_set = folds[i]

        # if logger:
        #     logger.info(f"CV fold {i+1}/{k}: "
        #                 f"train_parts={len(train_parts)} | "
        #                 f"val_len={len(val_set)} | "
        #                 f"train_len={len(train_set)}")

        score = train_func(
            training_set=train_set,
            validation_set=val_set,
            test_set=None,   # do not use the global test set during CV
            **params
        )
        scores.append(float(score))

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    if logger:
        logger.info(f"Cross-val summary (K={k}): mean={mean_score:.6f} ± {std_score:.6f}")
    return mean_score, std_score, scores


# -------------------------
# Random search helper
# -------------------------

def random_search(
        train_func: Callable[..., float],
        param_space_flat: Dict[str, Any],
        n_trials: int,
        seed: int,
        logger: Optional[logging.Logger],
        skip_duplicates: bool = True,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Simple with-replacement random search over param_space_flat.
    Avoids duplicates via a small 'seen' cache.
    """
    results: List[Tuple[Dict[str, Any], float]] = []
    seen = set()

    # random.seed(seed) # Uncomment to make random search deterministic

    try:
        while len(results) < n_trials:
            # sample = {name: _sample_from_domain(rng, domain)
            #           for name, domain in param_space_flat.items()}
            sample = sample_random_trial(param_space_flat)
            # Deduplicate exact repeats to get diverse trials
            hashable_items = tuple(sorted((k, _freeze_for_hash(v)) for k, v in sample.items()))
            if skip_duplicates and hashable_items in seen:
                continue
            seen.add(hashable_items)

            if logger:
                logger.info(f"Random trial params: {json.dumps(reconstruct_mapping(sample), indent=2)}")

            score = train_func(**sample)
            results.append((sample, score))
    except KeyboardInterrupt or mlflow.exceptions.MlflowException:
        mlflow.end_run()
        return results


    return results

def grid_search(
        train_func: Callable[..., float],
        params: Dict[str, List[Any]],
        **kwargs
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Performs a grid search by running `train_func` with all possible
    combinations of parameter configurations in `params`. Returns for each such
    combination the corresponding score returned by `train_func`

    Arguments:
        train_func: Callable[..., float], a callable that takes all the
            configurable parameters as arguments, performs a training loop and
            returns a metric indicating how "good" the parameter values were.
        params: Dict[str, List[Any]], a dictionary whose keys are the names of
            the configurable parameters and values are lists of corresponding
            parameter settings to try.

    Returns:
        List of tuples mapping a particular parameter configuration to the
        corresponding result (score). The list is sorted in descending order so
        that the parameter configuration yielding the best (highest) score is
        first.
    """
    results = []
    configs = [[(param, _to_number(value)) for value in values]
               for param, values in params.items()]
    experiments = \
        [dict(experiment) for experiment in itertools.product(*configs)]
    for experiment in experiments:
        result = train_func(**experiment, **kwargs)
        results.append((experiment, result))
    return results

# -------------------------
# Distribution sampling utils
# -------------------------

def _freeze_for_hash(x):
    """Recursively convert containers to hashable types for set/dict keys."""
    if isinstance(x, dict):
        return tuple(sorted((k, _freeze_for_hash(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_freeze_for_hash(v) for v in x)
    if isinstance(x, set):
        return tuple(sorted(_freeze_for_hash(v) for v in x))
    if isinstance(x, np.ndarray):
        return tuple(_freeze_for_hash(v) for v in x.tolist())
    return x

def _to_number(x: Any) -> Any:
    # Cast numeric-like strings (e.g., "1e-5") to float; leave others intact.
    if isinstance(x, str):
        if x == "None" or x == "none" or x == "null":
            return None
        try:
            return float(x)
        except ValueError:
            return x

    return x

def _is_dist_dict(d: Dict[str, Any]) -> bool:
    # True if this dict is a distribution spec like {"loguniform": [low, high]}
    return isinstance(d, dict) and len(d) == 1 and next(iter(d)) in _DIST_OPS

def _sample_loguniform(bounds: List[Any]) -> float:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"loguniform expects 2 bounds, got: {bounds}")
    low, high = map(_to_number, bounds)
    if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
        raise ValueError(f"loguniform bounds must be numeric, got: {bounds}")
    if low <= 0 or high <= 0:
        raise ValueError(f"loguniform bounds must be > 0, got: {bounds}")
    if low >= high:
        raise ValueError(f"loguniform low must be < high, got: {bounds}")
    # Sample in ln-space for numerical stability
    ln_low, ln_high = math.log(low), math.log(high)
    return math.exp(random.uniform(ln_low, ln_high))

def _sample_uniform(bounds: List[Any]) -> float:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"uniform expects 2 bounds, got: {bounds}")
    low, high = map(_to_number, bounds)
    if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
        raise ValueError(f"uniform bounds must be numeric, got: {bounds}")
    if low >= high:
        raise ValueError(f"uniform low must be < high, got: {bounds}")
    return random.uniform(low, high)

def _sample_int(bounds: List[Any]) -> int:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"int/randint expects 2 bounds, got: {bounds}")
    low, high = map(int, map(_to_number, bounds))
    if low > high:
        raise ValueError(f"int/randint low must be <= high, got: {bounds}")
    # Inclusive range
    return random.randint(low, high)

def _resolve_value(spec: Any) -> Any:
    # Resolve a single spec node to a concrete sampled value.
    if isinstance(spec, list):
        # List means categorical choice; cast elements if they are numeric-like.
        candidates = [_to_number(v) for v in spec]
        return random.choice(candidates)

    if isinstance(spec, dict):
        if _is_dist_dict(spec):
            k, v = next(iter(spec.items()))
            if k == "loguniform":
                return _sample_loguniform(v)
            if k == "uniform":
                return _sample_uniform(v)
            if k in ("int", "randint"):
                return _sample_int(v)
            if k == "choice":
                # Explicit categorical choice
                if not isinstance(v, (list, tuple)) or not v:
                    raise ValueError(f"choice expects non-empty list, got: {v}")
                return _resolve_value(list(v))
            raise ValueError(f"Unknown distribution operator: {k}")
        else:
            # Regular mapping: resolve children
            return {kk: _resolve_value(vv) for kk, vv in spec.items()}

    # Scalar: cast numeric-like strings
    return _to_number(spec)
