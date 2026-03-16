import argparse
import json
import logging
import os
import time
from typing import Optional, List

import mlflow
import yaml
from confidence import Configuration

from DNAnet.data.data_models.base import Metric
from DNAnet.grid_search_utils import reconstruct_mapping, grid_search, random_search, flatten_params_config, \
    cross_val_mean_score, get_train_func_single_dataset

from DNAnet.utils import generate_random_name
from config_io import load_config, load_dataset, load_evaluation_config

LOGGER = logging.getLogger('dnanet')

def run(data_config: str,
        model_config: str,
        evaluation_config: str,
        params_config: str,
        split: Optional[float] = None,
        training_config: Optional[str] = None,
        validation_split: Optional[str] = None,
        output_dir: Optional[str] = None,
        analyze: bool = False,
        cross_validation: Optional[int] = None,
        random_search_trials: Optional[int] = None,
        repeated_grid_search_runs: int = 1,
        seed: int = 42):
    """
    Performs hyperparameter search (grid by default; random if requested).
    Optionally evaluates each configuration with K-fold cross-validation.

    :param context: Context object containing various utilities for loading
        models, datasets, metrics, etc.
    :param config: Configuration object that will be used as a base
        configuration for all experiments. It should contain information such
        as the model to use and any necessary hyperparameters.
    :param data_config: Data configuration file path.
    :param model_config: Model configuration file path.
    :param evaluation_config: Evaluation configuration file path.
    :param params_config: Hyperparameter configuration file path.
    :param split: The fraction of the dataset used for training. The other
        fraction of the data will be used for testing. If `validation_split` is
        specified, the remaining training set will further be split.
    :param training_config: Training configuration file path.
    :param validation_split: Fraction of the dataset to use for validation.
        Note that this split is made after `split` which splits the dataset
        into train and test. The validation split is made on the remaining
        train set.
    :param output_dir: Directory to store output files.
    :param cleanup: Function to clean up resources after running.
    :param logger: Logger object for logging messages.
    :param analyze: If True, more logging will be performed. Defaults to False.
    :param cross_validation: If specified and >1, use K-fold cross-validation with K=cv
        instead of a single validation split.
    :param random_search_trials: If specified and >0, perform random search
        with the given number of trials instead of full grid search.
    :param seed: Random seed for dataset splitting and random search.
    """
    if random_search_trials and repeated_grid_search_runs > 1:
        raise ValueError("Cannot use repeated_grid_search_runs with random search")


    default_config = Configuration(
                                   load_config(model_config, kind='models'),
                                   load_config(evaluation_config, kind='evaluation'),
                                   load_config(data_config, kind='data'))

    if training_config:
        default_config = Configuration(default_config,
                                       load_config(training_config,
                                                   kind='training'))

    if not output_dir:
        output_dir = os.path.join('output/grid_search', generate_random_name())
    os.makedirs(output_dir, exist_ok=True)


    log_path = os.path.join(output_dir, 'log_grid_search.txt')

    LOGGER.info(f"Logs will be written to {log_path}")

    metrics_dict = load_evaluation_config(evaluation_config)
    metrics: List[Metric] = []
    for name, func in metrics_dict.items():
        met = Metric(func=func, name=name)
        metrics.append(met)

    if not metrics:
        raise ValueError("You have to specify at least 1 metric for search")

    metric = metrics[0]
    if len(metrics) > 1:
        LOGGER.warning(f"More than 1 metric was specified; only the first is used: {metric.__name__}")
    else:
        LOGGER.info(f"Using metric: {metric.__name__}")





    LOGGER.info("Loading dataset...")

    training_set = load_dataset(data_config)
    test_set = None
    validation_set = None
    if split:
        LOGGER.info(f"Splitting dataset, using {split * 100}% for training")

        training_set, test_set = training_set.split(split, seed=seed)

    # Optional single validation split (used only when cv is None or 1)
    if validation_split:
        try:
            validation_config = float(validation_split)
            LOGGER.info(f"Validation split found, using {validation_config * 100}% to create a "
                        f"validation set...")
            dataset, validation_set = training_set.split(1 - validation_config)
        except ValueError:
            LOGGER.info("Validation config found, loading validation set...")
            validation_set = load_dataset(validation_split)

    # Persist the default config for reproducibility
    config_path = os.path.join(output_dir, 'default_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(default_config._source, f)
    LOGGER.info(f"Config written to {config_path}")

    # Core train function (works for single-run or per-fold)
    train_func = get_train_func_single_dataset(default_config=default_config,
                                               logger=LOGGER,
                                               metric=metric,
                                               output_dir=output_dir,
                                               training_set=training_set,
                                               test_set=test_set,
                                               validation_set=validation_set)

    # Wrapper that optionally performs K-fold and returns a single float
    def wrapped_train_func(**kwargs) -> float:
        LOGGER.info(f"Starting experiment with parameters:\n"
                    f"{json.dumps(reconstruct_mapping(kwargs), indent=2)}")

        reload_data = any(k.startswith("data.") for k in kwargs)
        if not reload_data:
            tr_set, val_set, te_set = training_set, validation_set, test_set
        else:
            # rebuild datasets from the per-trial config
            trial_cfg = Configuration(default_config, reconstruct_mapping(kwargs))

            LOGGER.info("Loading dataset...")
            tr_set = load_dataset(trial_cfg.data)
            te_set = None
            val_set = None
            if split:
                LOGGER.info(f"Splitting dataset, using {split * 100}% for training")
                fr = split
                tr_set, te_set = tr_set.split(fr, seed=seed)


            # Optional single validation split (used only when cv is None or 1)
            if validation_split and not (cross_validation and cross_validation > 1):
                try:
                    val_cfg = float(validation_split)
                    LOGGER.info(f"Validation split found, using {validation_config * 100}% to create a "
                                f"validation set...")
                    tr_set, val_set = training_set.split(1 - val_cfg)
                except ValueError:
                    LOGGER.info("Validation config found, loading validation set...")
                    val_set = load_dataset(validation_split)



        if cross_validation and cross_validation > 1:
            if 'mlflow' in default_config:
                mlflow.set_tracking_uri(default_config.mlflow.tracking_uri)
                parent = mlflow.active_run()
                mlflow.start_run(run_name="cross_validation",
                                 experiment_id=default_config.mlflow.experiment_id,
                                 nested=parent is not None,
                                 )
                mlflow.autolog()
                # Flatten hierarchical params into dot-separated keys for MLflow
                reconstructed = reconstruct_mapping(kwargs)
                flat_for_mlflow = flatten_params_config(reconstructed)

                # MLflow requires params to be stringifiable
                mlflow.log_params({k: str(v) for k, v in flat_for_mlflow.items()})
                mlflow.set_tags({"cv_folds": cross_validation, "seed": seed})

            mean_score, std_score, scores = cross_val_mean_score(
                train_func=train_func,
                base_training_set=tr_set,
                k=cross_validation,
                seed=seed,
                logger=LOGGER,
                **kwargs
            )

            if 'mlflow' in default_config:
                mlflow.log_metric("cv_mean", mean_score)
                mlflow.log_metric("cv_std", std_score)
                mlflow.end_run()

            LOGGER.info(f"Experiment done! {metric.__name__.capitalize()} (CV mean ± std): "
                        f"{mean_score:.6f} ± {std_score:.6f}")
            # result_value = mean_score
            result_value = scores
        else:
            # Single train/val/test path (original behavior)
            result_value = train_func(training_set=tr_set,
                                      validation_set=val_set,
                                      test_set=te_set,
                                      **kwargs)
            LOGGER.info(f"Experiment done! {metric.__name__.capitalize()}: {result_value}")

        return result_value


    # Load hyperparameter search space
    params = load_config(params_config, kind="grid_search")
    params_path = os.path.join(output_dir, 'params_config.yaml')
    with open(params_path, 'w') as f:
        yaml.dump(params._source, f)
    LOGGER.info(f"Hyper-parameter config written to {params_path}")

    # Choose search algorithm
    param_space_flat = flatten_params_config(params)
    start = time.time()


    if random_search_trials and random_search_trials > 0:
        LOGGER.info(f"Running RANDOM SEARCH for {random_search_trials} trials (seed={seed})")

        # Top‑level run for the whole random search. Children will nest under this.
        if 'mlflow' in default_config:
            mlflow.set_tracking_uri(default_config.mlflow.tracking_uri)
            mlflow.start_run(
                run_name="random_search",
                experiment_id=default_config.mlflow.experiment_id,
                nested=False,
            )
            experiment = default_config.training.experiment_name if 'experiment_name' in default_config.training else "None"
            mlflow.set_tags({"search": "random", "trials": random_search_trials, "seed": seed, "Experiment": experiment})

        results = random_search(
            train_func=wrapped_train_func,
            param_space_flat=param_space_flat,
            n_trials=random_search_trials,
            seed=seed,
            logger=LOGGER,
            skip_duplicates=False
        )

        if 'mlflow' in default_config:
            mlflow.end_run()

    else:
        LOGGER.info("Running GRID SEARCH")
        results = []
        # Top‑level run for the whole grid search. Children will nest under this.
        if 'mlflow' in default_config:
            mlflow.set_tracking_uri(default_config.mlflow.tracking_uri)
            mlflow.start_run(
                run_name="repeated_grid_search",
                experiment_id=default_config.mlflow.experiment_id,
                nested=False,
            )
            experiment = default_config.training.experiment_name if 'experiment_name' in default_config.training else "None"
            mlflow.set_tags({"search": "grid", "total_runs": repeated_grid_search_runs, "Experiment": experiment})

        for i in range(repeated_grid_search_runs):
            LOGGER.info(f"Starting grid search run {i + 1} of {repeated_grid_search_runs}")
            if i == 0:
                results = grid_search(wrapped_train_func, param_space_flat)
            else:
                # Continue accumulating results
                new_results = grid_search(wrapped_train_func, param_space_flat)

                results.extend(new_results)

        if 'mlflow' in default_config:
            mlflow.end_run()

    LOGGER.info("Training interrupted!")

    end = time.time()
    LOGGER.info("Finished all search experiments!")


    # Persist results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump([{
            'params': reconstruct_mapping(trial_params),
            'result': result
        } for trial_params, result in results], f, indent=4)



    # Log top-N summary
    LOGGER.info("""
    +---------+
    | RESULTS |
    +---------+
    """)
    for i, (params, result) in enumerate(results, start=1):
        LOGGER.info(f"Experiment (ranked #{i}):\n"
                    f"{json.dumps(reconstruct_mapping(params), indent=4)}\n"
                    f"{metric.__name__.capitalize()}: {result}")
    LOGGER.info(f"Logs have been written to {log_path}")
    time_s = end - start
    time_m = time_s // 60
    time_h = int(time_m // 60)
    time_m = int(time_m % 60)
    time_s = time_s % 60
    LOGGER.info(f"Total search time: {time_h}:{time_m}:{time_s:.2f}, number of experiments: {len(results)}")


# -------------------------
# CLI
# -------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Perform grid search over '
                                                 'hyperparameters.')
    parser.add_argument('-d',
                        '--data-config',
                        type=str,
                        required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('-m',
                        '--model-config',
                        type=str,
                        required=True,
                        help='Path to model configuration file')
    parser.add_argument('-e',
                        '--evaluation-config',
                        type=str,
                        required=True,
                        help='Path to evaluation configuration file')
    parser.add_argument('-p',
                        '--params-config',
                        type=str,
                        required=True,
                        help='Path to parameter configuration file')
    parser.add_argument('-s',
                        '--split',
                        type=float,
                        required=True,
                        help='Split fraction to use for training. The other '
                             'fraction will be used for evaluation')
    parser.add_argument('-t',
                        '--training-config',
                        type=str,
                        required=False,
                        default=None,
                        help='Path to training configuration file')
    parser.add_argument('-v',
                        '--validation-split',
                        type=str,
                        required=False,
                        default=None,
                        help="The path to a .yaml file (or the basename without an "
                             "extension if it is located in `./config/data/`) "
                             "containing the configuration for the dataset we want to "
                             "use for validation during training (typically after each "
                             "epoch) or a fraction in the interval (0, 1] indicating "
                             "the fraction of the training set that should be reserved "
                             "for validation")
    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        required=False,
                        default=None,
                        help='Path to output directory')
    parser.add_argument('-a',
                        '--analyze',
                        action='store_true',
                        help='If True, more logging will be performed')
    parser.add_argument('-cv',
                        "--cross-validation",
                        type=int,
                        default=None,
                        help='K-fold cross-validation (e.g., 5). If >1, overrides --validation-split.')
    parser.add_argument('-rs',
                        '--random-search-trials',
                        type=int,
                        default=None,
                        help='Number of random trials. If not set, use full grid.')
    parser.add_argument('-r',
                        '--repeated-grid-search-runs',
                        type=int,
                        default=1,
                        help='Number of times to repeat the entire grid search.')
    parser.add_argument('--seed', type=int, default=42)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(**vars(args))
