import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

from config_io import (
    dump_config,
    load_dataset,
    load_evaluation_config,
    load_model,
    load_training_config,
)
from DNAnet.models.base import TrainableModel
from utils import add_file_handler_to_logger, prepare_output_file


LOGGER = logging.getLogger('dnanet')


def run(data_config: str,
        model_config: str,
        training_config: str,
        evaluation_config: str,
        nr_folds: int,
        output_dir: Optional[str] = None,
        validation_config: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,):

    if not output_dir:
        output_dir = f'output/crossval_{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'
        prepare_output_file(output_dir)

    log_path = prepare_output_file(os.path.join(output_dir, 'log_crossval.txt'))
    add_file_handler_to_logger(LOGGER, path=log_path)
    LOGGER.info(f"Logs will be written to {log_path}")

    LOGGER.info("Loading dataset...")
    dataset = load_dataset(data_config)

    validation_set = None

    metrics = load_evaluation_config(evaluation_config)

    LOGGER.info(f"Splitting dataset into {nr_folds} folds...")
    splits = dataset.split_k_fold(nr_folds, seed=42)
    agg_results = defaultdict(list)
    for i, (train_set, test_set) in enumerate(splits, start=1):
        output_dir_fold = prepare_output_file(os.path.join(output_dir, f"fold_{i}"))
        LOGGER.info(f"Starting fold {i} of {nr_folds}...")

        if validation_config:
            validation_config = float(validation_config)
            LOGGER.info(f"Validation split found, using {validation_config * 100}% to create a "
                        f"validation set...")
            train_set, validation_set = train_set.split(1 - validation_config)

        LOGGER.info("Loading model...")
        model = load_model(model_config)

        if not isinstance(model, TrainableModel):
            # Ensure the model has a .fit() method
            raise ValueError(f"Model {model} is not trainable.")

        if checkpoint_dir:
            model.load(checkpoint_dir)
            LOGGER.info(f"Loading previous model checkpoint from {checkpoint_dir}")
        else:
            LOGGER.info("Will start training from scratch")

        training_kwargs = load_training_config(training_config)
        training_kwargs.update({'validation_set': validation_set,
                                'checkpoint_dir': output_dir_fold})

        LOGGER.info("Starting training...")
        try:
            model.fit(train_set, **training_kwargs)
        except KeyboardInterrupt:
            LOGGER.info("Training interrupted!")

        LOGGER.info("Applying trained model...")
        predictions = model.predict_batch(test_set)

        results = {}
        if evaluation_config:
            LOGGER.info("Computing metrics...")
            for name, func in metrics.items():
                name = f"{name} ({str(func.keywords)})" if func.keywords else name
                value = func(images=test_set, predictions=predictions)
                results[name] = value
                if isinstance(value, float):
                    agg_results[name].append(value)

            results = json.dumps(results, indent=4)
            metrics_path = os.path.join(output_dir_fold, "metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write(results)
            LOGGER.info(f"Results of fold {i}: \n {results}")
            LOGGER.info(f"Results written to {metrics_path}")

        # Write the config to the log directory, so we can always retrace which
        # arguments were used.
        config_path = os.path.join(output_dir_fold, 'config.yaml')
        dump_config(config_path, data_config, model_config, training_config,
                    validation_config, dataset, model)
        LOGGER.info(f"Config written to {config_path}")

    metrics_path = os.path.join(output_dir, "agg_metrics.txt")
    with open(metrics_path, 'w') as f:
        for metric_name, values in agg_results.items():
            print_result = f"Mean {metric_name}: {sum(values) / len(values)}"
            LOGGER.info(print_result)
        f.write(print_result)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data-config',
        type=str,
        required=True,
        help="The path to a .yaml file (or the basename without an extension "
             "if located in `./config/data/`) containing the configuration "
             "for the dataset to train the model on"
    )
    parser.add_argument(
        '-m',
        '--model-config',
        type=str,
        required=True,
        help="The path to a .yaml file (or the basename without an extension "
             "if located in `./config/models/`) containing the configuration "
             "for the model to be trained"
    )
    parser.add_argument(
        '-t',
        '--training-config',
        type=str,
        required=True,
        help="The path to a .yaml file (or the basename without an "
             "extension if it is located in `./config/training/`) "
             "containing the configuration for the keyword arguments "
             "that will be passed to the model's `fit()` method"
    )
    parser.add_argument(
        '-e',
        '--evaluation-config',
        type=str,
        required=True,
        help="The path to a .yaml file (or the basename without an "
             "extension if it is located in `./config/evaluation/`) "
             "containing metrics to evaluate the model on."
    )
    parser.add_argument(
        '-k',
        '--nr-folds',
        type=int,
        required=True,
        help="The number of folds to split the dataset in."
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        required=False,
        default=None,
        help="Directory to save the trained model to."
    )
    parser.add_argument(
        '-v',
        '--validation-config',
        type=float,
        required=False,
        default=None,
        help="A fraction in the interval (0, 1) indicating "
             "the fraction of the training set that should be reserved "
             "for validation."
    )
    parser.add_argument(
        '-c',
        '--checkpoint-dir',
        type=str,
        required=False,
        default=None,
        help="If provided, training will resume from the model's checkpoint loaded from this "
             "directory."
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(**vars(args))
