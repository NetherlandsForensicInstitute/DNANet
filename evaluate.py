import argparse
import json
import logging
import os
from datetime import datetime
from typing import Optional

from config_io import dump_config, load_dataset, load_evaluation_config, load_model
from utils import add_file_handler_to_logger, prepare_output_file, save_predictions


LOGGER = logging.getLogger('dnanet')


def run(data_config: str,
        model_config: str,
        evaluation_config: Optional[str] = None,
        output_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        save_preds: Optional[bool] = False,
        split: Optional[float] = None,
        seed: Optional[int] = None) -> str:

    if not output_dir:
        output_dir = f'output/evaluate_{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'
        prepare_output_file(output_dir)

    log_path = prepare_output_file(os.path.join(output_dir, 'log_evaluation.txt'))
    add_file_handler_to_logger(LOGGER, path=log_path)
    LOGGER.info(f"Logs will be written to {log_path}")

    LOGGER.info("Loading model...")
    model = load_model(model_config)
    if checkpoint_dir:
        model.load(checkpoint_dir)

    LOGGER.info("Loading dataset...")
    dataset = load_dataset(data_config)

    if split:
        LOGGER.info(f"Splitting dataset, using {split * 100}% for evaluation")
        # use 1-split and a seed to ensure the same splitting is done as during training,
        # but we now take the 'second' dataset for evaluation
        _, dataset = dataset.split(1 - split, seed=seed)

    LOGGER.info("Applying model...")
    predictions = model.predict_batch(dataset)

    results = {}
    if evaluation_config:
        LOGGER.info("Computing metrics...")
        metrics = load_evaluation_config(evaluation_config)
        for name, func in metrics.items():
            name = f"{name} ({str(func.keywords)})" if func.keywords else name
            results[name] = func(images=dataset, predictions=predictions)
        results = json.dumps(results, indent=4)
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(results)
        LOGGER.info(f"Results: \n {results}")
        LOGGER.info(f"Results written to {metrics_path}")

    if save_preds:
        LOGGER.info("Saving predictions to JSON...")
        save_predictions(predictions, os.path.join(output_dir, "predictions.json"))

    # Write the config to the log directory, so we can always retrace which
    # arguments were used.
    config_path = os.path.join(output_dir, "config.yaml")
    dump_config(config_path, data_config, model_config, None, None, dataset, model)
    LOGGER.info(f"Config written to {config_path}")
    return results


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data-config',
        type=str,
        required=True,
        help="The path to a .yaml file (or the basename without an extension "
             "if located in `./config/data/`) containing the configuration "
             "for the dataset to evaluate the model on."
    )
    parser.add_argument(
        '-m',
        '--model-config',
        type=str,
        required=True,
        help="The path to a .yaml file (or the basename without an extension "
             "if located in `./config/models/`) containing the configuration "
             "for the model to be loaded."
    )
    parser.add_argument(
        '-e',
        '--evaluation-config',
        type=str,
        required=False,
        default=None,
        help="The path to a .yaml file (or the basename without an "
             "extension if it is located in `./config/evaluation/`) "
             "containing metrics to evaluate the model on."
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        required=False,
        default=None,
        help="The directory to save the evaluation results to."
    )
    parser.add_argument(
        '-c',
        '--checkpoint-dir',
        type=str,
        required=False,
        default=None,
        help="The directory from where to load the model's checkpoint (directory should "
             "contain a .pt file)."
    )
    parser.add_argument(
        '-p',
        '--save-preds',
        action="store_true",
        help="Whether to save the model predictions as json in the output folder."
    )
    parser.add_argument(
        '-s',
        '--split',
        type=float,
        required=False,
        default=None,
        help="An optional fraction in the interval (0, 1)git. If specified, only "
             "this fraction of the dataset is used for evaluation (typically "
             "used when the remainder was previously used for training)"
    )
    parser.add_argument(
        '-rs',
        '--seed',
        type=int,
        required=False,
        default=None,
        help="A random seed to use for splitting. This can be used during training and evaluation "
             "to ensure the same split."
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(**vars(args))
