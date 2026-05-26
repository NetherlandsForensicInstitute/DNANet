import argparse
import logging
import os
from datetime import datetime
from typing import Optional, Union

import mlflow

from config_io import dump_config, load_config, load_dataset, load_model, load_training_config
from DNAnet.models.base import TrainableModel
from utils import add_file_handler_to_logger, prepare_output_file


LOGGER = logging.getLogger('dnanet')


def run(data_config: str,
        model_config: str,
        training_config: str,
        split: Optional[float] = None,
        output_dir: Optional[str] = None,
        validation_config: Optional[Union[str, float]] = None,
        checkpoint_dir: Optional[str] = None,
        seed: Optional[int] = None):

    if not output_dir:
        output_dir = f'output/train_{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'
        prepare_output_file(output_dir)

    log_path = prepare_output_file(os.path.join(output_dir, 'log_training.txt'))
    add_file_handler_to_logger(LOGGER, path=log_path)
    LOGGER.info(f"Logs will be written to {log_path}")

    if mlflow_config := load_config(training_config, kind='training').get('mlflow', None):
        LOGGER.info("Configuring mlflow")
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.start_run(run_name=mlflow_config.get('run_name'),
                         experiment_id=mlflow_config.experiment_id)
        mlflow.autolog()

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

    LOGGER.info("Loading dataset...")
    dataset = load_dataset(data_config)
    if split:
        LOGGER.info(f"Splitting dataset, using {split * 100}% for training")
        dataset, _ = dataset.split(split, seed=seed)

    validation_set = None
    if validation_config:
        try:
            validation_config = float(validation_config)
            LOGGER.info(f"Validation split found, using {validation_config * 100}% to create a "
                        f"validation set...")
            dataset, validation_set = dataset.split(1 - validation_config)
        except ValueError:
            LOGGER.info("Validation config found, loading validation set...")
            validation_set = load_dataset(validation_config)

    training_kwargs = load_training_config(training_config)
    training_kwargs.update({'validation_set': validation_set,
                            'checkpoint_dir': output_dir})
    LOGGER.info("Starting training...")
    try:
        model.fit(dataset, **training_kwargs)
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted!")

    LOGGER.info("Saving model...")
    model.save(output_dir)
    LOGGER.info(f"Saved model to {output_dir}")

    # Write the config to the log directory, so we can always retrace which
    # arguments were used.
    config_path = os.path.join(output_dir, 'config.yaml')
    output_config = dump_config(config_path, data_config, model_config, training_config,
                                validation_config, dataset, model)
    LOGGER.info(f"Config written to {config_path}")

    if mlflow_config:
        mlflow.log_params(
            dict(**output_config, **{'split': split, 'checkpoint_dir': checkpoint_dir})
        )


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
        '-s',
        '--split',
        type=float,
        required=False,
        default=None,
        help="An optional fraction in the interval (0, 1). If specified, only "
             "this fraction of the dataset is used for training (leaving the "
             "remainder for testing later on)"
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
        type=str,
        required=False,
        default=None,
        help="The path to a .yaml file (or the basename without an "
             "extension if it is located in `./config/data/`) "
             "containing the configuration for the validation dataset, "
             "or a fraction in the interval (0, 1) indicating "
             "the fraction of the training set that should be reserved "
             "for validation"
    )
    parser.add_argument(
        '-c',
        '--checkpoint-dir',
        type=str,
        required=False,
        default=None,
        help="If provided, training will resume from the model checkpoint loaded from this "
             "directory."
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
