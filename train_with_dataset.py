import logging
import os
from datetime import datetime
from typing import Optional

import mlflow

from DNAnet.data.data_models.base import InMemoryDataset
from config_io import dump_config, load_config, load_model, load_training_config
from DNAnet.models.base import TrainableModel
from utils import add_file_handler_to_logger, prepare_output_file

LOGGER = logging.getLogger("dnanet")


def run_with_dataset(
    dataset: InMemoryDataset,
    model_config: str,
    training_config: str,
    output_dir: Optional[str] = None,
    validation_dataset: Optional[InMemoryDataset] = None,
    checkpoint_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Train a model using a pre-instantiated dataset (instead of a data config).
    """
    if not output_dir:
        output_dir = f'output/train_{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'
        prepare_output_file(output_dir)

    log_path = prepare_output_file(os.path.join(output_dir, "log_training.txt"))
    add_file_handler_to_logger(LOGGER, path=log_path)
    LOGGER.info(f"Logs will be written to {log_path}")

    if mlflow_config := load_config(training_config, kind="training").get("mlflow", None):
        LOGGER.info("Configuring mlflow")
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.start_run(run_name=mlflow_config.get("run_name"), experiment_id=mlflow_config.experiment_id)
        mlflow.autolog()

    LOGGER.info("Loading model...")
    model = load_model(model_config)
    if not isinstance(model, TrainableModel):
        raise ValueError(f"Model {model} is not trainable.")

    if checkpoint_dir:
        model.load(checkpoint_dir)
        LOGGER.info(f"Loading previous model checkpoint from {checkpoint_dir}")
    else:
        LOGGER.info("Will start training from scratch")


    training_kwargs = load_training_config(training_config)
    training_kwargs.update({"validation_set": validation_dataset, "checkpoint_dir": output_dir})

    LOGGER.info("Starting training...")
    try:
        model.fit(dataset, **training_kwargs)
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted!")

    LOGGER.info("Saving model...")
    model.save(output_dir)
    LOGGER.info(f"Saved model to {output_dir}")

    # Write the run metadata (including dataset serialization) to the log directory.
    config_path = os.path.join(output_dir, "config.yaml")
    output_config = dump_config(
        config_path,
        data_config_path="",
        model_config_path=model_config,
        training_config_path=training_config,
        validation_config=None,
        dataset=dataset,
        model=model,
    )
    LOGGER.info(f"Config written to {config_path}")

    if mlflow_config:
        mlflow.log_params(output_config.get("data", {}))


__all__ = ["run_with_dataset"]
