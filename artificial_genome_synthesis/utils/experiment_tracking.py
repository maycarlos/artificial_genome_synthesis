import re

import ignite
import torch
from ignite.contrib.handlers import MLflowLogger
from ignite.engine import Engine, Events

from ..models.trainers import WGANTrainerSetup
from .load_env import config


def seed_all(seed: int) -> torch.Generator:
    """_summary_

    Args
    ---
        seed (int): _description_

    Returns
    ---
        _type_: _description_
    """
    # Seed the random number generators
    print(f"Random Seed: {seed}")
    ignite.utils.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def attach_logger(
    trainer: Engine,
    evaluator: Engine,
    trainer_setup: WGANTrainerSetup,
    h_param,
):
    """

    Args:
        trainer (Engine): _description_
        evaluator (Engine): _description_
        trainer_setup (WGANTrainerSetup): _description_
        hyperparameters (_type_): _description_
    """

    mlrun_path = config["MLRUNS_PATH"]

    versions = {}
    for k, v in config.items():
        if re.match(r"version$", k):
            versions[k] = v

    mlflow_logger = MLflowLogger(mlrun_path)
    mlflow_logger.log_params(
        {
            "Generator": trainer_setup.generator,
            "Critic": trainer_setup.critic,
            "Seed": h_param.random_seed,
            "Batch Size": h_param.batch_size,
            "Discriminator Learning Rate": h_param.discriminator_lr,
            "Generator Learning Rate": h_param.generator_lr,
            "Lambda Gradient Penalty": h_param.lambda_gp,
            "Clipping interval": h_param.clip_val,
            **versions,
        }
    )

    mlflow_logger.attach_output_handler(
        engine=trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="Epoch Training",
        metric_names=["Average Critic Loss", "Average Generator Loss"],
    )

    mlflow_logger.attach_output_handler(
        engine=evaluator,
        event_name=trainer_setup.events,
        tag="Validating",
        metric_names=["Wasserstein Loss"],
    )
