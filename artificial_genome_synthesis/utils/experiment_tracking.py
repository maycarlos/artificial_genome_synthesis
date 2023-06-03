import re

import ignite
import torch
from ignite.contrib.handlers import MLflowLogger
from ignite.engine import Engine, Events

from ..models.trainers import WGANTrainerSetup
from . import load_env


def seed_all(seed):
    """_summary_

    Args:
        seed (_type_): _description_

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
    trainer: Engine, evaluator: Engine, trainer_setup: WGANTrainerSetup, hyperparameters
):
    mlrun_path = load_env.ENV_CONFIG["MLRUNS_PATH"]

    versions = {}
    for k, v in load_env.ENV_CONFIG.items():
        if re.match(r"version$", k):
            versions[k] = v

    mlflow_logger = MLflowLogger(mlrun_path)
    mlflow_logger.log_params(
        {
            "Generator": trainer_setup.generator,
            "Critic": trainer_setup.critic,
            "Seed": hyperparameters.random_seed,
            "Batch Size": hyperparameters.batch_size,
            "Discriminator Learning Rate": hyperparameters.discriminator_lr,
            "Generator Learning Rate": hyperparameters.generator_lr,
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
