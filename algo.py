import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import ignite
import numpy as np
import pandas as pd
import torch
from ignite.contrib.handlers import MLflowLogger
from ignite.engine import Events, Engine
from torch.utils.data import random_split

from artificial_genome_synthesis.manage_data import Genotype
from artificial_genome_synthesis.models.GANs import WGAN_Critic, WGAN_Generator
from artificial_genome_synthesis.models.trainers import WGANTrainerSetup
from artificial_genome_synthesis.utils import SelectWGAN, load_env

mlrun_path = Path(load_env.ENV_CONFIG["MLRUNS_PATH"])
# os.environ["MLFLOW_TRACKING_URI"] = mlrun_path.as_uri()


@dataclass
class HyperParameters:
    random_seed: int = int(load_env.ENV_CONFIG["RANDOM_SEED"])
    epochs: int = 100

    # discriminator learning rate
    discriminator_learning_rate: float = 5e-4

    # Generator learning rate
    generator_learning_rate: float = 1e-4

    batch_size: int = 64
    l2_penalty: float = 1e-3

    # size of noise input
    latent_size: int = 600

    # alpha value for LeakyReLU
    alpha: float = 0.01
    save_interval: int = 5
    adam_betas: tuple[float, float] = (
        0.5,
        0.9,
    )
    clip_val: float = 0.01
    lambda_gp: float = 20
    n_critic: int = 5


HP = HyperParameters()


def load_data():
    gwas_ssf = pd.read_csv(
        load_env.ENV_CONFIG["GWAS_SSF_FILE"], sep="\t", dtype_backend="pyarrow"
    ).iloc[:, :]

    genotype_file = (
        pd.read_pickle(load_env.ENV_CONFIG["WHOLE_INPUT_DATA"]).iloc[:5000, :].T
    )

    columns_of_interest = [
        "odds_ratio",
        "standard_error",
        "effect_allele_frequency",
        "neg_log_10_p_value",
    ]

    gwas_ssf = gwas_ssf[columns_of_interest].astype(np.float16)

    genotype = (
        genotype_file.pipe(lambda x: x / 2)
        .pipe(lambda x: x - np.random.uniform(0, 0.1, size=x.shape))
        .astype(np.float16)
    )

    return genotype, gwas_ssf


def seed_all(seed):
    # Seed the random number generators
    print(f"Random Seed: {seed}")
    ignite.utils.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def instanciate_models(dataloader):
    X, y = next(iter(dataloader))

    generator = WGAN_Generator(
        latent_size=y.shape[1],
        features=X.shape[1],
        alpha=HP.alpha,
    ).cuda()

    critic = WGAN_Critic(features=X.shape[1], alpha=HP.alpha).cuda()

    return generator, critic


def attach_logger(trainer: Engine, evaluator: Engine, trainer_setup: WGANTrainerSetup):
    mlrun_path = load_env.ENV_CONFIG["MLRUNS_PATH"]

    mlflow_logger = MLflowLogger(mlrun_path)
    mlflow_logger.log_params(
        {
            "Seed": 123,
            "Batch Size": HP.batch_size,
            "Generator": trainer_setup.generator,
            "Critic": trainer_setup.critic,
            "Torch version": torch.__version__,
            "Cuda version": torch.version.cuda,
            "Ignite version": ignite.__version__,
            "Discriminator Learning Rate": HP.discriminator_learning_rate,
            "Generator Learning Rate": HP.generator_learning_rate,
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


def main():
    g = seed_all(HP.random_seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    genotype, gwas_ssf = load_data()
    latent_size = gwas_ssf.shape[1]

    genotype = Genotype(genotype, gwas_ssf)

    train_subset, val_subset = random_split(
        dataset=genotype,
        lengths=[0.7, 0.3],
        generator=g,
    )

    train_genotype = Genotype.fromSubset(train_subset).make_dataloader(
        batch_size=32,
        num_workers=0,
        generator=g,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_genotype = Genotype.fromSubset(val_subset).make_dataloader(
        batch_size=32,
        num_workers=0,
        generator=g,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    generator, critic = instanciate_models(train_genotype)

    if os.name == "posix":
        generator = torch.compile(generator)
        critic = torch.compile(critic)

    print(generator)
    print(critic)

    wgan_trainer_setup = WGANTrainerSetup(
        generator=generator,
        critic=critic,
        wgan_type=SelectWGAN.GP,
        hyperparameters=HP,
        val_dataloader=val_genotype,
        latent_size=latent_size,
        save_interval=HP.save_interval,
        n_critic=HP.n_critic,
        lambda_gp=HP.lambda_gp,
    )

    trainer, evaluator = wgan_trainer_setup.trainer_setup()

    attach_logger(trainer, evaluator, wgan_trainer_setup)
    trainer.run(train_genotype, HP.epochs)

    model_path = Path(load_env.ENV_CONFIG["MODEL_PATH"])

    torch.save(generator, model_path / "generator.pt")
    torch.save(critic, model_path / "critic.pt")


if __name__ == "__main__":
    main()
