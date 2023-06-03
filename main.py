import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

from artificial_genome_synthesis.manage_data import Genotype
from artificial_genome_synthesis.models import instanciate_models
from artificial_genome_synthesis.models.trainers import WGANTrainerSetup
from artificial_genome_synthesis.utils import SelectWGAN, load_env
from artificial_genome_synthesis.utils.experiment_tracking import (
    attach_logger, seed_all)

mlrun_path = Path(load_env.ENV_CONFIG["MLRUNS_PATH"])


# os.environ["MLFLOW_TRACKING_URI"] = mlrun_path.as_uri()


def get_cmd_args():
    """
    Retrieves the command line arguments for the training
    """
    parser = argparse.ArgumentParser("artificial genome synthesis")

    train_parser = parser.add_argument_group("train")

    wgan_type = {"cp": SelectWGAN.CP, "gp": SelectWGAN.GP}

    train_parser.add_argument(
        "--wgan_type",
        type=lambda x: wgan_type[x],
        default=wgan_type["gp"],
        choices=["cp", "gp"],
        help="Select the type of training for the WGAN",
    )

    train_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
        help="Number of epochs of training",
    )

    train_parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=32,
        help="Batch size for the dataloader",
    )

    train_parser.add_argument(
        "--num_workers",
        "-a",
        type=int,
        default=1,
        help="Paralelize the dataloaders for faster load"
    )

    train_parser.add_argument(
        "--l2_pen",
        "-lp",
        type=float,
        default=1e-3,
        help="L2 Regularization",
    )

    train_parser.add_argument(
        "--generator_lr",
        "-glr",
        type=float,
        default=1e-4,
        help="Learning rate for the generator",
    )

    train_parser.add_argument(
        "--discriminator_lr",
        "-dlr",
        type=float,
        default=4e-4,
        help="Learning rate for the discriminator",
    )

    train_parser.add_argument(
        "--leaky_alpha",
        "-la",
        type=float,
        default=0.01,
        help="Leaky RElu alpha",
    )

    train_parser.add_argument(
        "--beta1",
        "-b1",
        type=float,
        default=0.5,
        help="First beta for the adam optimizer"
    )
    
    train_parser.add_argument(
        "--beta2",
        "-b2",
        type=float,
        default=0.999,
        help="Second beta for the adam optimizer"
    )

    train_parser.add_argument(
        "--n_critic",
        "-nc",
        type=int,
        default=5,
        help="Number of time the critic is trained"
    )

    train_parser.add_argument(
        "--clip_val",
        "-cv",
        type=float,
        default=1.0,
        help="Value for gradient clipping",
    )

    train_parser.add_argument(
        "--lambda_gp",
        "-lgp",
        type=float,
        default=10,
        help="lambda for wgan gp",
    )

    train_parser.add_argument(
        "--save_interval", "-s",
        type=int, default=10,
        help="save interval",
    )

    train_parser.add_argument(
        "--random_seed",
        type=int, default=123,
        help="seed random number generator",
    )

    params = parser.parse_args()

    return params


def load_data():
    # TODO mandar isto para o manage data
    """
    Loads the data
    """
    gwas_ssf = pd.read_csv(
        load_env.ENV_CONFIG["GWAS_SSF_FILE"],
        sep="\t", dtype_backend="pyarrow",
    ).iloc[:, :]

    genotype_file = (
        pd.read_pickle(
            load_env
            .ENV_CONFIG["WHOLE_INPUT_DATA"])
        .iloc[:5000, :]
        .T
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


def main():
    params = get_cmd_args()
    gen = seed_all(params.random_seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    genotype, gwas_ssf = load_data()
    latent_size = gwas_ssf.shape[1]

    genotype = Genotype(genotype, gwas_ssf)

    train_subset, val_subset = random_split(
        dataset=genotype,
        lengths=[0.7, 0.3],
        generator=gen,
    )

    train_genotype = Genotype.fromSubset(train_subset).make_dataloader(
        batch_size=32,
        num_workers=0,
        generator=gen,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_genotype = Genotype.fromSubset(val_subset).make_dataloader(
        batch_size=32,
        num_workers=0,
        generator=gen,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    generator, critic = instanciate_models(train_genotype, params.leaky_alpha)

    if os.name == "posix":
        generator = torch.compile(generator)
        critic = torch.compile(critic)

    print(generator)
    print(critic)

    wgan_trainer_setup = WGANTrainerSetup(
        generator=generator,
        critic=critic,
        wgan_type=params.wgan_type,
        hyperparameters=params,
        val_dataloader=val_genotype,
        latent_size=latent_size,
        save_interval=params.save_interval,
        n_critic=params.n_critic,
    )

    trainer, evaluator = wgan_trainer_setup.trainer_setup()

    attach_logger(trainer, evaluator, wgan_trainer_setup, params)
    trainer.run(train_genotype, params.epochs)

    model_path = Path(load_env.ENV_CONFIG["MODEL_PATH"])

    torch.save(
        generator,
        model_path / f"generator_{params.wgan_type.value}.pt",
    )

    torch.save(
        critic,
        model_path / f"critic_{params.wgan_type.value}.pt",
    )


if __name__ == "__main__":
    main()
