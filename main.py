import os
from pathlib import Path

from torch import compile, cuda, device, save
from torch.utils.data import random_split

from artificial_genome_synthesis.manage_data import Genotype, load_data
from artificial_genome_synthesis.models import instanciate_models
from artificial_genome_synthesis.models.trainers import WGANTrainerSetup
from artificial_genome_synthesis.utils.command_line import get_cmd_args
from artificial_genome_synthesis.utils.experiment_tracking import (
    attach_logger, seed_all)
from artificial_genome_synthesis.utils.load_env import config

# # * Get num_workers to work in windows
# if os.name == "nt":
#     os.environ["OMP_NUM_THREADS"] = "1"
#     # import torch.multiprocessing as mp
#     # mp.set_start_method("spawn")


def main():
    params = get_cmd_args()
    gen = seed_all(params.random_seed)

    dev = device("cuda" if cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    genotype, gwas_ssf = load_data(
        genotype_file=config["WHOLE_INPUT_DATA"], 
        gwas_ssf_file=config["GWAS_SSF_FILE"],
    )

    params.latent_size = gwas_ssf.shape[1]

    genotype = Genotype(genotype, gwas_ssf)

    train_subset, val_subset = random_split(
        dataset=genotype,
        lengths=[0.7, 0.3],
        generator=gen,
    )

    train_genotype = Genotype.fromSubset(train_subset).make_dataloader(
        batch_size=params.batch_size,
        num_workers=4,
        generator=gen,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_genotype = Genotype.fromSubset(val_subset).make_dataloader(
        batch_size=params.batch_size,
        num_workers=4,
        generator=gen,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    generator, critic = instanciate_models(train_genotype, params.leaky_alpha)

    if os.name == "posix":
        generator = compile(generator)
        critic = compile(critic)

    print(generator)
    print(critic)

    wgan_trainer_setup = WGANTrainerSetup(
        generator=generator,
        critic=critic,
        parameters=params,
        val_dataloader=val_genotype,
    )

    trainer, evaluator = wgan_trainer_setup.trainer_setup()

    attach_logger(trainer, evaluator, wgan_trainer_setup, params)
    trainer.run(train_genotype, params.epochs)

    model_path = Path(config["MODEL_PATH"])

    save(
        generator,
        model_path / f"generator_{params.wgan_type.value}.pt",
    )

    save(
        critic,
        model_path / f"critic_{params.wgan_type.value}.pt",
    )


if __name__ == "__main__":
        main()
