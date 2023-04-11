import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import dotenv_values, find_dotenv
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..GANs.gan_model import Discriminator, Generator
from ..GANs.train_loop import train_loop
from ...data import GenotypeData
from ...visualization.visualize import create_artificial, plot_losses, plot_pca


def default_pca_fig():
    pass


def set_reproductible(random_seed: int):
    """
    Make out randomizers to the seed provided

    Args:
        random_seed (int)

    Returns:
        g: Torch Generator for the dataloader
    """

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    g = torch.Generator()
    g.manual_seed(random_seed)

    print(f"Random Seed: {random_seed}")

    return g


def instanciate_models(*args, **kwargs):

    discriminator = Discriminator(
        features=kwargs["features"], alpha=kwargs["alpha"]
    ).to(kwargs["device"])

    generator = Generator(
        latent_size=kwargs["latent_size"],
        features=kwargs["features"],
        alpha=kwargs["alpha"],
    ).to(kwargs["device"])

    print(discriminator)

    print(generator)

    return discriminator, generator


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dotenv_file = find_dotenv()
    env_values = dotenv_values(dotenv_file)
    g = set_reproductible(int(env_values["RANDOM_SEED"]))

    # * Data
    # todo o valor the path vai mudar para ser fornecido pela command line
    file_path = "../../../data/processed/control_genotype.hapt"
    df = pd.csv(file_path, sep="\t").iloc[:, :7502]

    genotype = GenotypeData(df)

    genotype_dataloader = DataLoader(
        dataset=genotype,
        batch_size=int(env_values["BATCH_SIZE"]),
        shuffle=True,
        drop_last=True,
        generator=g,
        num_workers=4,
        pin_memory=True,
    )

    X = next(iter(genotype_dataloader))

    # * Models
    discriminator, generator = instanciate_models(
        features=X.shape[1],
        alpha=float(env_values["LEAKY_RELU_ALPHA"]),
        latent_size=int(env_values["LATENT_SIZE"]),
        device=device,
    )

    # * Optimizers
    disc_optimizer = Adam(
        params=discriminator.parameters(),
        lr=float(env_values["DISCRIMINATOR_LEARNING_RATE"]),
        weight_decay=float(env_values["L2_PENALTY"]),
    )

    gen_optimizer = Adam(
        params=generator.parameters(),
        lr=float(env_values["GENERATOR_LEARNING_RATE"]),
        weight_decay=float(env_values["L2_PENALTY"]),
    )

    loss = BCELoss()

    # * training

    real_label = 1.0
    fake_label = 0.0

    greater_gen_losses = []
    greater_disc_losses = []

    with mlflow.start_run() as current_run:
        epochs = int(dotenv_values["EPOCHS"])
        latent_size = int(env_values["LATENT_SIZE"])
        save_that = int(env_values["SAVE_THAT"])
        ag_size = int(env_values["AG_SIZE"])

        for epoch in range(epochs + 1):
            print(f"#### Epoch: {epoch + 1} ####")

            gen_losses, disc_losses = train_loop(
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=gen_optimizer,
                discriminator_optimizer=disc_optimizer,
                dataloader=genotype_dataloader,
                loss_function_1=loss,
                loss_function_2=loss,
                latent_size=latent_size,
                device=device,
            )

            greater_disc_losses.extend(disc_losses)
            greater_gen_losses.extend(gen_losses)

            # Summary writer for quick visualization to the model learning evolution
            mlflow.log_metric(
                "Generator Average Epoch Loss", np.mean(gen_losses), step=epoch
            )
            mlflow.log_metric(
                "Discriminator Average Epoch Loss", np.mean(disc_losses), step=epoch
            )

            # * save examples to check in intervals defined by save_that
            if epoch % save_that == 0 or epoch == epochs:

                loss_figures = plot_losses(
                    greater_disc_losses, greater_gen_losses, epoch=epoch
                )

                mlflow.log_figure(loss_figures, f"loss_plot_{epoch}.png")

                artificial_genome = create_artificial(
                    ags=ag_size,
                    latent_size=latent_size,
                    generator=generator,
                    device=device,
                )

                artificial_genome.to_csv(
                    f"../reports/{epoch}_output.hapt",
                    sep="\t",
                    header=False,
                    index=False,
                )

                pca_figure = plot_pca(df, artificial_genome, epoch)
                mlflow.log_figure(pca_figure, f"pca_figure_{epoch}.png")


if __name__ == "__main__":
    main()
