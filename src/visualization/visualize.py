import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from colorama import init, Fore
from ..utils.load_env import ENV_CONFIG
from ..utils.types_ import DataFrame, Device, Model

init(autoreset=True)

plt.rcParams["figure.figsize"] = (12, 10)
plt.style.use("fivethirtyeight")
plt.switch_backend("agg")


RUN_TIME = int(time.time())


print("Experiment nº ",end = "")
print(Fore.GREEN + f"{RUN_TIME}" + Fore.RESET)

default_figure_folder = Path(ENV_CONFIG["FIGURES_FOLDER"])
default_gendata_folder = Path(ENV_CONFIG["GENERATED_DATA_FOLDER"])


def create_artificial(
    n_ags: int,
    latent_size: int,
    generator: Model,
    epoch: int,
    device: Device,
    save_data: bool = True,
    save_location: Path = default_gendata_folder,
) -> DataFrame:
    # Create AGs from the trained model

    with torch.no_grad():

        latent_noise = torch.randn(n_ags, latent_size, device = device)
        artificial_data = generator.generation(latent_noise)

        artificial_data = artificial_data.cpu()

        artificial_data[artificial_data < 0] = 0
        artificial_data = np.rint(artificial_data)
        artificial_df = pd.DataFrame(artificial_data)
        artificial_df = artificial_df.astype(np.int8)

        artificial_df.insert(loc=0, column="Type", value="AG")

        gen_names = [f"AG_{i}" for i in range(len(artificial_df))]
        artificial_df.insert(loc=1, column="ID", value=gen_names)

        artificial_df.columns = [*range(artificial_df.shape[1])]

        if save_data:
            artificial_df.to_csv(
                save_location / f"{epoch}_artificial.hapt",
                sep="\t",
                header=False,
                index=False,
            )

        return artificial_df


# def init_pca(real_data: DataFrame):
#     """
#     Make initial pca plot for the real genome

#     Args:
#         df (_type_): _description_
#     """

#     real_pca = real_data.drop(real_data.columns[:2], axis=1)
#     real_pca.columns = [*range(real_pca.shape[1])]

#     pca = PCA(n_components=2)
#     real = pca.fit_transform(real_pca)

#     fig, ax = plt.subplots(1, 1)

#     ax.scatter(real[:, 0], real[:, 1], c="red", label="Real", s=50, alpha=0.2)

#     ax.legend(loc="best")
#     plt.tight_layout()


def plot_pca(
    real_data: DataFrame,
    artificial_data: DataFrame,
    epoch: int,
    save_fig: bool = True,
    save_location: Path = default_figure_folder,
):

    # prepare the pca dataframe for the real and the artificial genome
    real_pca = real_data.drop(real_data.columns[:2], axis=1)
    # real_pca = real_data.copy()
    real_pca.columns = [*range(real_pca.shape[1])]

    artificial_pca = artificial_data.drop(artificial_data.columns[:2], axis=1)
    artificial_pca.columns = [*range(artificial_pca.shape[1])]

    # Apply PCA to both the genomes and then plot them
    pca = PCA(n_components=2)
    real = pca.fit_transform(real_pca)
    fake = pca.transform(artificial_pca)

    fig, ax = plt.subplots(1, 1)

    ax.scatter(real[:, 0], real[:, 1], c="red", label="Real", s=50, alpha=0.2)
    ax.scatter(
        fake[:, 0], fake[:, 1], c="blue", label="Artificial", s=50, alpha=0.2
    )

    ax.legend(loc="best")
    plt.tight_layout()

    if save_fig:
        fig.savefig(save_location / f"{epoch}_pca_{RUN_TIME}.png", format="png")

    return fig


def plot_losses(
    discriminator_losses: list[float],
    generator_losses: list[float],
    epoch: int,
    save_fig: bool = True,
    save_location: Path = default_figure_folder,
):

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(
        discriminator_losses, linewidth=2, c="orange", label="Discriminator"
    )

    ax[0].set_title("Discriminator Loss Evolution")

    ax[1].plot(generator_losses, linewidth=2, c="blue", label="Generator")
    ax[1].set_title("Generator Loss Evolution")

    hlines = np.arange(0, len(generator_losses) + 1, step=195)

    [ax[0].axvline(x=i, linestyle="--", alpha=0.1) for i in hlines]
    [ax[1].axvline(x=i, linestyle="--", alpha=0.1) for i in hlines]

    plt.suptitle(f"Evolution until epoch nº {epoch}")
    plt.tight_layout()

    if save_fig:
        fig.savefig(
            save_location / f"{epoch}_loss_evolution_{RUN_TIME}.png",
            format="png",
        )
    return fig


def plot_hierachical_cluster():
    pass
