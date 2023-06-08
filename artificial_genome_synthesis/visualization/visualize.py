from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from colorama import Fore, init
from sklearn.decomposition import PCA
from umap import UMAP

from ..utils.load_env import config
from ..utils.types_ import Array, DataFrame, Device, Model, Tensor

init(autoreset=True)

RUN_TIME = config["RUN_TIME"]

plt.rcParams["figure.figsize"] = (12, 10)
plt.style.use("fivethirtyeight")
plt.switch_backend("agg")


default_figure_folder = Path(config["FIGURES_FOLDER"])
default_gendata_folder = Path(config["GENERATED_DATA_FOLDER"])


def create_artificial(
    n_ags: int,
    latent_size: int,
    generator: Model,
    epoch: int,
    title: str,
    device: Device,
    noise: Optional[Tensor] = None,
    save_data: bool = True,
    save_location: Path = default_gendata_folder,
) -> DataFrame:
    # Create AGs from the trained model

    if noise is None:
        noise = torch.randn(n_ags, latent_size)
    noise = noise.cuda()

    with torch.no_grad():
        artificial_data = generator.generation(noise)

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
                save_location / f"{epoch:03d}_artificial_{title}.hapt",
                sep="\t",
                header=False,
                index=False,
            )

        return artificial_df


def plot_umap(
    real_umap: Array,
    artificial_data: DataFrame,
    umap_obj,
    epoch: int,
    title: str,
    save_fig: Optional[bool] = True,
    save_location: Optional[Path] = default_figure_folder,
):
    real_umap = real_umap.copy()

    artificial_umap = artificial_data.drop(
        artificial_data.columns[:2],
        axis=1,
    )
    artificial_umap.columns = [*range(artificial_umap.shape[1])]

    fake_umap = umap_obj.transform(artificial_umap)

    fig, ax = plt.subplots(1, 1)

    ax.scatter(
        x=real_umap[:, 0],
        y=real_umap[:, 1],
        c="red",
        label="Real",
        s=50,
        alpha=0.2,
    )

    ax.scatter(
        x=fake_umap[:, 0],
        y=fake_umap[:, 1],
        c="blue",
        label="Fake",
        s=50,
        alpha=0.2,
    )

    ax.set_title(title + f" Epoch: {epoch:03d}")
    ax.legend(loc="best")
    plt.tight_layout()

    if save_fig:
        fig.savefig(
            save_location / f"{title}_{epoch:03d}_umap_{RUN_TIME}.png",
            format="png",
        )

    return fig


def plot_pca(
    real_data: DataFrame,
    artificial_data: DataFrame,
    epoch: int,
    title: str,
    save_fig: Optional[bool] = True,
    save_location: Optional[Path] = default_figure_folder,
):
    # prepare the pca dataframe for the real and the artificial genome
    # real_pca = real_data.drop(real_data.columns[:2], axis=1)
    real_pca: DataFrame = real_data.copy()
    real_pca.columns = [*range(real_pca.shape[1])]

    artificial_pca = artificial_data.drop(
        artificial_data.columns[:2],
        axis=1,
    )
    artificial_pca.columns = [*range(artificial_pca.shape[1])]

    # Apply PCA to both the genomes and then plot them
    pca = PCA(n_components=2)
    real = pca.fit_transform(real_pca)
    fake = pca.transform(artificial_pca)

    fig, ax = plt.subplots(1, 1)

    ax.scatter(
        real[:, 0],
        real[:, 1],
        c="red",
        label="Real",
        s=50,
        alpha=0.2,
    )
    ax.scatter(
        fake[:, 0],
        fake[:, 1],
        c="blue",
        label="Artificial",
        s=50,
        alpha=0.2,
    )

    ax.set_title(title + f" Epoch: {epoch}")
    ax.legend(loc="best")
    plt.tight_layout()

    if save_fig:
        fig.savefig(
            save_location / f"{title}_{epoch:03d}_pca_{RUN_TIME}.png",
            format="png",
        )

    return fig


def plot_losses(
    discriminator_losses: List[float],
    generator_losses: List[float],
    epoch: int,
    save_fig: bool = True,
    save_location: Path = default_figure_folder,
):
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(discriminator_losses, linewidth=2, c="orange", label="Discriminator")

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


def init_umap(original_data: DataFrame, labels: DataFrame):
    print(Fore.GREEN + "Initializing UMAP" + Fore.RESET)

    umap_object = UMAP(
        n_components=2,
        random_state=123,
        n_jobs=8,
    )

    umaped_real = umap_object.fit_transform(original_data, labels)

    return umap_object, umaped_real
