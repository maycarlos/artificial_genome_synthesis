import torch
import torch.nn.functional as F
from colorama import Fore, init
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Loss

from ...utils.types_ import (
    DataFrame,
    Dataloader,
    Device,
    LossFunction,
    Model,
    Optimizer,
)

init(autoreset=True)


def elbo_loss(x_reconstruct, x_true, miu, sigma):
    """

    :param x_reconstruct:
    :param x_true:
    :param miu:
    :param sigma:
    :return:
    """
    reconstructed_loss = F.binary_cross_entropy(
        x_reconstruct, x_true, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + sigma - miu.pow(2) - sigma.exp())
    return reconstructed_loss + kl_loss


def setup_trainer(model, loss_function, optimizer, device):
    def train_step(engine, batch):

        X_original = batch.to(device)

        X_reconstructed, miu, sigma = model(X_original)

        loss = loss_function(X_reconstructed, X_original, miu, sigma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics = {"VAE Loss": loss}

        engine.state.metrics = metrics

        return metrics

    trainer = Engine(train_step)

    r_bar = "| {n_fmt}/{total_fmt} [{postfix}]"

    ProgressBar(persist=True, bar_format="{l_bar}{bar}" + r_bar).attach(
        engine=trainer, metric_names=["VAE Loss"]
    )

    @trainer.on(Events.STARTED)
    def start_message():
        print(Fore.GREEN + "Begin training" + Fore.RESET)

    @trainer.on(Events.EPOCH_COMPLETED(every=20))
    def generate_images(engine):
        print(Fore.RED + "Create Examples" + Fore.RESET)

    return trainer
