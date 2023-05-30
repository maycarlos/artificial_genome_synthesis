import torch
import torch.nn.functional as F
from colorama import Fore, init
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Loss

from artificial_genome_synthesis.utils.types_ import Device

from ...base.BaseTrainSetup import BaseTrainerSetup
from ...utils.types_ import (
    DataFrame,
    Dataloader,
    Device,
    LossFunction,
    Model,
    Optimizer,
)

init(autoreset=True)


class VAETrainerSetup(BaseTrainerSetup):
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        val_dataloader,
        latent_size: int,
        n_ags: int,
        original_data: DataFrame,
        labels: DataFrame,
        device: Device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.val_dataloader = val_dataloader
        self.latent_size = latent_size
        self.n_ags = n_ags
        self.original_data = original_data
        self.labels = labels
        self.device = device

    def setup_trainer(self):
        trainer = Engine(self.__train_step)
        evaluator = Engine(self.__validation_step)

        r_bar = "| {n_fmt}/{total_fmt} [{postfix}]"

        ProgressBar(persist=True, bar_format="{l_bar}{bar}" + r_bar).attach(
            engine=trainer, metric_names=["VAE Loss"]
        )

        @trainer.on(Events.STARTED)
        def start_message():
            print(Fore.GREEN + "Begin training" + Fore.RESET)

        events = Events.EPOCH_COMPLETED(once=1) | Events.EPOCH_COMPLETED(every=10)

        @trainer.on(events)
        def run_validation(engine):
            print(Fore.GREEN + "Validation!" + Fore.RESET)
            evaluator.run(self.val_dataloader)

        return trainer

    def __train_step(self, engine, batch):
        X_original = batch.to(self.device)

        X_reconstructed, miu, sigma = self.model(X_original)

        loss = self.elbo_loss(X_reconstructed, X_original, miu, sigma)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {"VAE Loss": loss}

        engine.state.metrics = metrics

        return metrics

    def __validation_step(self, engine, batch):
        pass

    @staticmethod
    def elbo_loss(
        x_reconstruct,
        x_true,
        miu,
        sigma,
    ):
        """

        :param x_reconstruct:
        :param x_true:
        :param miu:
        :param sigma:
        :return:
        """
        reconstructed_loss = F.binary_cross_entropy(
            x_reconstruct, x_true, reduction="sum"
        )
        kl_loss = -0.5 * torch.sum(1 + sigma - miu.pow(2) - sigma.exp())

        return reconstructed_loss + kl_loss
