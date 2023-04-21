import torch
from colorama import Fore, init
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from torch.nn.utils import clip_grad_norm_
from umap import UMAP

from ...base_classes.BaseTrainSetup import BaseTrainerSetup
from ...utils.types_ import (DataFrame, Dataloader, Device, LossFunction,
                             Model, Optimizer)
from ...visualization.visualize import (create_artificial, init_umap, plot_pca,
                                        plot_umap)

init(autoreset=True)


class GANTrainerSetup(BaseTrainerSetup):
    def __init__(
        self,
        generator: Model,
        discriminator: Model,
        g_optimizer: Optimizer,
        d_optimizer: Optimizer,
        loss_function: LossFunction,
        latent_size: int,
        n_ags: int,
        original_data: DataFrame,
        labels: DataFrame,
        device: Device,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_function = loss_function
        self.latent_size = latent_size
        self.n_ags = n_ags
        self.original_data = original_data
        self.labels = labels
        self.device = device

    def trainer_setup(self):

        trainer = Engine(self.train_step)

        umap_obj, umap_real = init_umap(self.original_data, self.labels)

        r_bar = "| {n_fmt}/{total_fmt} [{postfix}]"

        ProgressBar(persist=True, bar_format="{l_bar}{bar}" + r_bar).attach(
            engine=trainer,
            metric_names=["Discriminator Loss", "Generator Loss"],
        )

        @trainer.on(Events.STARTED)
        def start_message():
            print(Fore.GREEN + "Begin training" + Fore.RESET)

        events = Events.EPOCH_COMPLETED(
            once=1) | Events.EPOCH_COMPLETED(every=20)

        @trainer.on(event_name=events)
        def generate_images(engine):
            print(Fore.RED + "Create Examples" + Fore.RESET)

            self.generator.eval()

            artificial_data = create_artificial(
                n_ags=self.n_ags,
                latent_size=self.latent_size,
                generator=self.generator,
                device=self.device,
                epoch=engine.state.epoch,
                title="GAN",
            )

            pca_figure = plot_pca(
                real_data=self.original_data,
                artificial_data=artificial_data,
                epoch=engine.state.epoch,
                title="GAN",
            )

            umap_figure = plot_umap(
                real_umap=umap_real,
                artificial_data=artificial_data,
                umap_obj=umap_obj,
                epoch=engine.state.epoch,
                title="GAN",
            )

        return trainer

    def train_step(self, engine, batch):

        self.generator.train()
        self.discriminator.train()

        batch_size = len(batch)

        noise = torch.randn(batch_size, self.latent_size, device=self.device)

        X_real = batch.cuda(non_blocking=True).float()
        X_fake = self.generator(noise)

        # * Train Discriminator
        disc_real = self.discriminator(X_real)
        disc_fake = self.discriminator(X_fake)

        uniform_noise = torch.FloatTensor(
            disc_real.shape[0], disc_real.shape[1]
        ).uniform_(0, 0.1)

        uniform_noise = uniform_noise.to(self.device)

        disc_loss_real = self.loss_function(
            disc_real, torch.ones_like(disc_real) - uniform_noise
        )
        disc_loss_fake = self.loss_function(
            disc_fake, torch.zeros_like(disc_fake)
        )

        disc_loss = disc_loss_real + disc_loss_fake

        self.discriminator.zero_grad()
        disc_loss.backward(retain_graph=True)
        clip_grad_norm_(self.discriminator.parameters(), 1)
        self.d_optimizer.step()

        # * Train Generator
        output = self.discriminator(X_fake).view(-1)
        gen_loss = self.loss_function(output, torch.ones_like(output))

        self.generator.zero_grad()
        gen_loss.backward()
        clip_grad_norm_(self.generator.parameters(), 1)
        self.g_optimizer.step()

        metrics = {"Discriminator Loss": disc_loss, "Generator Loss": gen_loss}

        engine.state.metrics = metrics

        return metrics
