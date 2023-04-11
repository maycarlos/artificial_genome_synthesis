import torch
from colorama import Fore, init
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from torch.nn.utils import clip_grad_norm_

from ...utils.types_ import (
    DataFrame,
    Dataloader,
    Device,
    LossFunction,
    Model,
    Optimizer,
)
from ...visualization.visualize import create_artificial, plot_pca

init(autoreset=True)


def trainer_setup(
    generator: Model,
    discriminator: Model,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    loss_function: LossFunction,
    latent_size: int,
    n_ags: int,
    original_data: DataFrame,
    device: Device,
):

    REAL_LABEL = 1.0
    FAKE_LABEL = 0.0

    def train_step(engine, batch):

        generator.train()
        discriminator.train()

        batch_size = len(batch)

        noise = torch.randn(batch_size, latent_size, device=device)

        X_real = batch.cuda(non_blocking=True).float()
        X_fake = generator(noise)

        # * Train Discriminator
        # ! with autocast():
        disc_real = discriminator(X_real)
        disc_fake = discriminator(X_fake)

        uniform_noise = torch.FloatTensor(
            disc_real.shape[0], disc_real.shape[1]
        ).uniform_(0, 0.1)

        uniform_noise = uniform_noise.cuda()

        disc_loss_real = loss_function(
            disc_real, torch.ones_like(disc_real) - uniform_noise
        )
        disc_loss_fake = loss_function(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = disc_loss_real + disc_loss_fake

        discriminator.zero_grad()
        disc_loss.backward(retain_graph=True)
        clip_grad_norm_(discriminator.parameters(), 1)
        discriminator_optimizer.step()

        # * Train Generator
        # ! with autocast():
        output = discriminator(X_fake).view(-1)
        gen_loss = loss_function(output, torch.ones_like(output))

        generator.zero_grad()
        gen_loss.backward()
        clip_grad_norm_(generator.parameters(), 1)
        generator_optimizer.step()

        metrics = {"Discriminator Loss": disc_loss, "Generator Loss": gen_loss}

        engine.state.metrics = metrics

        return metrics

    trainer = Engine(train_step)


    r_bar = "| {n_fmt}/{total_fmt} [{postfix}]"

    ProgressBar(persist=True, bar_format="{l_bar}{bar}" + r_bar).attach(
        engine=trainer, metric_names=["Discriminator Loss", "Generator Loss"]
    )

    @trainer.on(Events.STARTED)
    def start_message():
        print(Fore.GREEN + "Begin training" + Fore.RESET)


    events = Events.EPOCH_COMPLETED(once = 1) | Events.EPOCH_COMPLETED(every = 20) 

    @trainer.on(event_name=events)
    def generate_images(engine):
        print(Fore.RED + "Create Examples" + Fore.RESET)

        generator.eval()

        artificial_data = create_artificial(
            n_ags=n_ags,
            latent_size=latent_size,
            generator=generator,
            device=device,
            epoch=engine.state.epoch
        )

        pca_figure = plot_pca(
            real_data=original_data,
            artificial_data=artificial_data, 
            epoch = engine.state.epoch
        )

    return trainer


