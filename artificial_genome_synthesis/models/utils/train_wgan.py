"""
set up training of WGAN following the paper: Improved Training of Wasserstein GANs
"""
import numpy as np
import torch
import torch.autograd as autograd
from colorama import Fore, init
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from torch.autograd import Variable

from ...utils.types_ import DataFrame, Device, Model, Optimizer, Tensor
from ...visualization.visualize import (
    create_artificial,
    init_umap,
    plot_pca,
    plot_umap,
)

init(autoreset=True)


def trainer_setup(
    generator: Model,
    critic: Model,
    generator_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    latent_size: int,
    n_ags: int,
    original_data: DataFrame,
    labels,
    device: Device,
    n_critic: int = 5,
    clip_val: float = 0.01,
):

    REAL_LABEL = 1.0
    FAKE_LABEL = 0.0

    def train_step(engine, batch):
        generator.train()
        critic.train()

        for p in critic.parameters():
            p.requires_grad = True

        one = torch.tensor(1, dtype=torch.float).cuda()
        m_one = (one * -1).cuda()

        batch_size = len(batch)

        losses_dict = {"Critic Loss": 0, "Generator Loss": 0}

        # * Train Critic with wasserstein distance
        for _ in range(n_critic):
            critic_optimizer.zero_grad()

            noise = torch.rand((batch_size, latent_size)).cuda()

            X_real = batch.float().cuda()
            X_fake = generator(noise)

            grad_penalty = compute_gradient_penalty(
                critic,
                real_samples=X_real,
                fake_samples=X_fake,
            )

            crit_real = critic(X_real)
            crit_fake = critic(X_fake)

            crit_loss = crit_fake.mean() - crit_real.mean() + grad_penalty

            crit_loss.backward()
            critic_optimizer.step()

            # Clip weights
            for p in critic.parameters():
                p.data.clamp_(-clip_val, clip_val)

            losses_dict["Critic Loss"] = crit_loss.item()

        # * Train generator

        for p in critic.parameters():
            p.requires_grad = False

        generator_optimizer.zero_grad()

        X_fake = generator(noise)

        crit_fake = critic(X_fake)

        g_loss = crit_fake.mean()

        g_loss.backward(m_one)
        generator_optimizer.step()

        losses_dict["Generator Loss"] = g_loss.item()

        engine.state.metrics = losses_dict

        return losses_dict

    trainer = Engine(train_step)

    umap_obj, umap_real = init_umap(original_data, labels)

    r_bar = "| {n_fmt}/{total_fmt} [{postfix}]"

    ProgressBar(persist=True, bar_format="{l_bar}{bar}" + r_bar).attach(
        engine=trainer, metric_names=["Critic Loss", "Generator Loss"]
    )

    @trainer.on(Events.STARTED)
    def start_message():
        print(Fore.GREEN + "Begin training" + Fore.RESET)

    events = Events.EPOCH_COMPLETED(once=1) | Events.EPOCH_COMPLETED(every=20)

    @trainer.on(events)
    def generate_images(engine):
        print(Fore.RED + "Create Examples" + Fore.RESET)

        generator.eval()

        artificial_data = create_artificial(
            n_ags=n_ags,
            latent_size=latent_size,
            generator=generator,
            device=device,
            epoch=engine.state.epoch,
            title="WGAN",
        )

        pca_figure = plot_pca(
            real_data=original_data,
            artificial_data=artificial_data,
            epoch=engine.state.epoch,
            title="WGAN",
        )

        umap_figure = plot_umap(
            real_umap=umap_real,
            artificial_data=artificial_data,
            umap_obj=umap_obj,
            epoch=engine.state.epoch,
            title="WGAN",
        )

    return trainer


def compute_gradient_penalty(
    critic: Model,
    real_samples: Tensor,
    fake_samples: Tensor,
    lambda_gp: int = 10,
):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).cuda()

    # alpha = torch.FloatTensor(
    #     np.random.random((real_samples.size(0), 1, 1, 1))
    # ).cuda()

    # Get random interpolation between real and fake samples
    interpolated = alpha * real_samples + ((1 - alpha) * fake_samples)

    interpolated = Variable(interpolated, requires_grad=True)

    crit_interpolated = critic(interpolated)

    # fake = Variable(
    #     torch.FloatTensor(real_samples.shape[0], 1).cuda().fill_(1.0),
    #     requires_grad=False,
    # )

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=crit_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(crit_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    # gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp

    return gradient_penalty
