"""
set up training of WGAN three different ways:
    Wasserstein GAN - WGAN
    Wasserstein GAN with gradient clipping - WGAN-CP
    Wasserstein GAN with gradient penalty - WGAN-GP
"""
from enum import Enum
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore, init
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from torch import autograd
from torch.autograd import Variable
from torch.optim import Adam, RMSprop

from ...base_classes import BaseTrainerSetup
from ...utils.load_env import ENV_CONFIG
from ...utils.types_ import DataLoader, Model, Tensor
from .metrics import Average

init(autoreset=True)


class WGANTrainerSetup(BaseTrainerSetup):
    def __init__(
        self,
        generator: Model,
        critic: Model,
        wgan_type: Enum,
        hyperparameters,
        val_dataloader: DataLoader,
        latent_size: int,
        save_interval: int,
        n_critic: int = 5,
    ):
        self.generator = generator
        self.critic = critic
        self.wgan_type = wgan_type
        self.val_dataloader = val_dataloader
        self.__latent_size = latent_size
        self.__n_critic = n_critic

        if self.wgan_type.value == "CP":
            self.__init_optimizer_cp(hyperparameters)
            self.__clip_val = hyperparameters.clip_val
        elif self.wgan_type.value == "GP":
            self.__init_optimizer_gp(hyperparameters)
            self.__lambda_gp = hyperparameters.lambda_gp

        self.gen_losses: list = []
        self.crit_losses: list = []

        self.events = Events.EPOCH_COMPLETED(once=1) | Events.EPOCH_COMPLETED(
            every=save_interval
        )

    def trainer_setup(self) -> Tuple[Engine, Engine]:

        trainer = Engine(self.__train_step)
        evaluator = Engine(self.__validation_step)

        def score_function(engine):
            val_loss = engine.state.metrics["Wasserstein Loss"]
            return np.abs(val_loss)

        handler = EarlyStopping(
            patience=5, score_function=score_function, trainer=trainer
        )

        evaluator.add_event_handler(Events.COMPLETED, handler)

        def transform_output(res: dict):
            return res["Critic Loss"], res["Generator Loss"]

        Average(
            loss_fn=lambda x, y: x,
            output_transform=transform_output,
        ).attach(trainer, "Average Critic Loss")

        Average(
            loss_fn=lambda x, y: y,
            output_transform=transform_output,
        ).attach(trainer, "Average Generator Loss")

        Average(
            lambda x, y: x,
        ).attach(evaluator, "Wasserstein Loss")

        r_bar = "| {n_fmt}/{total_fmt} [{postfix}]"

        ProgressBar(
            persist=True,
            bar_format="{l_bar}{bar}" + r_bar,
        ).attach(
            engine=trainer,
            metric_names=["Critic Loss", "Generator Loss"],
        )

        @trainer.on(Events.STARTED)
        def start_message():
            print("Experiment nº ", end="")
            print(Fore.GREEN + f"{ENV_CONFIG['RUN_TIME']}" + Fore.RESET)

            print(Fore.GREEN + "Begin training!" + Fore.RESET)

        @trainer.on(self.events)
        def plot_losses(engine):
            fig, axs = plt.subplots(1, 2)

            axs[0].plot(self.crit_losses, label="Critic Loss", c="blue")
            axs[1].plot(self.gen_losses, label="Generator Loss", c="orange")

            fig.suptitle(
                f"Wasserstein loss evolution until epoch: {engine.state.epoch}"
            )

            save_location = Path(ENV_CONFIG["FIGURES_FOLDER"])

            fig.legend()
            fig.tight_layout()

            fig.savefig(
                save_location
                / f"wgan_{self.wgan_type.value}_{engine.state.epoch}_{ENV_CONFIG['RUN_TIME']}"
            )

        @trainer.on(self.events)
        def run_validation(engine):
            print(Fore.GREEN + "Validation!" + Fore.RESET)
            evaluator.run(self.val_dataloader)

            w_loss = evaluator.state.metrics["Wasserstein Loss"]

            state = f"Epoch: {engine.state.epoch} | Avg Wasserstein Loss: {w_loss:.3f}"
            print(state)

        return trainer, evaluator

    def __train_step(self, engine, batch):
        """Train step from a WGAN with gradient clipping"""

        self.generator.train()
        self.critic.train()

        for param in self.critic.parameters():
            param.requires_grad = True

        one = torch.tensor(1, dtype=torch.float).cuda()
        m_one = (one * -1).cuda()

        noise = None

        if len(batch) == 2:
            noise = batch[1]
            batch = batch[0]

        batch_size = len(batch)

        losses_dict = {"Critic Loss": 0, "Generator Loss": 0}

        # * Train Critic with wasserstein distance
        for _ in range(self.__n_critic):
            crit_loss_mean = []
            self.c_optimizer.zero_grad()

            if noise is None:
                noise = torch.rand((batch_size, self.__latent_size))

            noise = noise.float().cuda()

            real = batch.float().cuda()
            fake = self.generator(noise)

            crit_real = self.critic(real)
            crit_fake = self.critic(fake)

            crit_loss = crit_fake.mean() - crit_real.mean()

            if self.wgan_type.value == "GP":
                grad_penalty = self.__compute_gradient_penalty(
                    real_samples=real,
                    fake_samples=fake,
                    lambda_gp=self.__lambda_gp,
                )

                crit_loss += grad_penalty

            crit_loss.backward()
            self.c_optimizer.step()

            if self.wgan_type.value == "CP":
                # Clip weights
                for param in self.critic.parameters():
                    param.data.clamp_(-self.__clip_val, self.__clip_val)

            crit_loss_mean.append(crit_loss.item())

        losses_dict["Critic Loss"] = np.mean(crit_loss_mean)

        # * Train Generator
        for param in self.critic.parameters():
            param.requires_grad = False

        self.g_optimizer.zero_grad()

        fake = self.generator(noise)

        crit_fake = self.critic(fake)

        gen_loss = crit_fake.mean()

        gen_loss.backward(m_one)

        self.g_optimizer.step()

        losses_dict["Generator Loss"] = gen_loss.item()

        self.gen_losses.append(losses_dict["Generator Loss"])
        self.crit_losses.append(losses_dict["Critic Loss"])

        engine.state.metrics = losses_dict

        return losses_dict

    def __validation_step(self, engine, batch):
        self.generator.eval()
        self.critic.eval()

        noise = None

        if len(batch) == 2:
            noise = batch[1]
            batch = batch[0]

        batch_size = len(batch)

        if noise is None:
            noise = torch.randn((batch_size, self.__latent_size))

        with torch.no_grad():
            real = batch.float().cuda()
            noise = noise.float().cuda()

            fake = self.generator(noise)

            crit_real = self.critic(real)
            crit_fake = self.critic(fake)

            w_loss = crit_fake.mean() - crit_real.mean()

        # engine.state.metrics = w_loss.item()

        return (w_loss, 0)

    def __compute_gradient_penalty(
        self,
        real_samples: Tensor,
        fake_samples: Tensor,
        lambda_gp: float,
    ):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1)).cuda()

        # Get random interpolation between real and fake samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        interpolated = Variable(interpolated, requires_grad=True)

        crit_interpolated = self.critic(interpolated)

        fake = Variable(
            torch.ones_like(crit_interpolated),
            requires_grad=False,
        )

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=crit_interpolated,
            inputs=interpolated,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1) ** 2
        ).mean() * lambda_gp

        return gradient_penalty

    def __init_optimizer_cp(self, hyperparameters):
        """
        WGAN - CP parametes

        Args
        ---
            hyperparameters: Hyperparaters to use for the optimizers

        """
        self.c_optimizer = RMSprop(
            params=self.critic.parameters(),
            lr=hyperparameters.discriminator_lr,
            weight_decay=hyperparameters.l2_pen,
        )

        self.g_optimizer = RMSprop(
            params=self.generator.parameters(),
            lr=hyperparameters.generator_lr,
            # weight_decay = hyperparameters.l2_penalty,
        )

    def __init_optimizer_gp(self, hyperparameters):
        """
        Definition of the optimizers following the paper :''

        Args
        ---
            hyperparameters: Hyperparaters to use for the optimizers
        """
        self.c_optimizer = Adam(
            params=self.critic.parameters(),
            lr=hyperparameters.discriminator_lr,
            betas=(hyperparameters.beta1, hyperparameters.beta2),
        )

        self.g_optimizer = Adam(
            params=self.generator.parameters(),
            lr=hyperparameters.generator_lr,
            betas=(hyperparameters.beta1, hyperparameters.beta2),
        )
