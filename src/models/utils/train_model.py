import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from ..utils.types_ import Dataloader, Device, LossFunction, Model, Optimizer


# * weight_decay é L2 penalty


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def train_loop(
    generator: Model,
    discriminator: Model,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    dataloader: Dataloader,
    loss_function_1: LossFunction,
    loss_function_2: LossFunction,
    latent_size: torch.int,
    device: Device,
):
    gen_losses = []
    disc_losses = []

    r_bar = "| {n_fmt}/{total_fmt} [{postfix}]"

    loop = tqdm(
        iterable=enumerate(dataloader),
        total=len(dataloader),
        bar_format="{l_bar}{bar}" + r_bar,
    )

    for i, X in loop:
        # todo adicionar noise para estes treinos

        batch_size = len(X)

        noise = torch.randn(batch_size, latent_size, device=device)

        X_real = X.cuda(non_blocking=True).float()
        X_fake = generator(noise)

        # * Train Discriminator

        disc_real = discriminator(X_real)
        disc_fake = discriminator(X_fake)

        uniform_noise = torch.FloatTensor(
            disc_real.shape[0], disc_real.shape[1]
        ).uniform_(0, 0.1)

        uniform_noise = uniform_noise.cuda()

        disc_loss_real = loss_function_1(
            disc_real, torch.ones_like(disc_real) - uniform_noise
        )
        disc_loss_fake = loss_function_1(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = disc_loss_real + disc_loss_fake

        discriminator.zero_grad()
        disc_loss.backward(retain_graph=True)
        clip_grad_norm_(discriminator.parameters(), 1)
        discriminator_optimizer.step()

        # * Train Generator

        output = discriminator(X_fake).view(-1)

        gen_loss = loss_function_2(output, torch.ones_like(output))

        generator.zero_grad()
        gen_loss.backward()
        clip_grad_norm_(generator.parameters(), 1)
        generator_optimizer.step()

        disc_losses.append(disc_loss.item())
        gen_losses.append(gen_loss.item())

        losses_dict = {
            "Discriminator loss": disc_loss.item(),
            "Generator Loss": gen_loss.item(),
        }

        loop.set_postfix(losses_dict)

    return gen_losses, disc_losses


def vae_train_loop(
    model: Model,
    dataloader: Dataloader,
    optimizer: Optimizer,
    loss_function: LossFunction,
    device: Device,
):
    pass
