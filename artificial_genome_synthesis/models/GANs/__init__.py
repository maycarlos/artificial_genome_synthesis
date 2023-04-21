from .gan_model import Generator as GAN_Generator, Discriminator as GAN_Discriminator
from .wgan_model import Generator as WGAN_Generator, Critic as WGAN_Critic
from .cgan_model import Generator as CGAN_Generator, Discriminator as CGAN_Discriminator

__all__ = [
    "GAN_Generator",
    "GAN_Discriminator",
    "WGAN_Generator",
    "WGAN_Critic",
    "CGAN_Generator",
    "CGAN_Discriminator",
]
