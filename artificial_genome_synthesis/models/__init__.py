from .GANs import *
from .VAEs import *


def instanciate_models(dataloader, alpha):
    """


    Args
    ---
        dataloader: _description_
        alpha: _description_

    Returns
    ---
        Models: algo
    """
    X, y = next(iter(dataloader))

    generator = WGAN_Generator(
        latent_size=y.shape[1],
        features=X.shape[1],
        alpha=alpha,
    ).cuda()

    critic = WGAN_Critic(features=X.shape[1], alpha=alpha).cuda()

    return generator, critic
