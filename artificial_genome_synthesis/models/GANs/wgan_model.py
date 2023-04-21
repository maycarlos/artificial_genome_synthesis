import torch.nn as nn

from ...utils.types_ import Tensor


class Generator(nn.Module):
    def __init__(self, latent_size: int, features: int, alpha: float):
        super().__init__()

        self.generate = nn.Sequential(
            nn.Linear(in_features=latent_size,
                      out_features=int(features // 1.2)),
            nn.LeakyReLU(negative_slope=alpha),
            # nn.GELU(),
            nn.Linear(
                in_features=int(features // 1.2),
                out_features=int(features // 1.1),
            ),
            nn.LeakyReLU(negative_slope=alpha),
            # nn.GELU(),
            nn.Linear(in_features=int(features // 1.1), out_features=features),
            nn.Tanh(),
        )

    def forward(self, X: Tensor) -> Tensor:
        X = self.generate(X)
        return X

    def generation(self, noise: Tensor) -> Tensor:
        genome = self.forward(noise) * 2
        return genome


class Critic(nn.Module):
    def __init__(self, features: int, alpha: float) -> None:
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(in_features=features, out_features=int(features // 2)),
            nn.LeakyReLU(negative_slope=alpha),
            # nn.GELU(),
            nn.Linear(
                in_features=int(features // 2),
                out_features=int(features // 3),
            ),
            nn.LeakyReLU(negative_slope=alpha),
            # nn.GELU(),
            nn.Linear(in_features=int(features // 3), out_features=1),
        )

    def forward(self, X: Tensor) -> Tensor:
        X = self.critic(X)
        return X
