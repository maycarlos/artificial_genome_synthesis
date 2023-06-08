from typing import Tuple

import torch
from torch import nn

from ...base_classes import BaseModel
from ...utils.types_ import Tensor


class VariationalAutoEncoder(BaseModel):
    def __init__(self, latent_size: int, features: int, alpha: float):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=features, out_features=int(features // 2)),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Linear(in_features=int(features // 2), out_features=int(features // 3)),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Linear(in_features=int(features // 3), out_features=latent_size),
        )

        self.miu = nn.Linear(in_features=latent_size, out_features=latent_size)
        self.sigma = nn.Linear(in_features=latent_size, out_features=latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=int(features // 3)),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Linear(in_features=int(features // 3), out_features=int(features // 2)),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Linear(in_features=int(features // 3), out_features=features),
        )

    def _encode(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        X = self.encoder(X)
        miu, sigma = self.miu(X), self.sigma(X)

        return miu, sigma

    def _decode(self, Z: Tensor) -> Tensor:
        x_reconstruct = torch.sigmoid(self.decoder(Z))

        return x_reconstruct

    def forward(self, X: Tensor):
        miu, sigma = self._encode(X)

        z = self.reparameterize(miu, sigma)

        X_rec = self._decode(z)

        return X_rec

    @staticmethod
    def reparameterize(miu: Tensor, sigma: Tensor):
        std = torch.exp(sigma / 2)

        eps = torch.randn_like(sigma).cuda()

        z = miu + eps * std

        return z
