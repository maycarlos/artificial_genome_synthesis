from abc import ABCMeta, abstractmethod

from ...utils.types_ import (DataFrame, Dataloader, Device, LossFunction,
                             Model, Optimizer)


class BaseTrainerSetup(metaclass=ABCMeta):
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
        self.n_ags = n_ags
        self.original_data = original_data
        self.labels = labels
        self.device = device

    @abstractmethod
    def trainer_setup(self):
        pass

    @abstractmethod
    def train_step(self, engine, batch):
        pass
