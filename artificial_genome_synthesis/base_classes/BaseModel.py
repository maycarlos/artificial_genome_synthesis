from abc import ABCMeta, abstractmethod

from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, X):
        pass
