from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, metaclass = ABCMeta):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(X):
        pass

