from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import torch
from torch.types import Device
from torch.utils.data import DataLoader

DataFrame = TypeVar("DataFrame", pd.DataFrame, Any)
Tensor = TypeVar("Tensor", torch.Tensor, Any)
Array = TypeVar("Array", np.ndarray, Any)
Model = TypeVar("Model", torch.nn.Module, Any)
Dataloader = TypeVar("Dataloader", DataLoader, Any)
Optimizer = TypeVar("Optimizer", torch.optim.Optimizer, Any)
LossFunction = TypeVar("LossFunction", Callable, Any)


__all__ = [
    "DataFrame",
    "Tensor",
    "Array",
    "Model",
    "Dataloader",
    "Optimizer",
    "LossFunction",
    "Device",
]
