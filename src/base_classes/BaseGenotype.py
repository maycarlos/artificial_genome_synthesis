from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset
from typing import Optional
from ..utils.types_ import Array, Tensor, DataFrame 

Labels = Array | Tensor 

class BaseGenotype(Dataset, metaclass = ABCMeta):
    def __init__(self, data: DataFrame, labels : Optional[Labels] = None):
        self.data = data.values
        self.labels = labels

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    @abstractmethod
    def __len__(self):
        pass