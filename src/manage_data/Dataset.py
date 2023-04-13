from ..utils.types_ import DataFrame

from ..base_classes import BaseGenotype
from typing import Optional
from ..utils.types_ import Array, Tensor

Labels = Array | Tensor

class Genotype(BaseGenotype):
    def __init__(self, data : DataFrame, labels: Optional[Labels] = None) -> None:
        self.data = data.values
        self.labels = labels

    def __getitem__(self, index: int):

        if self.labels is None:
            return self.data[index]
        return self.data[index], self.labels[index]

        pass

    def __len__(self):
        return len(self.data)

