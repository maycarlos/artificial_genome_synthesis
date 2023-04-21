from typing import Optional
from torch.utils.data import DataLoader
from ..base_classes import BaseGenotype
from ..utils.types_ import Array, DataFrame, Tensor

Labels = Array | Tensor


class Genotype(BaseGenotype):
    def __init__(self, data: DataFrame, labels: Optional[Labels] = None) -> None:
        self.data = data.values
        self.labels = labels

    def __getitem__(self, index: int) -> Tensor:

        if self.labels is None:
            return self.data[index]
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def make_dataloader(
        self,
        batch_size: int,
        num_workers: int,
        generator,
        shuffle: bool,
        drop_last: bool,
        pin_memory: bool,
    ):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            generator=generator,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
