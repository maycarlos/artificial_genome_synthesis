from typing import Optional, Tuple

from torch.utils.data import DataLoader, Subset

from ..base_classes import BaseGenotype
from ..utils.types_ import Array, DataFrame, Tensor

Labels = Array | Tensor


class NotASubsetError(Exception):
    pass


class Genotype(BaseGenotype):
    def __init__(self, data: DataFrame, labels: Optional[Labels] = None) -> None:
        self.data = data
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if self.labels is None:
            return self.data[index]
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.data)

    def make_dataloader(
        self,
        batch_size: int,
        num_workers: int,
        generator,
        shuffle: bool,
        drop_last: bool,
        pin_memory: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            generator=generator,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    @classmethod
    def fromSubset(cls, subset: Subset):
        if isinstance(subset, Subset):
            X = subset.dataset.data.values
            y = subset.dataset.labels.values
            indices = subset.indices
            return cls(X[indices], y[indices])
        else:
            raise NotASubsetError
