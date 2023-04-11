from torch.utils.data import Dataset

from ..utils.types_ import DataFrame


class GenotypeData(Dataset):
    def __init__(self, data: DataFrame):
        self.X = data.values

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)
