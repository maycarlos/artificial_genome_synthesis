import unittest

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import your project modules
from artificial_genome_synthesis.manage_data import Genotype
from artificial_genome_synthesis.utils import config

# from myproject.data import DataLoader
# from myproject.model import MyModel


class TestDataLoader(unittest.TestCase):
    def test_data_shape(self):
        # Test that the input data is the expected shape
        df = pd.read_csv(config["CONTROL_INPUT_DATA"], sep="\t")

        self.assertEqual(df.shape[1], 1)
        # control_genotype  = Genotype(df)

    def test_data_preprocessing(self):
        # Test that the data is being preprocessed correctly
        pass


if __name__ == "__main__":
    unittest.main()
