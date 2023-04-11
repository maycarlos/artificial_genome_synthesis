import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from dotenv import dotenv_values, find_dotenv

print(f"{np.__name__} version = {np.__version__}")
print(f"{pd.__name__} version = {pd.__version__}")
print(f"{sklearn.__name__} version = {sklearn.__version__}")
print(f"{mpl.__name__} version = {mpl.__version__}")
print(f"{sns.__name__} version = {sns.__version__}")

# Env file loading
def load_env():
    env_file = find_dotenv(
        filename = "defaults.env",
        raise_error_if_not_found=True
        )
    config = dotenv_values(env_file)
    return config


ENV_CONFIG = load_env()
