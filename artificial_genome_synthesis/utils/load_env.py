import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from dotenv import dotenv_values, find_dotenv


# Env file loading
def load_env():
    env_file = find_dotenv(filename=".env", raise_error_if_not_found=True)
    config = dotenv_values(env_file)
    config.update(
        {
            "numpy version": np.__version__,
            "pandas version": pd.__version__,
            "scikit_learn version": sklearn.__version__,
            "matplotlib version": mpl.__version__,
            "seaborn version": sns.__version__,
        }
    )

    return config


ENV_CONFIG = load_env()

__all__ = ["ENV_CONFIG"]
