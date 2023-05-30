import time

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from colorama import Fore, init
from dotenv import dotenv_values, find_dotenv

init(autoreset=True)


RUN_TIME = int(time.time())

print("Experiment nº ", end="")
print(Fore.GREEN + f"{RUN_TIME}" + Fore.RESET)

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
            "RUN_TIME" : RUN_TIME,
        }
    )

    return config


ENV_CONFIG = load_env()

__all__ = ["ENV_CONFIG"]
