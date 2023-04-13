# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.load_env import ENV_CONFIG

input_file = Path(ENV_CONFIG["CONTROL_INPUT_DATA"])
RANDOM_SEED = int(ENV_CONFIG["RANDOM_SEED"])


def process_data(input_file: Path, out_dir: Path, seed : int = RANDOM_SEED):
    df: pd.DataFrame = pd.read_pickle(input_file)

    df_noname = (
        df.sample(frac=1, random_state=seed)
        .reset_index(drop=True)
        .drop(df.columns[0:2], axis=1)
        .pipe(lambda x: x / 2)
        .pipe(lambda x: x - np.random.uniform(0, 0.1, size=x.shape))
        .astype(np.float16)
    )

    df_noname.to_csv(out_dir / input_file.name.replace("pkl", "hapt"), sep="\t")


def main():
    process_data(input_file)


if __name__ == "__main__":
    main()
