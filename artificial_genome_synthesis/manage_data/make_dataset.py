from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.load_env import ENV_CONFIG

RANDOM_SEED = int(ENV_CONFIG["RANDOM_SEED"])


def load_n_process(
    input_file: Path, label_file: Optional[Path] = None, seed: int = RANDOM_SEED
):
    df = pd.read_pickle(input_file).iloc[:, :]

    has_labs = bool(label_file)

    if isinstance(label_file, Path):
        labs = pd.read_csv(label_file).iloc[:, 2]
        proc_labs = labs.astype(np.uint8)

        df = pd.concat([df, proc_labs], axis=1, ignore_index=True)

    df_noname = (
        df.sample(frac=1, random_state=seed)
        .reset_index(drop=True)
        .pipe(lambda x: x / 2)
        .pipe(apply_noise, has_labs=has_labs)
        # .pipe(lambda x: x - np.random.uniform(0, 0.1, size=x.shape))
        # .astype(np.float16)
    )

    return df_noname


def apply_noise(df, has_labs: bool):
    df = df.copy()
    if has_labs:
        piece = df.iloc[:, :-1]
        df.iloc[:, :-1] = piece - np.random.uniform(0, 0.1, size=piece.shape)
    else:
        df = df - np.random.uniform(0, 0.1, size=df.shape)

    return df
