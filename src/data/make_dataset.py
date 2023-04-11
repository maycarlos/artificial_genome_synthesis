# -*- coding: utf-8 -*-
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

RANDOM_SEED = 123


def process_data(input_file: Path, out_dir: Path):
    df: pd.DataFrame = pd.read_pickle(input_file)

    df_noname = (
        df.sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
        .drop(df.columns[0:2], axis=1)
        .pipe(lambda x: x / 2)
        .pipe(lambda x: x - np.random.uniform(0, 0.1, size=x.shape))
        .astype(np.float16)
    )

    df_noname.to_csv(out_dir / input_file.name.replace("pkl", "hapt"), sep="\t")


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    parser = argparse.ArgumentParser(description="Process the raw data to processed")

    parser.add_argument(
        "input_dir", type=Path, help="Directory where the raw data is located"
    )

    parser.add_argument(
        "out_dir", type=Path, help="Directory where the processed file will be stored"
    )

    args = parser.parse_args()

    pickle_files = args.input_dir.glob("*.pkl")

    r_bar = "| {n_fmt}/{total_fmt} [{postfix}]"
    loop = tqdm(
        iterable=pickle_files,
        bar_format="{l_bar}{bar}" + r_bar,
    )

    for file in loop:
        loop.set_description(f"Processing {file}")
        process_data(file, args.out_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    main()
