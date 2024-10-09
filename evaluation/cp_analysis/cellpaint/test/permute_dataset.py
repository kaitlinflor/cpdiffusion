import pandas as pd
import argparse
import logging
import random

from constants import constants

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

MEAN_MULTIPLIER_BOUNDS = (0.01, 0.1)
STDDEV_MULTIPLIER_BOUNDS = (0.01, 0.1)

def main():
    parser = argparse.ArgumentParser(description='Compares CellProfiler metrics from ground truth and generated images.')
    parser.add_argument('--merged_dataset_path',
                        help='Path to the folder where the merged Cell Profiler dataset for ground truth images resides.')
    parser.add_argument('--permuted_dataset_path',
                        help='Path to the folder where the merged Cell Profiler dataset for ground truth images resides.')
    parser.add_argument('--top_features_path',
                        help='Path to the folder where the .csv file with top features and their weights will be saved to.')

    args = parser.parse_args()
    logger.info(f'Loading dataset')
    df = pd.read_parquet(f"{args.merged_dataset_path}/merged_cellprofiler_data.parquet")
    logger.info(f'Loading top CP features')
    df_features = pd.read_csv(f"{args.top_features_path}/{constants.TOP_CELLPROFILER_FEATURES_FILENAME}.csv")

    chem_list = df[constants.CELLPROFILER_CHEM_COLUMN].unique()
    logger.info(f'Chemicals in the dataset: {chem_list}')

    # Randomly sample 3 chemicals to add a noise to
    selected_chems = np.random.choice(chem_list,3, replace=False)
    logger.info(f"Chemicals to be permuted: {selected_chems}")

    # Perturbing each "significant" feature of each selected chemical
    for chemical in selected_chems:
        for column_name in df_features['Features']:
            logger.info(f'Perturbing {column_name} for a chemical {chemical}')
            condition_single_col = df[constants.CELLPROFILER_CHEM_COLUMN] == chemical
            column_values = df[column_name][condition_single_col]

            # Adda  normal random noise, which will be compareable to column mean and
            # standard deviation.
            # Get column mean/std
            column_mean = np.mean(column_values)
            column_std = np.std(column_values)
            # Use metaparameters to generate mean/std of noise
            noise_mean_mult = np.random.uniform(*MEAN_MULTIPLIER_BOUNDS)
            noise_stddev_mult = np.random.uniform(*STDDEV_MULTIPLIER_BOUNDS)
            noise_mean = column_mean*noise_mean_mult
            noise_std = column_std * noise_stddev_mult
            logger.info(f'Column (mean, std): ({column_mean}, {column_std}). '+
                         f'Adding normal noise with (mean, std): ({noise_mean}, {noise_std}).'+
                         f'Perturbing total {df[condition_single_col].shape[0]} rows.')
            noise = np.random.normal(noise_mean, noise_std, df[condition_single_col].shape[0])
            df.loc[condition_single_col, column_name] += noise

    if args.permuted_dataset_path is not None:
        df.to_parquet(f"{args.permuted_dataset_path}/merged_cellprofiler_data.parquet", index=True)

main()
