import pandas as pd
import argparse
import logging

import numpy as np
from scipy.stats import chi2

from cellpaint.constants import constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ComparisonResult:
    diff_per_column: Dict[str, float] = field(default_factory=dict)
    diff_metric: float = 0.0

class DataframeComparer:
    def get_sample_(self, df, arr_features, chem_id):
        condition_single_col = df[constants.CELLPROFILER_CHEM_COLUMN] == chem_id
        sample = df.loc[condition_single_col, arr_features]
        return sample

    def compare(self, df_gt, df_generated, features):
        return None


class HotellingDataframeComparer(DataframeComparer):
    def compare(self, df_gt, df_generated, features):
        sample_gt = df_gt[features].to_numpy()
        sample_generated = df_generated[features].to_numpy()
        # Get dimensionalities of the ground truth and generated samples.
        nx = sample_gt.shape[0]
        ny = sample_generated.shape[0]
        px = sample_gt.shape[1]
        py = sample_generated.shape[1]
        # They must be equal
        assert(px == py)
        gt_mean = np.mean(sample_gt, axis=0)
        generated_mean = np.mean(sample_generated, axis=0)
        gt_cov = np.cov(sample_gt, rowvar=False)
        generated_cov = np.cov(sample_generated, rowvar=False)
        pooled_cov = (nx*gt_cov + ny*generated_cov)/(nx + ny + 2)
        # TODO: check a condition number and if the matrix is singular
        pooled_cov_inv = np.linalg.inv(pooled_cov)
        d_sample = gt_mean - generated_mean
        t2 = d_sample.T @ pooled_cov_inv @ d_sample
        t2 *= nx*ny/(nx+ny)
        # Now, once we have t2, let's recompute it into the value of F distribution
        # f_val = (t2*(nx + ny - px -1))/((nx + ny - 2)*px)
        chi2_val = chi2.ppf(0.1, px)
        return ComparisonResult(diff_metric=t2)

def main():
    parser = argparse.ArgumentParser(description='Compares CellProfiler metrics from ground truth and generated images.')
    parser.add_argument('--merged_dataset_path_groundtruth',
                        help='Path to the folder where the merged Cell Profiler dataset for ground truth images resides.')
    parser.add_argument('--merged_dataset_path_generated',
                        help='Path to the folder where the merged Cell Profiler dataset for generated images resides.')
    parser.add_argument('--top_features_path',
                        help='Path to the folder where the .csv file with top features and their weights will be saved to.')

    args = parser.parse_args()
    logger.info(f'Loading datasets')
    df_gt = pd.read_parquet(f"{args.merged_dataset_path_groundtruth}/merged_cellprofiler_data.parquet")
    df_generated = pd.read_parquet(f"{args.merged_dataset_path_generated}/merged_cellprofiler_data.parquet")
    logger.info(f'Loading top CP features')
    df_features = pd.read_csv(f"{args.top_features_path}/{constants.TOP_CELLPROFILER_FEATURES_FILENAME}.csv")

    features = df_features["Features"].to_list()
    features_and_chem = features.copy()
    features_and_chem.append(constants.CELLPROFILER_CHEM_COLUMN)

    chem_list_gt = df_gt[constants.CELLPROFILER_CHEM_COLUMN].unique()
    chem_list_generated = df_gt[constants.CELLPROFILER_CHEM_COLUMN].unique()
    assert(sorted(chem_list_gt) == sorted(chem_list_generated))

    comparer = HotellingDataframeComparer()

    for chemical in chem_list_gt:
        df_gt_filtered = df_gt[features_and_chem]
        df_generated_filtered = df_generated[features_and_chem]

        condition_gt = df_gt_filtered[constants.CELLPROFILER_CHEM_COLUMN] == chemical
        gt_values = df_gt_filtered[condition_gt]

        condition_generated = df_generated_filtered[constants.CELLPROFILER_CHEM_COLUMN] == chemical
        generated_values = df_generated_filtered[condition_generated]

        result = comparer.compare(gt_values, generated_values, features)
        print(f"Result for a chemical {chemical}: {result}")

main()