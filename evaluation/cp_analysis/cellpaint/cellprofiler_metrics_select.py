# Merges the output of a CellProfiler (multiple files) and performs intial analysis of
# it and selection of important features.
# Running:
#
# Parse CellProfiler output, store it into  folder, and run feature selection on it.
# cellprofiler_metrics_select.py --cellprofiler_output_path <cell profiler output folder> --merged_dataset_path <folder where to write merged metrics file>
# e.g.
# cellprofiler_metrics_select.py --cellprofiler_output_path /home/iscander/CellPainting/output --merged_dataset_path /home/iscander/CellPainting/output
#
# If we already parsed CP output and generated merged metrics file - load it and run feature selection.
# cellprofiler_metrics_select.py --merged_dataset_path /home/iscander/CellPainting/output

import pandas as pd
import argparse
from dataclasses import dataclass
import logging

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder

import numpy as np
from constants import constants


@dataclass
class InputTableMetadata:
    """Contians some metadata about the input tables from CellProfiler."""
    # A suffix that we'll add to each column in the merged table of features.
    # Individual CellProfiler tables contain same features with the same names. We need
    # to add a suffix to distinguish them in a merged table.
    suffix: str


# Names of the files that contain CellProfiler output. We'll be loading them one by one.
INPUT_NAMES_AND_PARAMS = {
    'Cells': InputTableMetadata(suffix='_cells'),
    'Cytoplasm': InputTableMetadata(suffix='_cytoplasm'),
}

# CellPofiler metadtaa columns
METADATA_COLUMNS = ['Metadata_InChIKey', 'Metadata_Plate', 'Metadata_Site', 'Metadata_Well']
# CellProfiler key columns - a unique identifier of each image / cell
KEY_COLUMNS = ['ImageNumber', 'ObjectNumber']
# Columns that we can remove upoon loading (no numerical features to analyze)
COLUMNS_TO_REMOVE = ['FileName_', 'PathName_']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


def load_and_merge_csv(input_path):
    """Load the files with CellProfiler Metrics. Merge them into a single dataframe"""

    # Load CellProfiler files
    cellprofiler_dataframes = {}
    for (input_name, table_metadata) in INPUT_NAMES_AND_PARAMS.items():
        logger.info(f'Loading CellProfiler output: {input_name}')
        df = pd.read_csv(f"{input_path}/{input_name}.csv", index_col=KEY_COLUMNS)
        cellprofiler_dataframes[input_name] = df

    # Merge files into a single dataframe
    merged_df = None
    for (input_name, input_df) in cellprofiler_dataframes.items():
        if merged_df is None:
            # Initialize the "merged" dataframe with metadata columns, which are same for all CellProfiler DFs
            merged_df = input_df[METADATA_COLUMNS]
        logger.info(f'Merging CellProfiler dataframe: {input_name}')
        # Drop metadata columns
        input_df = input_df.drop(columns=METADATA_COLUMNS)
        # Drop other unnecessary columns, like file names and paths
        for column_pattern in COLUMNS_TO_REMOVE:
            columns_to_remove = input_df.filter(like=column_pattern).columns
            input_df = input_df.drop(columns=columns_to_remove)
        suffix = INPUT_NAMES_AND_PARAMS[input_name].suffix
        input_df = input_df.add_suffix(suffix)
        # Join this dataframe with a merged dataframe. They will be joined by key which is image id and cell id
        merged_df = merged_df.merge(input_df, left_index=True, right_index=True, how='inner')
    return merged_df


def pick_top_cellprofiler_metrics(merged_df, top_metrics_to_pick=10, ntrees=100, max_tree_depth=5):
    y_text = merged_df[constants.CELLPROFILER_CHEM_COLUMN]  # Target variable
    predictor_columns = [item for item in merged_df.columns if item not in METADATA_COLUMNS]
    X = merged_df[predictor_columns]  # Features

    # For the reference, do cross-correleation between features.
    correlation_matrix = np.corrcoef(X, rowvar=False)
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.title("Cross-Correlation Matrix for all features")
    plt.colorbar()
    plt.show()

    le = LabelEncoder()
    y = le.fit_transform(y_text)
    label_to_class_id = dict(zip(le.classes_, range(len(le.classes_))))
    class_id_to_label = dict(zip(range(len(le.classes_)), le.classes_))

    # Do test/train splits for the model. Pick mettrics on a train split, but leave out
    # a test split for basic model evaluation.
    # If there are no bugs, the RF model should show at least some accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Train the RF classifier
    model = RandomForestClassifier(n_estimators=ntrees, max_depth=max_tree_depth).fit(X_train, y_train)
    # Do the evaluation. The model should be able to discriminate _some_ chemicals. It's probably OK
    # if it can't accurately predict some - because it means that chemical don't do much to cells.
    y_pred = model.predict(X_test)
    conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, cmap='coolwarm', interpolation='none')
    plt.title("Confusion matrix for the RF model")
    plt.yticks(range(len(le.classes_)), le.classes_)
    plt.subplots_adjust(left=0.25, bottom=0.1, right=0.9, top=0.9)
    plt.colorbar()
    plt.show()
    precision = sklearn.metrics.precision_score(y_test, y_pred, average=None)
    recall = sklearn.metrics.recall_score(y_test, y_pred, average=None)
    print('RF model accuracy - precision and recall:')
    print(precision)
    print(recall)

    # Fetch features from the model and their Gini indices.
    features_importance_list = list(zip(model.feature_importances_, predictor_columns))
    features_importance_list = sorted(features_importance_list, key=lambda x: x[0], reverse=True)
    gini_coefficients_sorted, feature_names_sorted = zip(*features_importance_list)
    plt.plot(model.feature_importances_)
    plt.xlabel("Feature no.")
    plt.ylabel("Gini information gain")
    plt.title("Feature importance")
    plt.show()

    top_N_features = feature_names_sorted[0:top_metrics_to_pick]
    print(f'Top {top_metrics_to_pick} most informative features: {top_N_features}')
    print(f'Corresponding Gini indices: {gini_coefficients_sorted[0:top_metrics_to_pick]}')
    top_metrics = merged_df[list(top_N_features)]
    # corr = top_metrics.corr()
    # plt.matshow(corr, fignum=f.number)
    # corr.style.background_gradient(cmap='coolwarm')

    correlation_matrix_top_N = np.corrcoef(top_metrics, rowvar=False)
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix_top_N, cmap='coolwarm', interpolation='none')
    plt.title(f"Cross-Correlation Matrix for top {top_metrics_to_pick} features")
    plt.yticks(range(top_metrics.shape[1]), top_metrics.columns)
    plt.subplots_adjust(left=0.4, bottom=0.1, right=0.9, top=0.9)
    plt.show()
    return top_N_features, gini_coefficients_sorted[0:top_metrics_to_pick], correlation_matrix_top_N


def main():
    parser = argparse.ArgumentParser(description='Load, merge, and process CSV files.')
    parser.add_argument('--cellprofiler_output_path',
                        help='Path to the folder where CSV files generated by CellProfiler reside')
    parser.add_argument('--merged_dataset_path',
                        help='Path to the folder where the merged Cell Profiler dataset resides.')
    parser.add_argument('--top_features_path',
                        help='Path to the folder where the .csv file with top features and their weights will be saved to.')

    args = parser.parse_args()

    merged_df = None
    # --cellprofiler_output_path specified: we are loading CP features and saving them into a merged .csv file
    if args.cellprofiler_output_path is not None:
        logger.info(f'Loading original CellProfiler data files')
        merged_df = load_and_merge_csv(args.cellprofiler_output_path)
        logger.info(f'Writing merged CellProfiler dataframe to: {args.merged_dataset_path}')
        merged_df.to_csv(f"{args.merged_dataset_path}/{constants.MERGED_CELLPROFILER_FILENAME}.csv", index=True)
        merged_df.to_parquet(f"{args.merged_dataset_path}/{constants.MERGED_CELLPROFILER_FILENAME}.parquet", index=True)
    #
    if merged_df is None and args.merged_dataset_path is not None:
        logger.info(f'Reading merged CellProfiler data')
        merged_df = pd.read_parquet(f"{args.merged_dataset_path}/merged_cellprofiler_data.parquet")
    if merged_df is None:
        logger.critical("Neither CellProfiler output, nor merged dataset are specified. Nothing to process")

    # Given all the metrics from CellProfiler, compute most` important features (metrics),
    # their weights and correlation between those.
    logger.info(f'Analyzing CellProfiler data to select most important metrics')
    features, weights, correlation_matrix = pick_top_cellprofiler_metrics(merged_df, 20)
    if args.top_features_path is not None:
        logger.info(f'Saving most important metrics')
        dict_features_weights = {
            "Features": features,
            "Weights": weights
        }
        df_features_weights = pd.DataFrame(dict_features_weights)
        df_features_weights.to_csv(f"{args.top_features_path}/{constants.TOP_CELLPROFILER_FEATURES_FILENAME}.csv", index=False)

main()
