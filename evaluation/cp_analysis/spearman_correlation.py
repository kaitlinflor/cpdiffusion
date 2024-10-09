import os
import pandas as pd
from scipy.stats import spearmanr
import polars as pl

# Define paths to the folders
generated_folder = '/gscratch/aims/kflores3/cellpainting/analysis/cp_outputs/output_test_gen'
groundtruth_folder = '/gscratch/aims/kflores3/cellpainting/analysis/cp_outputs/output_test_gt'

# List of compartments to process
# compartments = ['Nuclei', 'Cells', 'Cytoplasm']
compartments = ['Nuclei']
# Initialize the dictionary to store Spearman correlation results
# Initialize the dictionary to store Spearman correlation results
spearman_results = {}

# Loop over each compartment
for compartment in compartments:
    print(f"Going through compartment {compartment}")
    
    # Load the corresponding CSV files using polars
    gen_csv_path = os.path.join(generated_folder, f'{compartment}.csv')
    gt_csv_path = os.path.join(groundtruth_folder, f'{compartment}.csv')
    
    gen_df = pl.read_csv(gen_csv_path)
    gt_df = pl.read_csv(gt_csv_path)
    
    # Ensure both DataFrames have the same number of rows by cropping the longer one
    min_length = min(len(gen_df), len(gt_df))
    gen_df = gen_df.head(min_length)
    gt_df = gt_df.head(min_length)
    
    # Specify the feature columns to compare (columns 10 to 20)
    # feature_columns_gen = gen_df.columns[10:21]  # Columns 10 to 20
    # feature_columns_gt = gt_df.columns[10:21]
    feature_columns_gen = [col for col in gen_df.columns if "Zernike" in col]
    feature_columns_gt = [col for col in gt_df.columns if "Zernike" in col]
    
    # Ensure that the same columns are present in both dataframes
    common_features = set(feature_columns_gen) & set(feature_columns_gt)
    
    # Calculate Spearman correlation for the specified columns
    correlations = []
    for feature_gen, feature_gt in zip(feature_columns_gen, feature_columns_gt):
        print(f"Getting correlation for feature {feature_gen}")
        
        # Convert columns to numpy arrays for Spearman calculation
        feature_gen_values = gen_df[feature_gen].to_numpy()
        feature_gt_values = gt_df[feature_gt].to_numpy()
        
        # Calculate the correlation
        correlation, _ = spearmanr(feature_gen_values, feature_gt_values)
        print("Correlation : ", correlation)
        correlations.append(correlation)
    
    # Calculate the average correlation
    average_correlation = sum(correlations) / len(correlations) if correlations else None
    spearman_results[compartment] = average_correlation

# Output the average correlations
for compartment, avg_correlation in spearman_results.items():
    if avg_correlation is not None:
        print(f"Average Spearman Correlation for {compartment}: {avg_correlation}")
    else:
        print(f"No valid correlations for {compartment}.")