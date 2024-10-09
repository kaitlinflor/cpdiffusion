import os
import pandas as pd
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import numpy as np

# Paths to the CSV files
generated_folder_nuclei = '/gscratch/aims/kflores3/cellpainting/analysis/cp_outputs/output_train_gen/Nuclei.csv'
groundtruth_folder_nuclei = '/gscratch/aims/kflores3/cellpainting/analysis/cp_outputs/output_train_gt/Nuclei.csv'

# Load the generated and ground truth datasets
gen_df = pd.read_csv(generated_folder_nuclei)
gt_df = pd.read_csv(groundtruth_folder_nuclei)

# Select the relevant feature columns
gen_features = gen_df.iloc[:, 10:].copy()
gt_features = gt_df.iloc[:, 13:].copy()

# Add a column to label the datasets
gen_features['label'] = 'generated'
gt_features['label'] = 'groundtruth'

# Concatenate the datasets
combined_df = pd.concat([gen_features, gt_features])

# Extract features for PCA and UMAP
feature_columns = combined_df.columns[:-1]  # Exclude the label column
X = combined_df[feature_columns].dropna(axis=1).values
sample_indices = np.random.choice(X.shape[0], 10000, replace=False)
X_sample = X[sample_indices]

labels = combined_df['label'].values
# Select the corresponding labels for the sampled rows
labels_sample = labels[sample_indices]

# Perform PCA
pca = PCA(n_components=50)  # Reduce to 50 components before UMAP
# X_pca = pca.fit_transform(X)
X_pca = pca.fit_transform(X_sample)


# Perform UMAP
reducer = umap.UMAP(n_components=2)
X_umap = reducer.fit_transform(X_pca)

# Plot the UMAP results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=[1 if label == 'groundtruth' else 0 for label in labels_sample],
                      cmap='coolwarm', alpha=0.7)
plt.title('UMAP of Generated vs Ground Truth Features')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
# Add labels to each point

plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Generated'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Ground Truth')],
           loc='best')

# plt.colorbar(scatter, ticks=[0, 1], label='Generated / Ground Truth')
plt.savefig('umap.png')
# plt.show()
