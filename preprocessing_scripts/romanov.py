"""
Normalization, log-transform, and PCA on pre-processed Romanov data.
"""

import os
import pandas as pd
import scanpy as sc
import numpy as np
import gzip
import shutil

REPO_ROOT = os.path.expanduser("~/embedor")
DATA_DIR = os.path.join(REPO_ROOT, "raw_data", "romanov")
OUTPUT_DIR = os.path.join(REPO_ROOT, "preprocessed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "GSE74672_expressed_mols_with_classes.xlsx.gz")

# Decompress gzip file 
temp_xlsx = "/tmp/temp_romanov.xlsx"
with gzip.open(DATA_PATH, 'rb') as f_in:
    with open(temp_xlsx, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Read decompressed file
df = pd.read_excel(temp_xlsx, index_col=0, engine='openpyxl')
os.remove(temp_xlsx)

print(f"DataFrame shape: {df.shape}")
print(f"First few rows:")
print(df.head())

# The first 5 rows are metadata, the rest are gene expression counts
# Rows 0-4: metadata (cell types, age, sex)
# Rows 5-end: gene expression counts

# Extract metadata from first 5 rows
metadata = df.iloc[:5].T  # Transpose to have cells as rows
expression_data = df.iloc[5:]  # Gene expression data starts from row 5

print(f"Metadata shape: {metadata.shape}")
print(f"Expression data shape: {expression_data.shape}")

# Convert expression data to numeric
expression_numeric = expression_data.apply(pd.to_numeric, errors='coerce').fillna(0)

adata = sc.AnnData(X=expression_numeric.T.astype(np.float32))
adata.var_names = expression_numeric.index  # Gene names
adata.obs_names = expression_numeric.columns  # Cell barcodes

# Add metadata to observations
adata.obs['cell_type'] = metadata.iloc[:, 0].values  # level1 class
adata.obs['neuron_subtype'] = metadata.iloc[:, 1].values  # level2 class
adata.obs['age'] = metadata.iloc[:, 3].values  # age
adata.obs['sex'] = metadata.iloc[:, 4].values  # sex

print(f"AnnData shape: {adata.shape}")
print(f"Cell types: {adata.obs['cell_type'].unique()}")

# Normalization + log-transform + PCA
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata, n_comps=50)

# Save PCA embeddings
out_path = os.path.join(OUTPUT_DIR, "romanov.npy")
np.save(out_path, adata.obsm['X_pca'])
print(f"Saved PCA (50 PCs) with shape {adata.obsm['X_pca'].shape}")