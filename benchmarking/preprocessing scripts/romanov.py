"""
Normalize, log-transform, and perform PCA on Romanov dataset, save to .npy.
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

temp_xlsx = "/tmp/temp_romanov.xlsx"
with gzip.open(DATA_PATH, 'rb') as f_in:
    with open(temp_xlsx, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

df = pd.read_excel(temp_xlsx, index_col=0, engine='openpyxl')
os.remove(temp_xlsx)

metadata = df.iloc[:5].T  # Transpose to have cells as rows
expression_data = df.iloc[5:]  # Gene expression data starts from row 5

expression_numeric = expression_data.apply(pd.to_numeric, errors='coerce').fillna(0)

adata = sc.AnnData(X=expression_numeric.T.astype(np.float32))
adata.var_names = expression_numeric.index  # Gene names
adata.obs_names = expression_numeric.columns  # Cell barcodes

adata.obs['cell_type'] = metadata.iloc[:, 0].values  # level1 class
adata.obs['neuron_subtype'] = metadata.iloc[:, 1].values  # level2 class
adata.obs['age'] = metadata.iloc[:, 3].values  # age
adata.obs['sex'] = metadata.iloc[:, 4].values  # sex

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata, n_comps=50)

out_path = os.path.join(OUTPUT_DIR, "romanov.npy")
np.save(out_path, adata.obsm['X_pca'])
print(f"Saved PCA (50 PCs) with shape {adata.obsm['X_pca'].shape}")
