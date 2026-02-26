"""
Preprocess Zeisel dataset, save to .npy.
"""

import scanpy as sc
import numpy as np
import pandas as pd
import os

REPO_ROOT = os.path.expanduser("~/embedor")
DATA_DIR = os.path.join(REPO_ROOT, "raw_data", "zeisel")
OUTPUT_DIR = os.path.join(REPO_ROOT, "preprocessed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_NAME = "GSE60361_C1-3005-Expression.txt.gz"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

df = pd.read_csv(FILE_PATH, sep="\t", index_col=0)
adata = sc.AnnData(X=df.T) 

sc.pp.filter_cells(adata, min_genes=500)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50)

np.save(os.path.join(OUTPUT_DIR, "zeisel.npy"), adata.obsm['X_pca'])
print(f"Saved preprocessed data to {os.path.join(OUTPUT_DIR, 'zeisel.npy')}")
