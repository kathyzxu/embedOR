"""
Preprocess Wagner dataset, save to .npy.
"""

import scanpy as sc
import numpy as np
import os

DATA_DIR = "raw_data/wagner"
OUTPUT_DIR = "preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

adata = sc.read_h5ad(os.path.join(DATA_DIR, "WagnerScience2018.h5ad"))

sc.pp.filter_cells(adata, min_genes=500)      # filter low-quality cells
sc.pp.filter_genes(adata, min_cells=3)        # filter rarely expressed genes
sc.pp.normalize_total(adata, target_sum=1e4)  # normalize counts per cell
sc.pp.log1p(adata)                            # log-transform
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50)                 

np.save(os.path.join(OUTPUT_DIR, "wagner.npy"), adata.obsm['X_pca'])
print("Saved preprocessed data to preprocessed_data/wagner.npy")
