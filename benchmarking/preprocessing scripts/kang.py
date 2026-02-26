"""
Select HVGs and perform PCA on samples A, B, and C from Kang dataset, save to .npy.
"""

import os
import pandas as pd
import numpy as np
import scipy.io
import scanpy as sc
from sklearn.decomposition import PCA
import glob
import gzip

REPO_ROOT = os.path.expanduser("~/embedor")
DATA_DIR = os.path.join(REPO_ROOT, "raw_data", "kang", "GSE96583_RAW")
OUTPUT_DIR = os.path.join(REPO_ROOT, "preprocessed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for file_path in glob.glob(os.path.join(DATA_DIR, "*.mat.gz")):
    print(f"Loading {os.path.basename(file_path)}")
    
    with gzip.open(file_path, 'rb') as f:
        counts = scipy.io.mmread(f)
    
    counts = counts.T.toarray() if hasattr(counts, 'toarray') else counts.T

    n_genes = counts.shape[1]
    n_cells = counts.shape[0]
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    cell_names = [f"{os.path.basename(file_path)}_cell_{i}" for i in range(n_cells)]
    
    adata = sc.AnnData(X=counts.astype(np.float32))
    adata.var_names = gene_names
    adata.obs_names = cell_names
    df_numeric = pd.DataFrame(adata.X.T, index=adata.var_names)
    gene_var = df_numeric.var(axis=1)
    top_genes = gene_var.sort_values(ascending=False).head(2000).index
    df_hvg = df_numeric.loc[top_genes]
    print(f"After HVG selection: {df_hvg.shape}")
    
    pca = PCA(n_components=50)
    pcs = pca.fit_transform(df_hvg.T)
    print(f"PCA shape: {pcs.shape}")
    
    batch_name = os.path.basename(file_path).replace('.mat.gz', '')
    out_path = os.path.join(OUTPUT_DIR, f"kang_{batch_name}.npy")
    np.save(out_path, pcs)
    print(f"Saved to {out_path}\n")
