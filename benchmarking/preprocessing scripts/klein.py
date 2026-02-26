"""
Select top 2000 HVGs from Klein dataset, select first 4 PCs, save to .npy.
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = os.path.expanduser("~/embedor")
DATA_DIR = os.path.join(REPO_ROOT, "raw_data", "klein")
OUTPUT_DIR = os.path.join(REPO_ROOT, "preprocessed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_NAME = "GSM1599494_ES_d0_main.csv.bz2"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

df = pd.read_csv(FILE_PATH, index_col=0, compression='bz2')
print(f"Original shape (genes x cells): {df.shape}")

df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)

gene_var = df_numeric.var(axis=1)
top_genes = gene_var.sort_values(ascending=False).head(2000).index
df_hvg = df_numeric.loc[top_genes]
print(f"Shape after selecting HVGs: {df_hvg.shape}")

pca = PCA(n_components=4)
pcs = pca.fit_transform(df_hvg.T)
print(f"Shape of PCA output (cells x 4 PCs): {pcs.shape}")

out_path = os.path.join(OUTPUT_DIR, "klein.npy")
np.save(out_path, pcs)
print(f"Saved PCA array to {out_path}")
