"""
Select HVGs, perform PCA on Tsai dataset, save to .npy.
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

REPO_ROOT = os.path.expanduser("~/embedor")
DATA_DIR = os.path.join(REPO_ROOT, "raw_data", "tsai")
OUTPUT_DIR = os.path.join(REPO_ROOT, "preprocessed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_NAME = "GSE93374_Merged_all_020816_BatchCorrected_LNtransformed_doubletsremoved_Data.txt.gz"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

# read in chunks
CHUNK_ROWS = 1000      
N_TOP_GENES = 2000
N_COMPONENTS = 50
BATCH_SIZE = 1024      # number of samples (rows of memmap) processed per PCA 
DTYPE = np.float32

# read header and detect whether first column is gene index
header_cols = pd.read_csv(FILE_PATH, sep="\t", nrows=0, compression='gzip').columns.tolist()
# read first data row to check whether first column is gene id (non-numeric)
first_data = pd.read_csv(FILE_PATH, sep="\t", nrows=1, compression='gzip', header=0)
first_val = first_data.iloc[0, 0]
def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

if not is_number(first_val):
    # first column holds gene identifiers -> index column present
    index_name = header_cols[0]
    sample_names = header_cols[1:]
else:
    index_name = None
    sample_names = header_cols

n_samples = len(sample_names)
print(f"Detected n_samples = {n_samples}, index_name = {index_name}")

# compute per-gene variance chunkwise
gene_names = []
gene_vars = []

reader = pd.read_csv(FILE_PATH, sep="\t", index_col=index_name if index_name is not None else None,
                     chunksize=CHUNK_ROWS, compression='gzip')

total_rows = 0
for i, chunk in enumerate(reader):
    print(f"[var] processing chunk {i+1} with shape {chunk.shape} ...")
    # ensure numeric and fill NA
    chunk_num = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
    # compute variance across samples for each gene (axis=1)
    vars_chunk = chunk_num.var(axis=1).to_numpy(dtype=np.float64)
    gene_vars.append(vars_chunk)
    gene_names.extend(chunk_num.index.tolist())
    total_rows += chunk_num.shape[0]
    del chunk, chunk_num, vars_chunk

gene_vars = np.concatenate(gene_vars, axis=0)
print(f"Computed variances for {len(gene_names)} genes (total_rows={total_rows}).")

# choose top genes
n_top = min(N_TOP_GENES, len(gene_names))
top_idx = np.argpartition(-gene_vars, n_top - 1)[:n_top]
top_idx_sorted = top_idx[np.argsort(-gene_vars[top_idx])]
top_genes = [gene_names[i] for i in top_idx_sorted]

print(f"Selected top {len(top_genes)} genes. Variance range: {gene_vars[top_idx].min():.4g} - {gene_vars[top_idx].max():.4g}")

memmap_path = os.path.join(OUTPUT_DIR, "tsai_topgenes_mem.dat")
if os.path.exists(memmap_path):
    os.remove(memmap_path)
top_count = len(top_genes)
pca_input = np.memmap(memmap_path, dtype=DTYPE, mode="w+", shape=(n_samples, top_count))

top_gene_to_col = {g: i for i, g in enumerate(top_genes)}

# re-iterate file in chunks, write columns for top genes present in each chunk
reader = pd.read_csv(FILE_PATH, sep="\t", index_col=index_name if index_name is not None else None,
                     chunksize=CHUNK_ROWS, compression='gzip')

filled = 0
for i, chunk in enumerate(reader):
    print(f"[fill] processing chunk {i+1} with shape {chunk.shape} ...")
    chunk_num = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
    # loop over rows in chunk that are in top_genes
    for gene in chunk_num.index:
        if gene in top_gene_to_col:
            col_idx = top_gene_to_col[gene]
            pca_input[:, col_idx] = chunk_num.loc[gene].to_numpy(dtype=DTYPE, copy=False)
            filled += 1
    del chunk, chunk_num

print(f"Filled {filled}/{top_count} top-gene columns into memmap.")
pca_input.flush()

# run IncrementalPCA and write final PCA results to .npy
ipca = IncrementalPCA(n_components=N_COMPONENTS)

for start in range(0, n_samples, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_samples)
    batch = np.asarray(pca_input[start:end, :]) 
    ipca.partial_fit(batch)
    print(f"[ipca] partial_fit processed samples {start}:{end}")

# transform in batches 
pca_out_path = os.path.join(OUTPUT_DIR, "tsai.npy")
if os.path.exists(pca_out_path):
    os.remove(pca_out_path)
pca_mem = np.lib.format.open_memmap(pca_out_path, mode='w+', dtype=DTYPE, shape=(n_samples, N_COMPONENTS))

write_idx = 0
for start in range(0, n_samples, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_samples)
    batch = np.asarray(pca_input[start:end, :])
    trans = ipca.transform(batch) 
    pca_mem[start:end, :] = trans
    write_idx += trans.shape[0]
    print(f"[ipca] transform wrote rows {start}:{end}")

pca_mem.flush()
print(f"Saved PCA (50 PCs) to {pca_out_path} with shape {pca_mem.shape}")
