"""
Select HVGs from petropoulos dataset, save to .npy
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.expanduser("~/embedor")
DATA_DIR = os.path.join(REPO_ROOT, "raw_data", "petropoulos")
OUTPUT_DIR = os.path.join(REPO_ROOT, "preprocessed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_NAME = "E_MTAB_3929_data.csv"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

def compute_variance_in_chunks(file_path, chunk_size=100):
    print("Computing gene variances in chunks...")
    
    df_sample = pd.read_csv(file_path, nrows=1)
    gene_columns = [col for col in df_sample.columns if col != 'Label']

    gene_sums = np.zeros(len(gene_columns), dtype=np.float64)
    gene_squares = np.zeros(len(gene_columns), dtype=np.float64)
    n_cells = 0
    
    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, index_col=0)
    
    for i, chunk in enumerate(chunk_reader):
        print(f"Processing chunk {i+1}...")
        
        chunk = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        gene_sums += chunk.sum(axis=0).values  
        gene_squares += (chunk ** 2).sum(axis=0).values
        n_cells = chunk.shape[0]  
        
        del chunk
        
    gene_means = gene_sums / n_cells
    gene_variance = (gene_squares / n_cells) - (gene_means ** 2)
    
    return gene_columns, gene_variance

def extract_top_genes(file_path, top_genes, chunk_size=100):
    first_chunk = pd.read_csv(file_path, nrows=1)
    cell_labels = []
    data_chunks = []

    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, index_col=0)
    
    for chunk in chunk_reader:
        cell_labels.extend(chunk.index.tolist())

        chunk_numeric = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
        chunk_top = chunk_numeric[top_genes].astype(np.float32)
        data_chunks.append(chunk_top)
        
        del chunk, chunk_numeric

    data_array = pd.concat(data_chunks, axis=0)
    return data_array.values, cell_labels

gene_columns, gene_var = compute_variance_in_chunks(FILE_PATH, chunk_size=100)

top_idx = np.argpartition(-gene_var, 500)[:500]
top_genes = [gene_columns[i] for i in top_idx]

data_array, cell_labels = extract_top_genes(FILE_PATH, top_genes, chunk_size=100)

out_path = os.path.join(OUTPUT_DIR, "petropoulos.npy")
np.save(out_path, data_array)
print(f"Saved preprocessed data to {out_path} with shape {data_array.shape}")
