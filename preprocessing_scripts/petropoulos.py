"""
Read preprocessed Petropoulos data in chunks and selects top 500 variable genes.
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
    """Compute gene variances reading CSV in chunks to save memory."""
    print("Computing gene variances in chunks...")
    
    # Get gene names
    df_sample = pd.read_csv(file_path, nrows=1)
    gene_columns = [col for col in df_sample.columns if col != 'Label']
    print(f"Found {len(gene_columns)} genes")
    
    gene_sums = np.zeros(len(gene_columns), dtype=np.float64)
    gene_squares = np.zeros(len(gene_columns), dtype=np.float64)
    n_cells = 0
    
    # Read in chunks
    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, index_col=0)
    
    for i, chunk in enumerate(chunk_reader):
        print(f"Processing chunk {i+1}...")
        
        chunk = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Update statistics
        gene_sums += chunk.sum(axis=0).values  
        gene_squares += (chunk ** 2).sum(axis=0).values
        n_cells = chunk.shape[0]  
        
        del chunk
    
    # Compute variance
    gene_means = gene_sums / n_cells
    gene_variance = (gene_squares / n_cells) - (gene_means ** 2)
    
    return gene_columns, gene_variance

def extract_top_genes(file_path, top_genes, chunk_size=100):
    """Extract only the top genes in chunks and preserve labels."""
    print("Extracting top genes in chunks...")
    
    first_chunk = pd.read_csv(file_path, nrows=1)
    cell_labels = []
    data_chunks = []

    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, index_col=0)
    
    for chunk in chunk_reader:
        cell_labels.extend(chunk.index.tolist())
        
        # Select top genes
        chunk_numeric = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
        chunk_top = chunk_numeric[top_genes].astype(np.float32)
        data_chunks.append(chunk_top)
        
        del chunk, chunk_numeric
    
    # Combine chunks
    data_array = pd.concat(data_chunks, axis=0)
    return data_array.values, cell_labels

gene_columns, gene_var = compute_variance_in_chunks(FILE_PATH, chunk_size=100)

# Get indices of top 500 most variable genes
top_idx = np.argpartition(-gene_var, 500)[:500]
top_genes = [gene_columns[i] for i in top_idx]

print(f"Selected top 500 genes from {len(gene_columns)} total genes")
print(f"Variance range: {gene_var[top_idx].min():.4f} - {gene_var[top_idx].max():.4f}")
print(f"First 10 top genes: {top_genes[:10]}")

data_array, cell_labels = extract_top_genes(FILE_PATH, top_genes, chunk_size=100)

print(f"Final data shape: {data_array.shape}")  # Should be (cells × 500)
print(f"Number of cells: {len(cell_labels)}")
print(f"Cell labels sample: {cell_labels[:5]}")

# Save filtered data
out_path = os.path.join(OUTPUT_DIR, "petropoulos.npy")
np.save(out_path, data_array)
print(f"Saved preprocessed data to {out_path} with shape {data_array.shape}")