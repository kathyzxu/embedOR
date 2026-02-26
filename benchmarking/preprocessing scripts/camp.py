"""
Save preprocessed Camp dataset to .npy.
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.expanduser("~/embedor")
DATA_DIR = os.path.join(REPO_ROOT, "raw_data", "camp")
OUTPUT_DIR = os.path.join(REPO_ROOT, "preprocessed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_NAME = "GSE75140_hOrg.fetal.master.data.frame.txt.gz"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

df = pd.read_csv(FILE_PATH, sep="\t", index_col=0)
print(f"Original shape: {df.shape}")

df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
data_array = df_numeric.to_numpy()

out_path = os.path.join(OUTPUT_DIR, "camp.npy")
np.save(out_path, data_array)

print(f"Saved preprocessed data to {out_path} with shape {data_array.shape}")
