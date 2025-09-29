"""
Convert preprocessed Tsai dataset to .npy.
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.expanduser("~/embedor")
DATA_DIR = os.path.join(REPO_ROOT, "raw_data", "tsai")
OUTPUT_DIR = os.path.join(REPO_ROOT, "preprocessed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_NAME = "GSE93374_Merged_all_020816_BatchCorrected_LNtransformed_doubletsremoved_Data.txt.gz"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

df = pd.read_csv(FILE_PATH, sep="\t", index_col=0)
print(f"Original shape: {df.shape}")

data_array = df.to_numpy()

out_path = os.path.join(OUTPUT_DIR, "tsai.npy")
np.save(out_path, data_array)

print(f"Saved preprocessed data to {out_path} with shape {data_array.shape}")
