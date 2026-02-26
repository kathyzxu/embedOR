"""
Concatenate preprocessed Baron dataset into human and mouse files, save to .npy.
"""

import os
import glob
import pandas as pd
import numpy as np

REPO_ROOT = os.path.expanduser("~/embedor")
DATA_DIR = os.path.join(REPO_ROOT, "raw_data", "baron")
OUTPUT_DIR = os.path.join(REPO_ROOT, "preprocessed_data", "baron")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_combine(files):
    dfs = [pd.read_csv(f, index_col=0).select_dtypes(include=np.number) for f in files]
    combined = pd.concat(dfs, axis=0)
    print(f"Combined shape: {combined.shape}")
    return combined

def save_numpy(df, fname):
    arr = df.to_numpy()
    np.save(fname, arr)
    print(f"Saved NumPy array to {fname} with shape {arr.shape}")

if __name__ == "__main__":
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv.gz")))
    human_files = all_files[:4]
    mouse_files = all_files[-2:]

    print("Human files:", human_files)
    print("Mouse files:", mouse_files)

    human_df = load_and_combine(human_files)
    mouse_df = load_and_combine(mouse_files)

    save_numpy(human_df, os.path.join(OUTPUT_DIR, "human.npy"))
    save_numpy(mouse_df, os.path.join(OUTPUT_DIR, "mouse.npy"))
