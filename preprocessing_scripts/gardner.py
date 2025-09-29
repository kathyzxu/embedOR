"""
Save preprocessed 'rat r' Gardner datasets by day.
"""

import os
import glob
import numpy as np

RAW_DIR = "/home/kathy/embedor/raw_data/gardner"
OUTPUT_DIR = "/home/kathy/embedor/preprocessed_data/gardner"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop through all .npz files
npz_files = glob.glob(os.path.join(RAW_DIR, "*.npz"))

if not npz_files:
    raise FileNotFoundError(f"No .npz files found in {RAW_DIR}")

for fpath in npz_files:
    print(f"\nProcessing file: {fpath}")
    data = np.load(fpath, allow_pickle=True)
    
    # Loop through modules in the file (keys starting with 'spikes_mod')
    module_keys = [k for k in data.keys() if k.startswith("spikes_mod")]
    
    for mod_key in module_keys:
        print(f"  Extracting {mod_key}...")
        spikes_obj = data[mod_key].item()  # convert 0-d object array to dict
        
        # Convert dict of cells to 2D array (timepoints, n_cells)
        cells = sorted(spikes_obj.keys())
        min_len = min(len(spikes_obj[c]) for c in cells)
        spikes_matrix = np.column_stack([spikes_obj[c][:min_len] for c in cells])
        
        # Save as .npy
        base_name = os.path.splitext(os.path.basename(fpath))[0]
        out_file = os.path.join(OUTPUT_DIR, f"{base_name}_{mod_key}.npy")
        np.save(out_file, spikes_matrix)
        print(f"    Saved to: {out_file}")
