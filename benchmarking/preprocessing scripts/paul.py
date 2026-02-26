"""
Convert preprocessed Paul dataset to .npy.
"""

import os
import argparse
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

def rds_to_numpy(r_file_path, save_path=None):
    if not os.path.exists(r_file_path):
        raise FileNotFoundError(f"{r_file_path} does not exist.")
    
    sce = ro.r['readRDS'](r_file_path)
    SummarizedExperiment = importr('SummarizedExperiment')

    assay_matrix = SummarizedExperiment.assay(sce, "counts")
    
    r_func = ro.r('function(mat) { as.matrix(mat) }')
    dense_matrix = r_func(assay_matrix)
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        np_array = ro.conversion.rpy2py(dense_matrix)
    
    if np_array.shape[0] > np_array.shape[1]:
        np_array = np_array.T

    if save_path is None:
        save_path = os.path.splitext(r_file_path)[0] + '.npy'
    np.save(save_path, np_array)
    print(f"Saved NumPy array with shape {np_array.shape} to {save_path}")
    return np_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RDS file to NumPy array")
    parser.add_argument("--r_file", type=str, required=True, help="Path to .rds file")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save .npy file")
    args = parser.parse_args()
    
    rds_to_numpy(args.r_file, args.save_path)
