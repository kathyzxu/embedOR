import logging
import os
from datetime import datetime
import argparse
import json
import numpy as np
import pandas as pd

from src.embedor import *
from src.plotting import *
from src.data.data import *

REPO_ROOT = os.path.expanduser("~/embedor")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
RESULTS_DIR = os.path.join(REPO_ROOT, "timing_results")
DATASET_DIR = os.path.join(REPO_ROOT, "timing_data")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

exp_params = {
    'p': 3,
    'mode': 'nbrs',
    'n_neighbors': 15,
}

# logging
dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(LOG_DIR, f"{dt_string}.log")

logger = logging.getLogger("embedor")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

print(f"[INFO] Logs saved to: {log_path}")


def run_pipeline(np_file, n_points=None, seed=42, save_root=RESULTS_DIR):
    np.random.seed(seed)

    data = np.load(np_file)
    if n_points is not None:
        n_points = min(n_points, data.shape[0])
        idx = np.random.choice(data.shape[0], size=n_points, replace=False)
        data = data[idx]

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    zero_rows = np.where(np.linalg.norm(data, axis=1) == 0)[0]
    if len(zero_rows) > 0:
        data[zero_rows] += 1e-6

    dataset_name = os.path.splitext(os.path.basename(np_file))[0]
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_root, dataset_name, dt_string)
    os.makedirs(save_path, exist_ok=True)

    logger.info(f"Running EmbedOR for dataset: {dataset_name} | shape={data.shape}")

    # EmbedOR (ORC)
    embedor = EmbedOR(exp_params)
    embedding = embedor.fit_transform(data)
    embedor.print_timing_summary()

    # EmbedOR (Euclidean)
    embedor_euc = EmbedOR(exp_params, edge_weight='euclidean')
    embedding_euc = embedor_euc.fit_transform(data)
    embedor_euc.print_timing_summary()

    # timings
    timing_data = {}
    if embedor.get_timings():
        timing_data["embedor"] = embedor.get_timings()
    if embedor_euc.get_timings():
        timing_data["embedor_euc"] = embedor_euc.get_timings()

    if timing_data:
        with open(os.path.join(save_path, "timings.json"), "w") as f:
            json.dump(timing_data, f, indent=4)

    logger.info(f"Finished {dataset_name}")
    return timing_data


def run_all(datasets_folder=DATASET_DIR, n_trials=3, n_points=None, base_seed=42):
    npy_files = sorted(
        os.path.join(datasets_folder, f)
        for f in os.listdir(datasets_folder)
        if f.endswith(".npy")
    )

    summary_rows = []

    for np_file in npy_files:
        dataset_name = os.path.splitext(os.path.basename(np_file))[0]
        logger.info(f"=== Dataset: {dataset_name} ===")

        for trial in range(n_trials):
            seed = base_seed
            logger.info(f"--- Trial {trial} | seed={seed} ---")
            timings = run_pipeline(
                np_file,
                n_points=n_points,
                seed=seed
            )

            summary_rows.append({
                "dataset": dataset_name,
                "trial": trial,
                "seed": seed,
                "timings": timings
            })


    with open(os.path.join(RESULTS_DIR, "all_summary.json"), "w") as f:
        json.dump(summary_rows, f, indent=4)

    logger.info("All runs completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_folder", type=str, default=DATASET_DIR)
    parser.add_argument("--n_points", type=int, default=5000)
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    run_all(
        datasets_folder=args.datasets_folder,
        n_trials=args.n_trials,
        n_points=args.n_points,
        base_seed=args.base_seed
    )
