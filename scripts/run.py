from src.data.data import *
from src.embedor import *
from src.plotting import *
from src.utils.orcmanl import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import argparse
import umap
import numpy as np
from sklearn.manifold import TSNE, Isomap
import phate
import json
import os

REPO_ROOT = os.getenv('PYTHONPATH') or os.getcwd()
sns.set_theme()

exp_params = {
    'p': 3,
    'mode': 'nbrs',
    'n_neighbors': 15,
}

def run_pipeline(np_file, n_points=None, seed=0):
    np.random.seed(seed)

    # Load dataset
    data = np.load(np_file)
    if n_points is not None:
        n_points = min(n_points, data.shape[0])
        indices = np.random.choice(data.shape[0], size=n_points, replace=False)
        data = data[indices]

    # Safe numeric preprocessing
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    zero_rows = np.where(np.linalg.norm(data, axis=1) == 0)[0]
    if len(zero_rows) > 0:
        data[zero_rows] += 1e-6

    # Output folder
    from datetime import datetime
    dataset_name = os.path.splitext(os.path.basename(np_file))[0] 
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    REPO_ROOT = os.path.expanduser("~/embedor")  # save outputs to home directory
    save_path = os.path.join(REPO_ROOT, 'outputs', dataset_name, dt_string)
    os.makedirs(save_path, exist_ok=True)

    labels = np.zeros(data.shape[0], dtype=int)  # dummy labels

    # ORCManL graph
    orcmanl = ORCManL(verbose=True)
    orcmanl.fit(data)

    # APSP distances
    nodes = list(orcmanl.G_pruned.nodes())
    n = len(nodes)
    apsp = np.zeros((n, n))
    lengths = dict(nx.all_pairs_shortest_path_length(orcmanl.G_pruned))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            apsp[i, j] = lengths[u].get(v, 1e10)
    apsp[apsp > 1e10] = 1e10

    # Embeddings
    embedor = EmbedOR(exp_params)
    embedding = embedor.fit_transform(data)
    embedor_euc = EmbedOR(exp_params, edge_weight='euclidean')
    embedding_euc = embedor_euc.fit_transform(data)
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(data)
    umap_orcmanl_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='precomputed').fit_transform(apsp)
    tsne_emb = TSNE(n_components=2, perplexity=30, max_iter=300, init='random').fit_transform(data)
    tsne_orcmanl_emb = TSNE(n_components=2, perplexity=30, max_iter=300, metric='precomputed', init='random').fit_transform(apsp)
    phate_emb = phate.PHATE(n_jobs=-2).fit_transform(data)
    iso_emb = Isomap(n_neighbors=15, n_components=2).fit_transform(data)

    # Low/high energy edges
    edge_energies = embedor.distances
    sorted_idx = np.argsort(edge_energies)
    low_idx = sorted_idx[:len(embedor.G.edges()) // 3]
    high_idx = sorted_idx[-len(embedor.G.edges()) // 50:]

    low_energy_graph = nx.Graph()
    low_energy_graph.add_nodes_from(embedor.G.nodes())
    low_energy_graph.add_edges_from([e for i, e in enumerate(embedor.G.edges()) if i in low_idx])

    high_energy_graph = nx.Graph()
    high_energy_graph.add_nodes_from(embedor.G.nodes())
    high_energy_graph.add_edges_from([e for i, e in enumerate(embedor.G.edges()) if i in high_idx])

    # Edge widths
    affinities = embedor.affinities
    max_thickness = 0.5
    edge_widths = np.array(affinities)**1.5 * (max_thickness / np.max(np.array(affinities)**1.5))

    # Stats dictionary
    stats_dict = {}
    stats_dict['dataset'] = {
        'embedor': dict(zip(['z_scores_mean','z_scores_std'], low_energy_edge_stats(embedding, embedor.G, low_energy_graph)[:2])),
        'embedor_euc': dict(zip(['z_scores_mean','z_scores_std'], low_energy_edge_stats(embedding_euc, embedor_euc.G, low_energy_graph)[:2])),
        'umap': dict(zip(['z_scores_mean','z_scores_std'], low_energy_edge_stats(umap_emb, embedor.G, low_energy_graph)[:2])),
        'umap_orcmanl': dict(zip(['z_scores_mean','z_scores_std'], low_energy_edge_stats(umap_orcmanl_emb, embedor.G, low_energy_graph)[:2])),
        'tsne': dict(zip(['z_scores_mean','z_scores_std'], low_energy_edge_stats(tsne_emb, embedor.G, low_energy_graph)[:2])),
        'tsne_orcmanl': dict(zip(['z_scores_mean','z_scores_std'], low_energy_edge_stats(tsne_orcmanl_emb, embedor.G, low_energy_graph)[:2])),
        'phate': dict(zip(['z_scores_mean','z_scores_std'], low_energy_edge_stats(phate_emb, embedor.G, low_energy_graph)[:2])),
        'iso': dict(zip(['z_scores_mean','z_scores_std'], low_energy_edge_stats(iso_emb, embedor.G, low_energy_graph)[:2])),
    }

    embeddings = {
        'embedor': embedding,
        'embedor_euc': embedding_euc,
        'umap': umap_emb,
        'umap_orcmanl': umap_orcmanl_emb,
        'tsne': tsne_emb,
        'tsne_orcmanl': tsne_orcmanl_emb,
        'phate': phate_emb,
        'iso': iso_emb
    }

    # Save plots
    for name, emb in embeddings.items():
        emb_path = os.path.join(save_path, name)
        os.makedirs(emb_path, exist_ok=True)

        plt.figure(figsize=(10, 10))
        plot_graph_2D(emb, embedor.G, node_color=labels[embedor.G.nodes()], edge_width=0, node_size=0.1, edge_color='red')
        plt.savefig(os.path.join(emb_path, 'class_annot.png'))
        plt.close()

        plt.figure(figsize=(10, 10))
        plot_graph_2D(emb, low_energy_graph, node_color=None, edge_width=0.1, node_size=0.1, edge_color='green')
        plt.savefig(os.path.join(emb_path, 'low_energy_graph.png'))
        plt.close()

        plt.figure(figsize=(10, 10))
        plot_graph_2D(emb, high_energy_graph, node_color=None, edge_width=0.02, node_size=0.1, edge_color='red')
        plt.savefig(os.path.join(emb_path, 'high_energy_graph.png'))
        plt.close()

        plt.figure(figsize=(10, 10))
        plot_graph_2D(emb, embedor.G, node_color=None, edge_width=edge_widths, node_size=0.1, edge_color='green')
        plt.savefig(os.path.join(emb_path, 'variable_edge_widths.png'))
        plt.close()

    # Save stats
    stats_path = os.path.join(save_path, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Stats saved to {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EmbedOR on any .npy dataset.")
    parser.add_argument("np_file", type=str, help="Path to .npy dataset")
    parser.add_argument("--n_points", type=int, default=None, help="Number of points to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    run_pipeline(args.np_file, n_points=args.n_points, seed=args.seed)
