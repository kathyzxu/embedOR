"""
Permutation testing with 10, 000 resamples

Compares low ∆E edge expansion between EmbedOR, tSNE, and UMAP.
"""

import numpy as np
import gc
from typing import Dict, Tuple, Optional
import argparse
from scipy.stats import permutation_test


class MemoryEfficientEmbeddingTester:
    def __init__(self, beta: float = 0.33, n_perm: int = 10000,
                 chunk_size: int = 100, seed: int = 42, max_points: int = 5000):
        self.beta = beta
        self.n_perm = n_perm
        self.chunk_size = chunk_size
        self.seed = seed
        self.max_points = max_points
        np.random.seed(seed)

    def subsample_data(self, data: np.ndarray) -> np.ndarray:
        """Subsample data if it's too large."""
        if data.shape[0] <= self.max_points:
            return data
        
        print(f"Subsampling from {data.shape[0]} to {self.max_points} points")
        rng = np.random.RandomState(self.seed)
        indices = rng.choice(data.shape[0], self.max_points, replace=False)
        return data[indices]

    def get_low_energy_edge_indices(self, embedor) -> np.ndarray:
        """Get indices of lowest beta fraction of edges."""
        edge_energies = embedor.distances
        n_edges = len(edge_energies)

        if n_edges == 0:
            return np.array([], dtype=np.int32)

        k = int(self.beta * n_edges)
        k = max(1, min(n_edges, k))

        # Use full sort to match notebook exactly
        sorted_idx = np.argsort(edge_energies)
        low_idx = sorted_idx[:k]
        return low_idx

    def compute_all_edge_distances(self, embedding: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Compute distances for ALL edges (for mean/std calculation)."""
        if edges.size == 0:
            return np.zeros(0, dtype=np.float32)

        n_edges = len(edges)
        distances = np.empty(n_edges, dtype=np.float32)

        for i in range(0, n_edges, self.chunk_size):
            end_idx = min(i + self.chunk_size, n_edges)
            chunk_edges = edges[i:end_idx]

            a = embedding[chunk_edges[:, 0]]
            b = embedding[chunk_edges[:, 1]]
            diff = a - b
            distances[i:end_idx] = np.sqrt(np.sum(diff ** 2, axis=1))

        return distances

    def compute_low_energy_distances(self, embedding: np.ndarray, low_edges: np.ndarray) -> np.ndarray:
        """Compute distances only for low-energy edges."""
        if low_edges.size == 0:
            return np.zeros(0, dtype=np.float32)

        n_edges = len(low_edges)
        distances = np.empty(n_edges, dtype=np.float32)

        for i in range(0, n_edges, self.chunk_size):
            end_idx = min(i + self.chunk_size, n_edges)
            chunk_edges = low_edges[i:end_idx]

            a = embedding[chunk_edges[:, 0]]
            b = embedding[chunk_edges[:, 1]]
            diff = a - b
            distances[i:end_idx] = np.sqrt(np.sum(diff ** 2, axis=1))

        return distances

    def compute_z_scores_correctly(self, embedding: np.ndarray, all_edges: np.ndarray, low_edges: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Compute z-scores EXACTLY like the notebook:
        - Use ALL edges to compute mean and std
        - Apply to low-energy edges only
        """
        print("    Computing distances for ALL edges (for mean/std)...")
        all_distances = self.compute_all_edge_distances(embedding, all_edges)
        
        avg_distance = np.mean(all_distances)
        std_distance = np.std(all_distances)
        print(f"    All edges - mean: {avg_distance:.4f}, std: {std_distance:.4f}")

        print("    Computing distances for low-energy edges...")
        low_distances = self.compute_low_energy_distances(embedding, low_edges)
        
        # Compute z-scores: (low_distance - mean_of_all) / std_of_all
        z_scores = (low_distances - avg_distance) / std_distance
        
        mean_z_score = np.mean(z_scores)
        std_z_score = np.std(z_scores)
        print(f"    Low-energy edges - mean z: {mean_z_score:.4f}, std z: {std_z_score:.4f}")

        # Clean up
        del all_distances, low_distances
        gc.collect()
        
        return mean_z_score, std_z_score, z_scores

    def compute_single_embedding(self, data: np.ndarray, method: str) -> np.ndarray:
        """Compute a single embedding."""
        print(f"    Computing {method} embedding...")
        
        try:
            if method == 'embedor':
                from src.embedor import EmbedOR
                embedor = EmbedOR({'p': 3, 'mode': 'nbrs', 'n_neighbors': 15})
                embedding = embedor.fit_transform(data)
                del embedor
            elif method == 'umap':
                import umap
                embedding = umap.UMAP(
                    n_neighbors=15, 
                    min_dist=0.1, 
                    metric='euclidean',
                    random_state=self.seed, 
                    low_memory=True
                ).fit_transform(data)
            elif method == 'tsne':
                from sklearn.manifold import TSNE
                embedding = TSNE(
                    n_components=2, 
                    perplexity=min(30, data.shape[0] - 1),
                    init='random',
                    random_state=self.seed, 
                    method='barnes_hut'
                ).fit_transform(data)
            else:
                raise ValueError(f"Unknown method: {method}")

            return embedding

        except ImportError as e:
            print(f"    Warning: {method} not available, using random embedding")
            return np.random.randn(data.shape[0], 2).astype(np.float32)
        except Exception as e:
            print(f"    Error computing {method}: {e}")
            return np.random.randn(data.shape[0], 2).astype(np.float32)

    def run_test(self, data: np.ndarray) -> Dict:
        """Main testing procedure with FIXED z-score calculation."""
        print(f"Step 1: Subsampling data (max_points={self.max_points})...")
        data_subsampled = self.subsample_data(data)
        print(f"  Using {data_subsampled.shape[0]} points")

        print("Step 2: Computing EmbedOR graph...")
        try:
            from src.embedor import EmbedOR
            embedor_ref = EmbedOR({'p': 3, 'mode': 'nbrs', 'n_neighbors': 15})
            _ = embedor_ref.fit_transform(data_subsampled)

            all_edges = np.array(list(embedor_ref.G.edges()), dtype=np.int32)
            print(f"  Graph has {len(all_edges)} total edges")

            low_idx = self.get_low_energy_edge_indices(embedor_ref)
            low_edges = all_edges[low_idx] if low_idx.size > 0 else np.zeros((0, 2), dtype=np.int32)
            print(f"  Selected {len(low_edges)} low-energy edges (beta={self.beta})")

            del embedor_ref
            gc.collect()

        except ImportError:
            print("Error: EmbedOR required")
            return {}
        except Exception as e:
            print(f"Error in EmbedOR: {e}")
            return {}

        if low_edges.size == 0:
            print("Warning: No low-energy edges found")
            return {}

        print("Step 3: Computing embeddings and z-scores...")
        z_scores_dict = {}
        stats_dict = {}
        methods = ['embedor', 'umap', 'tsne']

        for method in methods:
            print(f"  Processing {method}...")
            
            embedding = self.compute_single_embedding(data_subsampled, method)
            
            # Compute z-scores EXACTLY like notebook
            mean_z, std_z, z_scores = self.compute_z_scores_correctly(embedding, all_edges, low_edges)
            
            z_scores_dict[method] = z_scores
            stats_dict[method] = {'mean_z': mean_z, 'std_z': std_z}
            
            del embedding
            gc.collect()

        print("Step 4: Running permutation tests...")
        
        embedor_z = z_scores_dict['embedor']
        
        try:
            print("  Testing UMAP vs EmbedOR...")
            result_umap = permutation_test(
                (embedor_z, z_scores_dict['umap']),
                statistic=lambda x, y: np.mean(x) - np.mean(y),
                batch=100,
                alternative='less',
                n_resamples=self.n_perm,
                random_state=self.seed
            )
            umap_p = result_umap.pvalue
            print(f"    UMAP p-value: {umap_p:.4f}")
        except Exception as e:
            print(f"    UMAP test failed: {e}")
            umap_p = 1.0

        try:
            print("  Testing t-SNE vs EmbedOR...")
            result_tsne = permutation_test(
                (embedor_z, z_scores_dict['tsne']),
                statistic=lambda x, y: np.mean(x) - np.mean(y),
                batch=100,
                alternative='less',
                n_resamples=self.n_perm,
                random_state=self.seed
            )
            tsne_p = result_tsne.pvalue
            print(f"    t-SNE p-value: {tsne_p:.4f}")
        except Exception as e:
            print(f"    t-SNE test failed: {e}")
            tsne_p = 1.0

        # Compile results
        results = {
            'embedor': {
                'z_mean': float(stats_dict['embedor']['mean_z']),
                'z_std': float(stats_dict['embedor']['std_z']),
            },
            'umap': {
                'z_mean': float(stats_dict['umap']['mean_z']),
                'z_std': float(stats_dict['umap']['std_z']),
                'perm_p': float(umap_p)
            },
            'tsne': {
                'z_mean': float(stats_dict['tsne']['mean_z']),
                'z_std': float(stats_dict['tsne']['std_z']),
                'perm_p': float(tsne_p)
            },
            'test_parameters': {
                'beta': self.beta,
                'n_permutations': self.n_perm,
                'alpha': 0.01,
                'max_points': self.max_points,
                'n_low_energy_edges': len(low_edges),
                'n_total_edges': len(all_edges)
            }
        }

        del z_scores_dict, stats_dict, all_edges, low_edges
        gc.collect()

        return results


def load_data_efficiently(file_path: str) -> np.ndarray:
    """Load data with memory mapping."""
    try:
        data = np.load(file_path, mmap_mode='r')
        print(f"Loaded data: {data.shape}, {data.dtype}")
        
        if data.dtype in [np.float64, np.int64]:
            data = data.astype(np.float32)
        
        return data

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Fixed permutation tests")
    parser.add_argument("np_file", type=str, help="Path to .npy dataset file")
    parser.add_argument("--beta", type=float, default=0.33)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--max_points", type=int, default=5000)

    args = parser.parse_args()

    print(f"Loading data from {args.np_file}...")
    data = load_data_efficiently(args.np_file)
    print(f"Original data shape: {data.shape}")

    print("Running FIXED permutation tests...")
    tester = MemoryEfficientEmbeddingTester(
        beta=args.beta,
        n_perm=args.n_perm,
        chunk_size=args.chunk_size,
        seed=args.seed,
        max_points=args.max_points
    )

    results = tester.run_test(data)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if not results:
        print("No results obtained.")
        return

    params = results['test_parameters']
    print(f"Data: {params['max_points']} points")
    print(f"Edges: {params['n_total_edges']} total, {params['n_low_energy_edges']} low-energy")
    print()
    
    print(f"EmbedOR - Mean z-score: {results['embedor']['z_mean']:.4f} ± {results['embedor']['z_std']:.4f}")
    print(f"UMAP    - Mean z-score: {results['umap']['z_mean']:.4f} ± {results['umap']['z_std']:.4f}")
    print(f"t-SNE   - Mean z-score: {results['tsne']['z_mean']:.4f} ± {results['tsne']['z_std']:.4f}")
    print()
    print(f"UMAP p-value:  {results['umap']['perm_p']:.4f}")
    print(f"t-SNE p-value: {results['tsne']['perm_p']:.4f}")

    alpha = 0.01
    umap_sig = results['umap']['perm_p'] < alpha
    tsne_sig = results['tsne']['perm_p'] < alpha

    print(f"\nSignificance at alpha={alpha}:")
    print(f"EmbedOR < UMAP:  {'SIGNIFICANT' if umap_sig else 'not significant'}")
    print(f"EmbedOR < t-SNE: {'SIGNIFICANT' if tsne_sig else 'not significant'}")
    print("="*60)


if __name__ == "__main__":
    main()
