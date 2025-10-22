"""
Combined permutation tests for EmbedOR paper:
- Low ∆E edge expansion (Table 6) 
- High ∆E edge contraction (Table 8)

Two hypothesis tests for each:
Expansion: H1: μ_UMAP > μ_EmbedOR, H1: μ_tSNE > μ_EmbedOR  
Contraction: H1: μ_UMAP > μ_EmbedOR, H1: μ_tSNE > μ_EmbedOR
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

        sorted_idx = np.argsort(edge_energies)
        low_idx = sorted_idx[:k]
        return low_idx

    def get_high_energy_edge_indices(self, embedor) -> np.ndarray:
        """Get indices of highest beta fraction of edges."""
        edge_energies = embedor.distances
        n_edges = len(edge_energies)

        if n_edges == 0:
            return np.array([], dtype=np.int32)

        k = int(self.beta * n_edges)
        k = max(1, min(n_edges, k))

        sorted_idx = np.argsort(edge_energies)
        high_idx = sorted_idx[-k:]
        return high_idx

    def compute_all_edge_distances(self, embedding: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Compute distances for ALL edges."""
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

    def compute_edge_distances_batch(self, embedding: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Compute distances for specific edges."""
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
        low_distances = self.compute_edge_distances_batch(embedding, low_edges)
        
        z_scores = (low_distances - avg_distance) / std_distance
        
        mean_z_score = np.mean(z_scores)
        std_z_score = np.std(z_scores)
        print(f"    Low-energy edges - mean z: {mean_z_score:.4f}, std z: {std_z_score:.4f}")

        del all_distances, low_distances
        gc.collect()
        
        return mean_z_score, std_z_score, z_scores

    def compute_mean_deltaE_for_contracted_edges(self, embedor_ref, embedding: np.ndarray, all_edges: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute μ_ALG_β,• = mean ΔE of smallest beta% embedded edges.
        """
        if all_edges.size == 0:
            return 0.0, np.zeros(0, dtype=np.float32)
        
        print("    Finding smallest embedded edges...")
        embedded_distances = self.compute_all_edge_distances(embedding, all_edges)
        
        n_edges = len(all_edges)
        k = int(self.beta * n_edges)
        k = max(1, min(n_edges, k))
        
        smallest_indices = np.argsort(embedded_distances)[:k]
        smallest_edges = all_edges[smallest_indices]
        
        edge_energies = np.asarray(embedor_ref.distances, dtype=np.float32)
        
        edge_to_index = {}
        for idx, edge in enumerate(all_edges):
            edge_tuple = (edge[0], edge[1])
            edge_to_index[edge_tuple] = idx
        
        contracted_indices = []
        for edge in smallest_edges:
            edge_tuple = (edge[0], edge[1])
            if edge_tuple in edge_to_index:
                contracted_indices.append(edge_to_index[edge_tuple])
        
        contracted_indices = np.array(contracted_indices, dtype=np.int32)
        
        if contracted_indices.size == 0:
            return 0.0, np.zeros(0, dtype=np.float32)
        
        contracted_deltaE = edge_energies[contracted_indices]
        mean_deltaE = np.mean(contracted_deltaE)
        
        del embedded_distances
        gc.collect()
        
        return mean_deltaE, contracted_deltaE

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

    def run_expansion_test(self, data: np.ndarray) -> Dict:
        """Run expansion test (low ∆E edges)."""
        print("="*60)
        print("RUNNING EXPANSION TEST (Low ∆E Edges)")
        print("="*60)
        
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
                'n_total_edges': len(all_edges),
                'test_type': 'expansion'
            }
        }

        del z_scores_dict, stats_dict, all_edges, low_edges
        gc.collect()

        return results
    def run_contraction_test(self, data: np.ndarray) -> Dict:
        """Run contraction test (high ∆E edges)."""
        print("="*60)
        print("RUNNING CONTRACTION TEST (High ∆E Edges)")
        print("="*60)
        
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

            edge_energies = embedor_ref.distances
            print(f"  ΔE range: [{np.min(edge_energies):.4f}, {np.max(edge_energies):.4f}]")

        except ImportError:
            print("Error: EmbedOR required")
            return {}
        except Exception as e:
            print(f"Error in EmbedOR: {e}")
            return {}

        if all_edges.size == 0:
            print("Warning: No edges found")
            return {}

        print("Step 3: Computing embeddings and contraction z-scores...")
        z_scores_dict = {}
        stats_dict = {}
        methods = ['embedor', 'umap', 'tsne']

        for method in methods:
            print(f"  Processing {method}...")
            
            embedding = self.compute_single_embedding(data_subsampled, method)
            
            # Get contracted edges and their ΔE values
            mean_deltaE, contracted_deltaE = self.compute_mean_deltaE_for_contracted_edges(
                embedor_ref, embedding, all_edges
            )
            
            # Compute z-scores for contracted ΔE values (same logic as expansion)
            avg_deltaE_all = np.mean(edge_energies)
            std_deltaE_all = np.std(edge_energies)
            
            contracted_z_scores = (contracted_deltaE - avg_deltaE_all) / std_deltaE_all
            
            mean_z = np.mean(contracted_z_scores)
            std_z = np.std(contracted_z_scores)
            
            z_scores_dict[method] = contracted_z_scores
            stats_dict[method] = {'mean_z': mean_z, 'std_z': std_z}
            
            print(f"    {method.upper()} - Mean z-score: {mean_z:.4f} ± {std_z:.4f} (n={len(contracted_deltaE)})")
            
            del embedding
            gc.collect()

        print("Step 4: Running permutation tests...")
        
        embedor_z = z_scores_dict['embedor']
        
        try:
            print("  Testing UMAP > EmbedOR...")
            result_umap = permutation_test(
                (z_scores_dict['umap'], embedor_z),
                statistic=lambda x, y: np.mean(x) - np.mean(y),
                batch=100,
                alternative='greater',
                n_resamples=self.n_perm,
                random_state=self.seed
            )
            umap_p = result_umap.pvalue
            print(f"    UMAP p-value: {umap_p:.4f}")
        except Exception as e:
            print(f"    UMAP test failed: {e}")
            umap_p = 1.0

        try:
            print("  Testing t-SNE > EmbedOR...")
            result_tsne = permutation_test(
                (z_scores_dict['tsne'], embedor_z),
                statistic=lambda x, y: np.mean(x) - np.mean(y),
                batch=100,
                alternative='greater',
                n_resamples=self.n_perm,
                random_state=self.seed
            )
            tsne_p = result_tsne.pvalue
            print(f"    t-SNE p-value: {tsne_p:.4f}")
        except Exception as e:
            print(f"    t-SNE test failed: {e}")
            tsne_p = 1.0

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
                'n_total_edges': len(all_edges),
                'n_contracted_edges': int(self.beta * len(all_edges)),
                'test_type': 'contraction'
            }
        }

        del z_scores_dict, stats_dict, all_edges
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
    parser = argparse.ArgumentParser(description="Combined expansion and contraction permutation tests")
    parser.add_argument("np_file", type=str, help="Path to .npy dataset file")
    parser.add_argument("--test_type", type=str, choices=['expansion', 'contraction', 'both'], 
                       default='both', help="Which test(s) to run")
    parser.add_argument("--beta", type=float, default=0.33)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--max_points", type=int, default=5000)

    args = parser.parse_args()

    print(f"Loading data from {args.np_file}...")
    data = load_data_efficiently(args.np_file)
    print(f"Original data shape: {data.shape}")

    tester = MemoryEfficientEmbeddingTester(
        beta=args.beta,
        n_perm=args.n_perm,
        chunk_size=args.chunk_size,
        seed=args.seed,
        max_points=args.max_points
    )

    all_results = {}

    if args.test_type in ['expansion', 'both']:
        expansion_results = tester.run_expansion_test(data)
        all_results['expansion'] = expansion_results

    if args.test_type in ['contraction', 'both']:
        contraction_results = tester.run_contraction_test(data)
        all_results['contraction'] = contraction_results

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    alpha = 0.01
    
    if 'expansion' in all_results and all_results['expansion']:
        exp = all_results['expansion']
        params = exp['test_parameters']
        print("EXPANSION TEST RESULTS (Low ∆E Edges)")
        print(f"Data: {params['max_points']} points")
        print(f"Edges: {params['n_total_edges']} total, {params['n_low_energy_edges']} low-energy")
        print()
        print(f"EmbedOR - Mean z-score: {exp['embedor']['z_mean']:.4f} ± {exp['embedor']['z_std']:.4f}")
        print(f"UMAP    - Mean z-score: {exp['umap']['z_mean']:.4f} ± {exp['umap']['z_std']:.4f}")
        print(f"t-SNE   - Mean z-score: {exp['tsne']['z_mean']:.4f} ± {exp['tsne']['z_std']:.4f}")
        print()
        print(f"UMAP p-value:  {exp['umap']['perm_p']:.4f}")
        print(f"t-SNE p-value: {exp['tsne']['perm_p']:.4f}")
        print()
        print(f"Significance at alpha={alpha}:")
        umap_sig = exp['umap']['perm_p'] < alpha
        tsne_sig = exp['tsne']['perm_p'] < alpha
        print(f"EmbedOR < UMAP:  {'SIGNIFICANT' if umap_sig else 'not significant'}")
        print(f"EmbedOR < t-SNE: {'SIGNIFICANT' if tsne_sig else 'not significant'}")
        print("\n" + "-"*40)

    if 'contraction' in all_results and all_results['contraction']:
        cont = all_results['contraction']
        params = cont['test_parameters']
        print("CONTRACTION TEST RESULTS (High ∆E Edges)")
        print(f"Data: {params['max_points']} points")
        print(f"Edges: {params['n_total_edges']} total, {params['n_contracted_edges']} contracted")
        print()
        print(f"EmbedOR - Mean z_score: {cont['embedor']['z_mean']:.4f} ± {cont['embedor']['z_std']:.4f}")
        print(f"UMAP    - Mean z_score: {cont['umap']['z_mean']:.4f} ± {cont['umap']['z_std']:.4f}")
        print(f"t-SNE   - Mean z_score: {cont['tsne']['z_mean']:.4f} ± {cont['tsne']['z_std']:.4f}")
        print()
        print(f"UMAP > EmbedOR p-value:  {cont['umap']['perm_p']:.4f}")
        print(f"t-SNE > EmbedOR p-value: {cont['tsne']['perm_p']:.4f}")
        print()
        print(f"Significance at alpha={alpha}:")
        umap_sig = cont['umap']['perm_p'] < alpha
        tsne_sig = cont['tsne']['perm_p'] < alpha
        print(f"UMAP contracts more high ΔE than EmbedOR:  {'SIGNIFICANT' if umap_sig else 'not significant'}")
        print(f"t-SNE contracts more high ΔE than EmbedOR: {'SIGNIFICANT' if tsne_sig else 'not significant'}")

    print("="*60)


if __name__ == "__main__":
    main()
