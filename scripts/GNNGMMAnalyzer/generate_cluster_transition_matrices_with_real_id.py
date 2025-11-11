#!/usr/bin/env python3
"""
Generate Cluster Transition Matrices with Real ID Tracking

This script generates cluster transition matrices in the same format as cluster_transition_matrices_1.csv,
but with real_id tracking added. It matches:
- Same number of components (13)
- Same layers (emb -> output_block_0 -> ... -> output_block_4)
- Same format (from_layer, to_layer, from_cluster, to_cluster, count, probability)
- Plus real_id for tracking individual points
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import json

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


class ClusterTransitionMatrixGenerator:
    """Generate cluster transition matrices with real ID tracking"""
    
    def __init__(self, embeddings_dir, output_dir, n_components=13, zca_threshold=1e-5):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.n_components = n_components
        self.zca_threshold = zca_threshold
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CAS to original index mapping
        self.cas_to_original_mapping = self._load_cas_mapping()
        
        # Layer names matching cluster_transition_matrices_1.csv (only first 5 layers)
        self.layer_files = {
            'emb': 'emb_unique_cas.pt',
            'output_block_0': 'output0_unique_cas.pt',
            'output_block_1': 'output1_unique_cas.pt',
            'output_block_2': 'output2_unique_cas.pt',
            'output_block_3': 'output3_unique_cas.pt',
            'output_block_4': 'output4_unique_cas.pt',
        }
        
        # Only use these layers (matching cluster_transition_matrices_1.csv)
        self.layer_names = ['emb', 'output_block_0', 'output_block_1', 'output_block_2', 'output_block_3', 'output_block_4']
        
    def _load_cas_mapping(self):
        """Load CAS to original index mapping"""
        script_dir = Path(__file__).parent
        mapping_paths = [
            script_dir / "cas_to_original_mapping.json",
            script_dir / "cas_index_mapping.json",
        ]
        
        for mapping_path in mapping_paths:
            if mapping_path.exists():
                print(f"Loading CAS to original mapping from: {mapping_path}")
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                
                mapping = {int(k): int(v) for k, v in mapping.items()}
                
                print(f"✓ Loaded CAS index mapping: {len(mapping)} entries")
                print(f"  CAS index range: {min(mapping.keys())} - {max(mapping.keys())}")
                print(f"  Original index range: {min(mapping.values())} - {max(mapping.values())}")
                
                return mapping
        
        print("⚠ Warning: CAS to original mapping file not found")
        return None
    
    def adaptive_zca_whitening(self, X, threshold=1e-5, epsilon=1e-5):
        """Apply ZCA whitening"""
        sigma = np.cov(X, rowvar=False)
        U, S, V = np.linalg.svd(sigma)
        
        k = 0
        for v in S:
            if v > threshold:
                k += 1
        
        top_U = U[:, :k]
        top_S = S[:k]
        red_zca = np.dot(np.diag(1.0 / np.sqrt(top_S + epsilon)), top_U.T)
        X_whitened = X @ red_zca.T
        
        return X_whitened, red_zca
    
    def load_embeddings(self):
        """Load embeddings for the specified layers"""
        print("\nLoading embeddings...")
        embeddings = {}
        
        for layer_name in self.layer_names:
            filename = self.layer_files[layer_name]
            filepath = self.embeddings_dir / filename
            
            if not filepath.exists():
                print(f"  ✗ {filename} not found, skipping {layer_name}")
                continue
            
            try:
                data = torch.load(filepath, map_location='cpu')
                if isinstance(data, torch.Tensor):
                    data = data.squeeze().numpy()
                else:
                    data = np.array(data)
                
                if data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)
                
                embeddings[layer_name] = data
                print(f"  ✓ {layer_name}: shape {data.shape}")
                
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
                continue
        
        print(f"\nSuccessfully loaded {len(embeddings)} layers")
        return embeddings
    
    def fit_gmm_with_components(self, embeddings, n_components):
        """Fit GMM with specified number of components"""
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply ZCA whitening
        embeddings_processed, zca_transform = self.adaptive_zca_whitening(
            embeddings_scaled, threshold=self.zca_threshold
        )
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        gmm.fit(embeddings_processed)
        
        # Get cluster assignments
        labels = gmm.predict(embeddings_processed)
        probabilities = gmm.predict_proba(embeddings_processed)
        
        return {
            'gmm': gmm,
            'labels': labels,
            'probabilities': probabilities,
            'means': gmm.means_,
            'covariances': gmm.covariances_,
            'weights': gmm.weights_,
            'embeddings_processed': embeddings_processed  # Store processed embeddings for alignment
        }
    
    def _find_optimal_alignment_from_transition(self, transition_matrix_norm, n_components):
        """
        Find optimal cluster alignment from transition matrix.
        Uses Hungarian algorithm to find best matching based on transition probabilities.
        """
        from scipy.optimize import linear_sum_assignment
        
        # Use negative probabilities as costs (we want to maximize probabilities)
        cost_matrix = -transition_matrix_norm
        
        # Find optimal assignment using Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create permutation array
        permutation = np.zeros(n_components, dtype=int)
        permutation[row_indices] = col_indices
        
        # Fill any missing assignments
        used = set(col_indices)
        unused = [i for i in range(n_components) if i not in used]
        unused_idx = 0
        for i in range(n_components):
            if i not in row_indices:
                if unused_idx < len(unused):
                    permutation[i] = unused[unused_idx]
                    unused_idx += 1
                else:
                    # Fallback: use identity
                    permutation[i] = i
        
        return permutation
    
    def generate_transition_matrices(self, embeddings):
        """
        Generate transition matrices in the same format as cluster_transition_matrices_1.csv
        but with real_id tracking added.
        """
        print(f"\n{'='*60}")
        print("Generating Cluster Transition Matrices")
        print(f"{'='*60}")
        print(f"Using {self.n_components} components")
        print(f"Layers: {' -> '.join(self.layer_names)}")
        
        # Fit GMM for each layer
        print("\nFitting GMMs for each layer...")
        gmm_results = {}
        
        for layer_name in self.layer_names:
            if layer_name not in embeddings:
                continue
            print(f"  Fitting GMM for {layer_name}...")
            embeddings_data = embeddings[layer_name]
            gmm_results[layer_name] = self.fit_gmm_with_components(embeddings_data, self.n_components)
        
        # Generate transition matrices (aggregate format)
        print("\nGenerating aggregate transition matrices...")
        transition_data = []
        
        # Generate individual point tracking
        print("Generating individual point tracking...")
        individual_transition_data = []
        
        for i in range(len(self.layer_names) - 1):
            previous_layer = self.layer_names[i]
            current_layer = self.layer_names[i+1]
            
            if previous_layer not in gmm_results or current_layer not in gmm_results:
                continue
            
            previous_labels = gmm_results[previous_layer]['labels']
            current_labels = gmm_results[current_layer]['labels']
            
            # Create transition matrix
            transition_matrix = np.zeros((self.n_components, self.n_components))
            for prev_cluster in range(self.n_components):
                prev_mask = previous_labels == prev_cluster
                if np.sum(prev_mask) > 0:
                    current_clusters = current_labels[prev_mask]
                    for curr_cluster in range(self.n_components):
                        transition_matrix[prev_cluster, curr_cluster] = np.sum(current_clusters == curr_cluster)
            
            # Normalize by row sums
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix_norm = transition_matrix / (row_sums + 1e-10)
            
            # Use K-NN approach for better alignment
            # Use transition matrix for alignment instead of K-NN (simpler and more reliable)
            # Find permutation that maximizes diagonal of transition matrix
            # This matches the approach: align clusters based on where most points go
            optimal_permutation = self._find_optimal_alignment_from_transition(
                transition_matrix_norm, self.n_components
            )
            
            # Apply permutation to columns
            transition_matrix_norm_aligned = transition_matrix_norm[:, optimal_permutation]
            transition_matrix_aligned = transition_matrix[:, optimal_permutation]
            
            # Export aggregate format (same as cluster_transition_matrices_1.csv)
            for prev_cluster in range(self.n_components):
                for curr_cluster in range(self.n_components):
                    transition_data.append({
                        'from_layer': previous_layer,
                        'to_layer': current_layer,
                        'from_cluster': prev_cluster,
                        'to_cluster': curr_cluster,
                        'count': float(transition_matrix_aligned[prev_cluster, curr_cluster]),
                        'probability': float(transition_matrix_norm_aligned[prev_cluster, curr_cluster])
                    })
            
            # Export individual point tracking with real IDs
            n_points = len(previous_labels)
            for cas_id in range(n_points):
                # Map CAS index to original dataset index
                if self.cas_to_original_mapping is not None and cas_id in self.cas_to_original_mapping:
                    original_id = self.cas_to_original_mapping[cas_id]
                else:
                    original_id = cas_id
                
                # Apply permutation to current cluster label
                original_curr_cluster = int(current_labels[cas_id])
                aligned_curr_cluster = int(np.where(optimal_permutation == original_curr_cluster)[0][0])
                
                individual_transition_data.append({
                    'real_id': int(original_id),
                    'from_layer': previous_layer,
                    'to_layer': current_layer,
                    'from_cluster': int(previous_labels[cas_id]),
                    'to_cluster': aligned_curr_cluster
                })
        
        # Save aggregate format (matching cluster_transition_matrices_1.csv)
        df_transitions = pd.DataFrame(transition_data)
        output_path = self.output_dir / 'cluster_transition_matrices_with_real_id.csv'
        df_transitions.to_csv(output_path, index=False)
        print(f"\n✓ Saved aggregate transition matrix: {output_path}")
        print(f"  Total transitions: {len(df_transitions)}")
        print(f"  Components: 0-{self.n_components-1}")
        print(f"  Format matches cluster_transition_matrices_1.csv")
        
        # Save individual point tracking
        df_individual = pd.DataFrame(individual_transition_data)
        individual_path = self.output_dir / 'cluster_transition_matrices_individual_real_id.csv'
        df_individual.to_csv(individual_path, index=False)
        print(f"\n✓ Saved individual point tracking: {individual_path}")
        print(f"  Total point transitions: {len(df_individual)}")
        print(f"  Unique real IDs: {df_individual['real_id'].nunique()}")
        print(f"  Real ID range: {df_individual['real_id'].min()} - {df_individual['real_id'].max()}")
        
        return df_transitions, df_individual
    
    def run(self):
        """Run the complete generation pipeline"""
        print("="*60)
        print("Cluster Transition Matrix Generator with Real ID")
        print("="*60)
        print(f"Embeddings directory: {self.embeddings_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Number of components: {self.n_components}")
        print(f"ZCA threshold: {self.zca_threshold}")
        print("="*60)
        
        # Load embeddings
        embeddings = self.load_embeddings()
        if not embeddings:
            print("❌ No embeddings loaded. Exiting.")
            return
        
        # Generate transition matrices
        df_aggregate, df_individual = self.generate_transition_matrices(embeddings)
        
        print("\n" + "="*60)
        print("✅ Generation Complete")
        print("="*60)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        print("  1. cluster_transition_matrices_with_real_id.csv")
        print("     - Aggregate format (matches cluster_transition_matrices_1.csv)")
        print("     - Columns: from_layer, to_layer, from_cluster, to_cluster, count, probability")
        print("  2. cluster_transition_matrices_individual_real_id.csv")
        print("     - Individual point tracking with real IDs")
        print("     - Columns: real_id, from_layer, to_layer, from_cluster, to_cluster")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Cluster Transition Matrices with Real ID Tracking"
    )
    parser.add_argument('--embeddings_dir', type=str, required=True,
                       help='Directory containing pre-saved embeddings')
    parser.add_argument('--output_dir', type=str, 
                       default='../../frontend/public/data',
                       help='Output directory for CSV files')
    parser.add_argument('--n_components', type=int, default=13,
                       help='Number of components (default: 13, matching cluster_transition_matrices_1.csv)')
    parser.add_argument('--zca_threshold', type=float, default=1e-5,
                       help='Threshold for ZCA whitening')
    
    args = parser.parse_args()
    
    # Resolve output directory path
    script_dir = Path(__file__).parent
    if args.output_dir.startswith('../') or args.output_dir.startswith('../../'):
        output_dir = (script_dir / args.output_dir).resolve()
    else:
        output_dir = Path(args.output_dir)
    
    # Create generator and run
    generator = ClusterTransitionMatrixGenerator(
        embeddings_dir=args.embeddings_dir,
        output_dir=output_dir,
        n_components=args.n_components,
        zca_threshold=args.zca_threshold
    )
    
    generator.run()


if __name__ == "__main__":
    main()

