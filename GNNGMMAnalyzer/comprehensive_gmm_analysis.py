#!/usr/bin/env python3
"""
Comprehensive GMM Analysis: 4 Combinations
1. Fixed components + No ZCA
2. Fixed components + With ZCA  
3. Adaptive components + No ZCA
4. Adaptive components + With ZCA

For each combination, extract GMM centers and nearest neighbors, save to CSV
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import pandas as pd
import json
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from datetime import datetime

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from gnn_gmm_analysis import GNNGMMAnalyzer

class ComprehensiveGMMAnalyzer:
    """Comprehensive GMM analysis with 4 combinations"""
    
    def __init__(self, embeddings_dir, base_output_dir, max_components=30, zca_threshold=1e-5):
        self.embeddings_dir = embeddings_dir
        self.base_output_dir = Path(base_output_dir)
        self.max_components = max_components
        self.zca_threshold = zca_threshold
        
        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load index mapping
        self.index_mapping = self._load_index_mapping()
        
        # Load embeddings once
        self.layer_embeddings = self._load_embeddings()
        
    def _load_index_mapping(self):
        """Load the correct CAS to original index mapping"""
        try:
            import json
            from pathlib import Path
            
            # Try to load the pre-computed mapping
            mapping_path = Path("/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzer/cas_to_original_mapping.json")
            
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                
                # Convert string keys to integers
                mapping = {int(k): int(v) for k, v in mapping.items()}
                
                print(f"✓ Loaded CAS index mapping: {len(mapping)} entries")
                print("Sample mappings:")
                for i in range(min(5, len(mapping))):
                    print(f"  CAS index {i} -> SMILES index {mapping[i]}")
                
                return mapping
            else:
                print(f"✗ Mapping file not found: {mapping_path}")
                print("Please run analyze_cas_deduplication.py first to generate the mapping")
                return {}
            
        except Exception as e:
            print(f"Warning: Could not create CAS index mapping: {e}")
            print("Using embedding indices directly.")
            return None
    
    def _load_embeddings(self):
        """Load all layer embeddings"""
        print("Loading embeddings...")
        embeddings_path = Path(self.embeddings_dir)
        
        # Define expected layer files and their names (prioritize _cas files)
        layer_mappings = {
            'emb': {'unique_cas': 'emb_unique_cas.pt', 'unique': 'emb_unique.pt', 'original': 'emb.pt'},
            'output_block_0': {'unique_cas': 'output0_unique_cas.pt', 'unique': 'output0_unique.pt', 'original': 'output0.pt'},
            'output_block_1': {'unique_cas': 'output1_unique_cas.pt', 'unique': 'output1_unique.pt', 'original': 'output1.pt'},
            'output_block_2': {'unique_cas': 'output2_unique_cas.pt', 'unique': 'output2_unique.pt', 'original': 'output2.pt'},
            'output_block_3': {'unique_cas': 'output3_unique_cas.pt', 'unique': 'output3_unique.pt', 'original': 'output3.pt'},
            'output_block_4': {'unique_cas': 'output4_unique_cas.pt', 'unique': 'output4_unique.pt', 'original': 'output4.pt'},
            'u_mlp_input': {'unique_cas': 'u_mlp_input_cas.pt', 'unique': 'u_mlp_input.pt', 'original': 'u_mlp_input.pt'},
            'u_mlp_l1': {'unique_cas': 'u_mlp_l1_cas.pt', 'unique': 'u_mlp_l1.pt', 'original': 'u_mlp_l1.pt'},
            'u_mlp_l2': {'unique_cas': 'u_mlp_l2_cas.pt', 'unique': 'u_mlp_l2.pt', 'original': 'u_mlp_l2.pt'}
        }
        
        layer_embeddings = {}
        
        for layer_name in ['emb', 'output_block_0', 'output_block_1', 'output_block_2', 'output_block_3', 'output_block_4', 
                           'u_mlp_input', 'u_mlp_l1', 'u_mlp_l2']:
            
            # Try unique_cas file first, then unique file, then fallback to original
            unique_cas_filename = layer_mappings[layer_name]['unique_cas']
            unique_filename = layer_mappings[layer_name]['unique']
            original_filename = layer_mappings[layer_name]['original']
            
            unique_cas_filepath = embeddings_path / unique_cas_filename
            unique_filepath = embeddings_path / unique_filename
            original_filepath = embeddings_path / original_filename
            
            if unique_cas_filepath.exists():
                filepath = unique_cas_filepath
                filename = unique_cas_filename
                print(f"  Loading {filename} -> {layer_name} (unique_cas)")
            elif unique_filepath.exists():
                filepath = unique_filepath
                filename = unique_filename
                print(f"  Loading {filename} -> {layer_name} (unique)")
            elif original_filepath.exists():
                filepath = original_filepath
                filename = original_filename
                print(f"  Loading {filename} -> {layer_name} (original)")
            else:
                print(f"  Warning: Neither {unique_cas_filename}, {unique_filename}, nor {original_filename} found")
                continue
            
            try:
                embeddings_tensor = torch.load(filepath, map_location='cpu')
                embeddings_np = embeddings_tensor.detach().cpu().numpy()
                
                if embeddings_np.ndim > 2:
                    embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)
                
                layer_embeddings[layer_name] = embeddings_np
                print(f"    Shape: {embeddings_np.shape}")
                
            except Exception as e:
                print(f"    Error loading {filename}: {e}")
        
        print(f"Successfully loaded embeddings from {len(layer_embeddings)} layers")
        return layer_embeddings
    
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
    
    def find_optimal_components(self, embeddings, use_zca=True):
        """Find optimal number of components"""
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        if use_zca:
            embeddings_processed, _ = self.adaptive_zca_whitening(embeddings_scaled, threshold=self.zca_threshold)
        else:
            embeddings_processed = embeddings_scaled
        
        n_components_range = range(2, self.max_components + 1)
        aic_scores = []
        
        for n_components in n_components_range:
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42,
                    max_iter=100
                )
                gmm.fit(embeddings_processed)
                aic_scores.append(gmm.aic(embeddings_processed))
            except:
                aic_scores.append(float('inf'))
        
        optimal_aic = n_components_range[np.argmin(aic_scores)]
        return optimal_aic
    
    def fit_gmm_and_extract_centers(self, embeddings, n_components, use_zca=True, layer_name="layer"):
        """Fit GMM and extract center data points and nearest neighbors"""
        print(f"  Fitting GMM for {layer_name} with {n_components} components, ZCA: {use_zca}")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        if use_zca:
            embeddings_processed, zca_transform = self.adaptive_zca_whitening(embeddings_scaled, threshold=self.zca_threshold)
        else:
            embeddings_processed = embeddings_scaled
            zca_transform = None
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        gmm.fit(embeddings_processed)
        
        # Get GMM means (centers)
        gmm_means = gmm.means_
        
        # Transform means back to original space for nearest neighbor search
        if use_zca:
            # Transform means back from whitened space to scaled space
            means_scaled = gmm_means @ zca_transform
            # Then inverse transform from scaled space to original space
            means_original = scaler.inverse_transform(means_scaled)
        else:
            means_original = scaler.inverse_transform(gmm_means)
        
        # Get component assignments for each data point
        component_labels = gmm.predict(embeddings_processed)
        
        center_data = []
        used_indices = set()  # Track used data points to avoid overlap across components
        
        for comp_idx in range(n_components):
            # Get all data points assigned to this component
            component_mask = (component_labels == comp_idx)
            component_indices = np.where(component_mask)[0]
            component_embeddings = embeddings[component_indices]
            
            if len(component_indices) == 0:
                # Skip empty components
                continue
            
            print(f"    Component {comp_idx}: {len(component_indices)} members")
            
            # Find the data point closest to the GMM center (this is our "center data point")
            distances_to_center = np.linalg.norm(component_embeddings - means_original[comp_idx], axis=1)
            closest_member_idx = np.argmin(distances_to_center)
            center_data_point_idx = component_indices[closest_member_idx]
            
            # Get original molecule index if mapping is available
            original_molecule_index = self.index_mapping.get(center_data_point_idx, center_data_point_idx) if self.index_mapping else center_data_point_idx
            
            # Store the center data point
            center_data.append({
                'layer': layer_name,
                'component': comp_idx,
                'component_weight': gmm.weights_[comp_idx],
                'center_coordinates': means_original[comp_idx].tolist(),
                'neighbor_rank': 0,  # Rank 0 for the center itself
                'neighbor_index': int(center_data_point_idx),  # Embedding index
                'original_molecule_index': int(original_molecule_index),  # Original SMILES index
                'neighbor_coordinates': embeddings[center_data_point_idx].tolist(),
                'distance_to_center': float(distances_to_center[closest_member_idx]),
                'is_center': True,
                'component_size': len(component_indices)
            })
            
            # Mark this center as used
            used_indices.add(center_data_point_idx)
            
            # Find nearest neighbors within this component only
            # Sort component members by distance to center (excluding the center itself)
            member_distances = [(idx, dist) for idx, dist in zip(component_indices, distances_to_center) 
                              if idx != center_data_point_idx]
            member_distances.sort(key=lambda x: x[1])  # Sort by distance
            
            # Determine how many neighbors to select
            n_neighbors_needed = min(10, len(component_indices) - 1)  # -1 because we exclude the center
            
            # Select neighbors, ensuring no overlap with other components
            neighbors_selected = 0
            for member_idx, distance in member_distances:
                if neighbors_selected >= n_neighbors_needed:
                    break
                    
                # Only use this neighbor if it hasn't been used by another component
                if member_idx not in used_indices:
                    # Get original molecule index if mapping is available
                    original_molecule_index = self.index_mapping.get(member_idx, member_idx) if self.index_mapping else member_idx
                    
                    center_data.append({
                        'layer': layer_name,
                        'component': comp_idx,
                        'component_weight': gmm.weights_[comp_idx],
                        'center_coordinates': means_original[comp_idx].tolist(),
                        'neighbor_rank': neighbors_selected + 1,
                        'neighbor_index': int(member_idx),  # Embedding index
                        'original_molecule_index': int(original_molecule_index),  # Original SMILES index
                        'neighbor_coordinates': embeddings[member_idx].tolist(),
                        'distance_to_center': float(distance),
                        'is_center': False,
                        'component_size': len(component_indices)
                    })
                    
                    used_indices.add(member_idx)
                    neighbors_selected += 1
            
            print(f"    Component {comp_idx}: Selected {neighbors_selected + 1} points (1 center + {neighbors_selected} neighbors)")
        
        return center_data, gmm, scaler, zca_transform, embeddings_processed
    
    def run_combination(self, combination_name, use_fixed_components, use_zca):
        """Run analysis for one combination"""
        print(f"\n{'='*60}")
        print(f"Running combination: {combination_name}")
        print(f"Fixed components: {use_fixed_components}, ZCA: {use_zca}")
        print(f"{'='*60}")
        
        # Create output directory for this combination
        output_dir = self.base_output_dir / combination_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine optimal components for final layer
        final_layer = list(self.layer_embeddings.keys())[-1]
        optimal_components = self.find_optimal_components(
            self.layer_embeddings[final_layer], 
            use_zca=use_zca
        )
        
        print(f"Optimal components for {final_layer}: {optimal_components}")
        
        # Run analysis for each layer
        all_center_data = []
        gmm_results = {}
        
        for layer_name, embeddings in self.layer_embeddings.items():
            print(f"\nProcessing layer: {layer_name}")
            
            if use_fixed_components:
                # Use the same number of components for all layers
                n_components = optimal_components
            else:
                # Find optimal components for each layer individually
                n_components = self.find_optimal_components(embeddings, use_zca=use_zca)
                print(f"  Layer-specific optimal components: {n_components}")
            
            # Fit GMM and extract centers
            center_data, gmm, scaler, zca_transform, embeddings_processed = self.fit_gmm_and_extract_centers(
                embeddings, n_components, use_zca=use_zca, layer_name=layer_name
            )
            
            all_center_data.extend(center_data)
            
            # Store GMM results
            gmm_results[layer_name] = {
                'gmm': gmm,
                'scaler': scaler,
                'zca_transform': zca_transform,
                'embeddings_processed': embeddings_processed,
                'embeddings_original': embeddings,
                'n_components': n_components,
                'labels': gmm.predict(embeddings_processed),
                'probabilities': gmm.predict_proba(embeddings_processed)
            }
        
        # Save center and nearest neighbor data to CSV
        df_centers = pd.DataFrame(all_center_data)
        centers_csv_path = output_dir / 'gmm_centers_and_neighbors.csv'
        df_centers.to_csv(centers_csv_path, index=False)
        print(f"\nCenter and neighbor data saved to: {centers_csv_path}")
        
        # Save summary statistics
        summary_data = []
        for layer_name in self.layer_embeddings.keys():
            gmm_result = gmm_results[layer_name]
            n_components = gmm_result['n_components']
            
            # Calculate some basic statistics
            labels = gmm_result['labels']
            component_sizes = [np.sum(labels == i) for i in range(n_components)]
            
            summary_data.append({
                'layer': layer_name,
                'n_components': n_components,
                'min_component_size': np.min(component_sizes),
                'max_component_size': np.max(component_sizes),
                'avg_component_size': np.mean(component_sizes),
                'std_component_size': np.std(component_sizes),
                'use_zca': use_zca,
                'use_fixed_components': use_fixed_components
            })
        
        df_summary = pd.DataFrame(summary_data)
        summary_csv_path = output_dir / 'analysis_summary.csv'
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"Summary statistics saved to: {summary_csv_path}")
        
        # Save GMM results (pickle for later analysis if needed)
        import pickle
        results_path = output_dir / 'gmm_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(gmm_results, f)
        print(f"GMM results saved to: {results_path}")
        
        return gmm_results, all_center_data, output_dir
    
    def run_all_combinations(self):
        """Run all 4 combinations"""
        print("Starting comprehensive GMM analysis with 4 combinations...")
        start_time = time.time()
        
        combinations = [
            ("fixed_no_zca", True, False),
            ("fixed_with_zca", True, True),
            ("adaptive_no_zca", False, False),
            ("adaptive_with_zca", False, True)
        ]
        
        results = {}
        
        for combination_name, use_fixed_components, use_zca in combinations:
            try:
                gmm_results, center_data, output_dir = self.run_combination(
                    combination_name, use_fixed_components, use_zca
                )
                results[combination_name] = {
                    'gmm_results': gmm_results,
                    'center_data': center_data,
                    'output_dir': output_dir
                }
                print(f"{combination_name} completed successfully")
                
            except Exception as e:
                print(f"{combination_name} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Create overall summary
        self._create_overall_summary(results)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"All combinations completed in {total_time:.2f} seconds")
        print(f"Results saved in: {self.base_output_dir}")
        print(f"{'='*60}")
        
        return results
    
    def _create_overall_summary(self, results):
        """Create overall summary comparing all combinations"""
        print("\nCreating overall summary...")
        
        summary_data = []
        
        for combination_name, result in results.items():
            if result is None:
                continue
                
            gmm_results = result['gmm_results']
            center_data = result['center_data']
            
            # Extract combination parameters
            use_zca = 'with_zca' in combination_name
            use_fixed = 'fixed' in combination_name
            
            # Count total centers across all layers
            total_centers = len([d for d in center_data if d['neighbor_rank'] == 0])
            
            # Count unique nearest neighbors
            unique_neighbors = len(set(d['neighbor_index'] for d in center_data))
            
            summary_data.append({
                'combination': combination_name,
                'use_zca': use_zca,
                'use_fixed_components': use_fixed,
                'total_gmm_centers': total_centers,
                'unique_nearest_neighbors': unique_neighbors,
                'total_data_points': len(center_data),
                'layers_analyzed': len(gmm_results)
            })
        
        df_overall = pd.DataFrame(summary_data)
        overall_summary_path = self.base_output_dir / 'overall_summary.csv'
        df_overall.to_csv(overall_summary_path, index=False)
        print(f"Overall summary saved to: {overall_summary_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("OVERALL SUMMARY:")
        print("="*60)
        for _, row in df_overall.iterrows():
            print(f"{row['combination']}:")
            print(f"  ZCA: {row['use_zca']}, Fixed components: {row['use_fixed_components']}")
            print(f"  Total GMM centers: {row['total_gmm_centers']}")
            print(f"  Unique nearest neighbors: {row['unique_nearest_neighbors']}")
            print(f"  Layers analyzed: {row['layers_analyzed']}")
            print()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive GMM Analysis - 4 Combinations")
    parser.add_argument('--embeddings_dir', type=str, 
                       default="/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings",
                       help='Directory containing pre-saved embeddings')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory for all results')
    parser.add_argument('--max_components', type=int, default=30,
                       help='Maximum number of components to test')
    parser.add_argument('--zca_threshold', type=float, default=1e-5,
                       help='Threshold for ZCA whitening')
    
    args = parser.parse_args()
    
    # Create default output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzerResults/comprehensive_analysis_{timestamp}"
    
    print("Starting Comprehensive GMM Analysis...")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max components: {args.max_components}")
    print(f"ZCA threshold: {args.zca_threshold}")
    
    # Create analyzer and run all combinations
    analyzer = ComprehensiveGMMAnalyzer(
        embeddings_dir=args.embeddings_dir,
        base_output_dir=args.output_dir,
        max_components=args.max_components,
        zca_threshold=args.zca_threshold
    )
    
    results = analyzer.run_all_combinations()
    
    print("\n All analyses completed successfully!")
    print(f"Results saved in: {args.output_dir}")
    print("\n Each combination folder contains:")
    print("  - gmm_centers_and_neighbors.csv (center points and nearest neighbors)")
    print("  - analysis_summary.csv (layer-wise summary)")
    print("  - gmm_results.pkl (full GMM results)")
    print("  - overall_summary.csv (comparison across all combinations)")

if __name__ == "__main__":
    main()
