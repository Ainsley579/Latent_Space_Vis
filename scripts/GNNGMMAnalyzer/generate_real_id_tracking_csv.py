#!/usr/bin/env python3
"""
Generate Real ID Tracking CSV Files

This script generates two separate CSV files:
1. real_id_layer_transitions.csv - Tracks real IDs through layer transitions (Sankey through layers)
2. real_id_component_transitions.csv - Tracks real IDs through component count changes for one layer

These files allow tracking individual points through the Sankey diagram and component changes.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import from existing analysis scripts
try:
    from gnn_gmm_analysis import GNNGMMAnalyzer
    from component_flow_analysis import ComponentFlowAnalyzer
except ImportError:
    print("Warning: Could not import from existing analysis scripts")
    GNNGMMAnalyzer = None
    ComponentFlowAnalyzer = None


class RealIDTracker:
    """Track real IDs through layer and component transitions"""
    
    def __init__(self, embeddings_dir, output_dir, zca_threshold=1e-5):
        self.embeddings_dir = embeddings_dir
        self.output_dir = Path(output_dir)
        self.zca_threshold = zca_threshold
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CAS to original index mapping
        self.cas_to_original_mapping = self._load_cas_mapping()
        
        # Load embeddings
        self.layer_embeddings = self._load_embeddings()
        
        # Component counts to test for component flow analysis
        self.component_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    def _load_cas_mapping(self):
        """Load CAS to original index mapping"""
        import json
        
        # Try to find the mapping file in the script directory
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
                
                # Convert string keys to integers
                mapping = {int(k): int(v) for k, v in mapping.items()}
                
                print(f"✓ Loaded CAS index mapping: {len(mapping)} entries")
                print(f"  CAS index range: {min(mapping.keys())} - {max(mapping.keys())}")
                print(f"  Original index range: {min(mapping.values())} - {max(mapping.values())}")
                print("  Sample mappings:")
                for i in range(min(5, len(mapping))):
                    print(f"    CAS {i} -> Original {mapping[i]}")
                
                return mapping
        
        print("⚠ Warning: CAS to original mapping file not found")
        print("  Will use CAS indices directly (not mapped to original dataset)")
        print("  Expected files: cas_to_original_mapping.json or cas_index_mapping.json")
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
                print("  Loading {} -> {} (unique_cas)".format(filename, layer_name))
            elif unique_filepath.exists():
                filepath = unique_filepath
                filename = unique_filename
                print("  Loading {} -> {} (unique)".format(filename, layer_name))
            elif original_filepath.exists():
                filepath = original_filepath
                filename = original_filename
                print("  Loading {} -> {} (original)".format(filename, layer_name))
            else:
                print("  Warning: Neither {}, {}, nor {} found".format(unique_cas_filename, unique_filename, original_filename))
                continue
            
            try:
                import torch
                embeddings_tensor = torch.load(filepath, map_location='cpu')
                embeddings_np = embeddings_tensor.detach().cpu().numpy()
                
                if embeddings_np.ndim > 2:
                    embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)
                
                layer_embeddings[layer_name] = embeddings_np
                print("    Shape: {}".format(embeddings_np.shape))
                
            except Exception as e:
                print("    Error loading {}: {}".format(filename, e))
        
        print("Successfully loaded embeddings from {} layers".format(len(layer_embeddings)))
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
    
    def fit_gmm_with_components(self, embeddings, n_components):
        """Fit GMM with specified number of components"""
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply ZCA whitening
        embeddings_processed, zca_transform = self.adaptive_zca_whitening(embeddings_scaled, threshold=self.zca_threshold)
        
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
            'weights': gmm.weights_
        }
    
    def generate_layer_transition_tracking(self, n_components=None):
        """
        Generate CSV tracking real IDs through layer transitions (Sankey through layers)
        
        This tracks each point's cluster assignment as it moves through layers.
        Saves to a separate folder: real_id_layer_transitions/
        """
        print("\n" + "="*60)
        print("Generating Real ID Layer Transition Tracking")
        print("="*60)
        
        if len(self.layer_embeddings) == 0:
            print("Error: No embeddings loaded")
            return
        
        # Create output folder for layer transitions
        layer_transitions_dir = self.output_dir / 'real_id_layer_transitions'
        layer_transitions_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine optimal components if not provided
        if n_components is None:
            print("Finding optimal number of components...")
            # Use first layer to determine optimal components
            first_layer = list(self.layer_embeddings.keys())[0]
            first_embeddings = self.layer_embeddings[first_layer]
            
            # Try different component counts and use AIC
            best_aic = np.inf
            best_n = 2
            for n in range(2, 21):
                try:
                    gmm_result = self.fit_gmm_with_components(first_embeddings, n)
                    # Calculate AIC using the GMM's built-in method
                    # Get processed embeddings for scoring
                    scaler = StandardScaler()
                    embeddings_scaled = scaler.fit_transform(first_embeddings)
                    embeddings_processed, _ = self.adaptive_zca_whitening(embeddings_scaled, threshold=self.zca_threshold)
                    
                    # Use GMM's score_samples for log likelihood
                    log_likelihood = np.sum(gmm_result['gmm'].score_samples(embeddings_processed))
                    n_samples = embeddings_processed.shape[0]
                    n_features = embeddings_processed.shape[1]
                    n_params = n * (1 + n_features + n_features * (n_features + 1) / 2) - 1
                    aic = 2 * n_params - 2 * log_likelihood
                    if aic < best_aic:
                        best_aic = aic
                        best_n = n
                except Exception as e:
                    print(f"    Warning: Could not fit {n} components: {e}")
                    continue
            n_components = best_n
            print(f"Optimal components: {n_components}")
        
        # Fit GMM for each layer
        print("Fitting GMMs for each layer...")
        gmm_results = {}
        layer_names = list(self.layer_embeddings.keys())
        
        for layer_name in layer_names:
            print(f"  Fitting GMM for {layer_name}...")
            embeddings = self.layer_embeddings[layer_name]
            gmm_results[layer_name] = self.fit_gmm_with_components(embeddings, n_components)
        
        # Track individual point transitions through layers
        print("Tracking point transitions through layers...")
        transition_data = []
        
        for i in range(len(layer_names) - 1):
            previous_layer = layer_names[i]
            current_layer = layer_names[i+1]
            
            previous_labels = gmm_results[previous_layer]['labels']
            current_labels = gmm_results[current_layer]['labels']
            
            # For each point, track its transition
            n_points = len(previous_labels)
            for cas_id in range(n_points):
                # Map CAS index to original dataset index if mapping is available
                if self.cas_to_original_mapping is not None and cas_id in self.cas_to_original_mapping:
                    original_id = self.cas_to_original_mapping[cas_id]
                else:
                    original_id = cas_id
                
                transition_data.append({
                    'real_id': int(original_id),
                    'from_layer': previous_layer,
                    'to_layer': current_layer,
                    'from_cluster': int(previous_labels[cas_id]),
                    'to_cluster': int(current_labels[cas_id])
                })
        
        # Save to CSV in the layer transitions folder
        df_transitions = pd.DataFrame(transition_data)
        output_path = layer_transitions_dir / 'real_id_layer_transitions.csv'
        df_transitions.to_csv(output_path, index=False)
        print(f"\n✓ Layer transition tracking saved to: {output_path}")
        print(f"  Total transitions: {len(df_transitions)}")
        print(f"  Unique real IDs: {df_transitions['real_id'].nunique()}")
        
        return df_transitions
    
    def generate_component_transition_tracking(self, layer_name=None):
        """
        Generate CSV tracking real IDs through component count changes for one layer
        
        This tracks each point's cluster assignment as the number of components changes.
        Saves to a separate folder: real_id_component_transitions/
        """
        print("\n" + "="*60)
        print("Generating Real ID Component Transition Tracking")
        print("="*60)
        
        if len(self.layer_embeddings) == 0:
            print("Error: No embeddings loaded")
            return
        
        # Create output folder for component transitions
        component_transitions_dir = self.output_dir / 'real_id_component_transitions'
        component_transitions_dir.mkdir(parents=True, exist_ok=True)
        
        # Use first layer if not specified
        if layer_name is None:
            layer_name = list(self.layer_embeddings.keys())[0]
            print(f"Using layer: {layer_name}")
        
        if layer_name not in self.layer_embeddings:
            print(f"Error: Layer {layer_name} not found")
            return
        
        embeddings = self.layer_embeddings[layer_name]
        print(f"Layer {layer_name} embeddings shape: {embeddings.shape}")
        
        # Fit GMMs with different component counts
        print("Fitting GMMs with different component counts...")
        gmm_results = {}
        
        for n_components in self.component_counts:
            print(f"  Fitting GMM with {n_components} components...")
            gmm_results[n_components] = self.fit_gmm_with_components(embeddings, n_components)
        
        # Track individual point transitions through component counts
        print("Tracking point transitions through component counts...")
        transition_data = []
        
        for i in range(len(self.component_counts) - 1):
            current_n_components = self.component_counts[i]
            next_n_components = self.component_counts[i + 1]
            
            current_labels = gmm_results[current_n_components]['labels']
            next_labels = gmm_results[next_n_components]['labels']
            
            # For each point, track its transition
            n_points = len(current_labels)
            for cas_id in range(n_points):
                # Map CAS index to original dataset index if mapping is available
                if self.cas_to_original_mapping is not None and cas_id in self.cas_to_original_mapping:
                    original_id = self.cas_to_original_mapping[cas_id]
                else:
                    original_id = cas_id
                
                transition_data.append({
                    'real_id': int(original_id),
                    'layer': layer_name,
                    'from_components': current_n_components,
                    'to_components': next_n_components,
                    'from_cluster': int(current_labels[cas_id]),
                    'to_cluster': int(next_labels[cas_id])
                })
        
        # Save to CSV in the component transitions folder
        df_transitions = pd.DataFrame(transition_data)
        output_path = component_transitions_dir / 'real_id_component_transitions.csv'
        df_transitions.to_csv(output_path, index=False)
        print(f"\n✓ Component transition tracking saved to: {output_path}")
        print(f"  Total transitions: {len(df_transitions)}")
        print(f"  Unique real IDs: {df_transitions['real_id'].nunique()}")
        print(f"  Component transitions: {len(self.component_counts) - 1}")
        
        return df_transitions
    
    def run_all_tracking(self, n_components=None, layer_name=None):
        """Run both tracking analyses"""
        print("\n" + "="*60)
        print("Starting Real ID Tracking Analysis")
        print("="*60)
        
        # Generate layer transition tracking
        layer_df = self.generate_layer_transition_tracking(n_components=n_components)
        
        # Generate component transition tracking
        component_df = self.generate_component_transition_tracking(layer_name=layer_name)
        
        print("\n" + "="*60)
        print("Real ID Tracking Analysis Complete")
        print("="*60)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated folders and files:")
        print("  1. real_id_layer_transitions/")
        print("     - real_id_layer_transitions.csv - Tracks real IDs through layer transitions")
        print("  2. real_id_component_transitions/")
        print("     - real_id_component_transitions.csv - Tracks real IDs through component count changes")
        
        return layer_df, component_df


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Real ID Tracking CSV Files")
    parser.add_argument('--embeddings_dir', type=str, required=True,
                       help='Directory containing pre-saved embeddings')
    parser.add_argument('--output_dir', type=str, 
                       default='../../frontend/public/data',
                       help='Output directory for CSV files (default: frontend/public/data)')
    parser.add_argument('--n_components', type=int, default=None,
                       help='Number of components for layer transition analysis (default: auto-detect)')
    parser.add_argument('--layer_name', type=str, default=None,
                       help='Layer name for component transition analysis (default: first layer)')
    parser.add_argument('--zca_threshold', type=float, default=1e-5,
                       help='Threshold for ZCA whitening')
    
    args = parser.parse_args()
    
    # Resolve output directory path
    script_dir = Path(__file__).parent
    if args.output_dir.startswith('../') or args.output_dir.startswith('../../'):
        output_dir = (script_dir / args.output_dir).resolve()
    else:
        output_dir = Path(args.output_dir)
    
    print("Real ID Tracking CSV Generator")
    print("="*60)
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Output directory: {output_dir}")
    print(f"ZCA threshold: {args.zca_threshold}")
    if args.n_components:
        print(f"Number of components: {args.n_components}")
    if args.layer_name:
        print(f"Layer for component analysis: {args.layer_name}")
    print("="*60)
    
    # Create tracker and run analysis
    tracker = RealIDTracker(
        embeddings_dir=args.embeddings_dir,
        output_dir=output_dir,
        zca_threshold=args.zca_threshold
    )
    
    tracker.run_all_tracking(
        n_components=args.n_components,
        layer_name=args.layer_name
    )


if __name__ == "__main__":
    main()

