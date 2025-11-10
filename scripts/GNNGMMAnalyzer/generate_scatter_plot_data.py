#!/usr/bin/env python3
"""
Generate Scatter Plot Data CSV Files

This script generates CSV files for scatter plots based on the decorrelation-cas notebook.
It creates 2D MDS projections of embeddings for each layer, with real IDs mapped to the original dataset.

Output CSV files contain:
- real_id: Original dataset index (mapped from CAS indices)
- x, y: 2D MDS coordinates
- color_value: Prediction value for coloring
- layer: Layer name
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.manifold import MDS
import json

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


def get_euclidean_distances(X):
    """
    Compute pairwise Euclidean distances between all points in X.
    
    Args:
        X: Array of shape (n_samples, n_features)
    
    Returns:
        Distance matrix of shape (n_samples, n_samples)
    """
    # Compute squared differences
    X_squared = np.sum(X**2, axis=1, keepdims=True)
    distances_squared = X_squared + X_squared.T - 2 * X @ X.T
    # Ensure non-negative (handle numerical errors)
    distances_squared = np.maximum(distances_squared, 0)
    return np.sqrt(distances_squared)


def adaptive_zca_whitening(X, threshold=1e-5, epsilon=1e-5):
    """
    Apply adaptive ZCA whitening to X.
    
    Args:
        X: Matrix to whiten
        threshold: Minimum singular value to keep
        epsilon: Small value to avoid division by zero
    
    Returns:
        Whitened matrix
    """
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
    
    return X_whitened


class ScatterPlotDataGenerator:
    """Generate scatter plot data for frontend visualization"""
    
    def __init__(self, embeddings_dir, output_dir, zca_threshold=1e-5, random_state=42):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.zca_threshold = zca_threshold
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CAS to original index mapping
        self.cas_to_original_mapping = self._load_cas_mapping()
        
        # Layer names matching the notebook
        self.layer_files = {
            'emb': 'emb_unique_cas.pt',
            'output_block_0': 'output0_unique_cas.pt',
            'output_block_1': 'output1_unique_cas.pt',
            'output_block_2': 'output2_unique_cas.pt',
            'output_block_3': 'output3_unique_cas.pt',
            'output_block_4': 'output4_unique_cas.pt',
            'u_mlp_input': 'u_mlp_input_cas.pt',
            'u_mlp_l1': 'u_mlp_l1_cas.pt',
            'u_mlp_l2': 'u_mlp_l2_cas.pt',
        }
        
    def _load_cas_mapping(self):
        """Load CAS to original index mapping"""
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
                
                return mapping
        
        print("⚠ Warning: CAS to original mapping file not found")
        print("  Will use CAS indices directly (not mapped to original dataset)")
        return None
    
    def load_embeddings(self):
        """Load all layer embeddings"""
        print("\nLoading embeddings...")
        embeddings = {}
        
        for layer_name, filename in self.layer_files.items():
            filepath = self.embeddings_dir / filename
            
            if not filepath.exists():
                print(f"  Warning: {filename} not found, skipping {layer_name}")
                continue
            
            try:
                data = torch.load(filepath, map_location='cpu')
                if isinstance(data, torch.Tensor):
                    data = data.squeeze().numpy()
                else:
                    data = np.array(data)
                
                # Ensure 2D array
                if data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)
                
                embeddings[layer_name] = data
                print(f"  ✓ {layer_name}: shape {data.shape}")
                
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
                continue
        
        print(f"\nSuccessfully loaded {len(embeddings)} layers")
        return embeddings
    
    def load_predictions(self):
        """Load predictions for coloring"""
        preds_file = self.embeddings_dir / 'u_preds_cas.pt'
        
        if not preds_file.exists():
            print(f"⚠ Warning: {preds_file} not found, will use dummy predictions")
            return None
        
        try:
            preds = torch.load(preds_file, map_location='cpu')
            if isinstance(preds, torch.Tensor):
                preds = preds.squeeze().numpy()
            else:
                preds = np.array(preds)
            
            # Flatten if needed
            if preds.ndim > 1:
                preds = preds.flatten()
            
            print(f"✓ Loaded predictions: shape {preds.shape}")
            return preds
            
        except Exception as e:
            print(f"✗ Error loading predictions: {e}")
            return None
    
    def compute_mds_positions(self, embeddings, n_components=2):
        """
        Compute MDS positions for all layers.
        Uses consistent random_state for reproducibility.
        """
        print(f"\nComputing MDS positions (n_components={n_components})...")
        positions = {}
        
        reducer = MDS(
            n_components=n_components,
            dissimilarity='precomputed',
            random_state=self.random_state,
            n_init=1  # Use single initialization for consistency
        )
        
        for layer_name, embedding in embeddings.items():
            print(f"  Computing MDS for {layer_name}...")
            
            # Compute pairwise distances
            distances = get_euclidean_distances(embedding)
            
            # Fit MDS
            pos = reducer.fit_transform(distances)
            positions[layer_name] = pos
            
            print(f"    ✓ {layer_name}: MDS shape {pos.shape}")
        
        return positions
    
    def generate_scatter_plot_csvs(self, embeddings, positions, predictions=None):
        """
        Generate CSV files for scatter plots.
        One CSV per layer with real_id, x, y, color_value, layer.
        """
        print("\nGenerating scatter plot CSV files...")
        
        all_data = []
        
        for layer_name in embeddings.keys():
            if layer_name not in positions:
                continue
            
            pos = positions[layer_name]
            n_points = pos.shape[0]
            
            # Get color values (predictions or dummy)
            if predictions is not None and len(predictions) >= n_points:
                color_values = predictions[:n_points]
            else:
                color_values = np.zeros(n_points)
            
            # Create data for this layer
            for cas_id in range(n_points):
                # Map CAS index to original dataset index
                if self.cas_to_original_mapping is not None and cas_id in self.cas_to_original_mapping:
                    original_id = self.cas_to_original_mapping[cas_id]
                else:
                    original_id = cas_id
                
                all_data.append({
                    'real_id': int(original_id),
                    'x': float(pos[cas_id, 0]),
                    'y': float(pos[cas_id, 1]),
                    'color_value': float(color_values[cas_id]),
                    'layer': layer_name
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Save combined CSV
        combined_path = self.output_dir / 'scatter_plot_data_all_layers.csv'
        df.to_csv(combined_path, index=False)
        print(f"✓ Saved combined CSV: {combined_path}")
        print(f"  Total points: {len(df)}")
        print(f"  Layers: {df['layer'].nunique()}")
        print(f"  Real ID range: {df['real_id'].min()} - {df['real_id'].max()}")
        
        # Save individual CSV files per layer
        layer_dir = self.output_dir / 'scatter_plot_data_by_layer'
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        for layer_name in df['layer'].unique():
            layer_df = df[df['layer'] == layer_name].copy()
            layer_path = layer_dir / f'scatter_plot_data_{layer_name}.csv'
            layer_df.to_csv(layer_path, index=False)
            print(f"  ✓ {layer_name}: {len(layer_df)} points -> {layer_path.name}")
        
        return df
    
    def generate_metadata(self, positions):
        """Generate metadata file with min/max x/y for consistent plotting"""
        print("\nGenerating metadata...")
        
        # Compute global min/max across all layers
        all_x = []
        all_y = []
        
        for pos in positions.values():
            all_x.extend(pos[:, 0].tolist())
            all_y.extend(pos[:, 1].tolist())
        
        metadata = {
            'min_x': float(np.min(all_x)),
            'max_x': float(np.max(all_x)),
            'min_y': float(np.min(all_y)),
            'max_y': float(np.max(all_y)),
            'layers': list(positions.keys()),
            'n_points_per_layer': {layer: int(pos.shape[0]) for layer, pos in positions.items()}
        }
        
        # Save metadata as JSON
        metadata_path = self.output_dir / 'scatter_plot_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata: {metadata_path}")
        print(f"  X range: [{metadata['min_x']:.4f}, {metadata['max_x']:.4f}]")
        print(f"  Y range: [{metadata['min_y']:.4f}, {metadata['max_y']:.4f}]")
        
        return metadata
    
    def run(self):
        """Run the complete data generation pipeline"""
        print("="*60)
        print("Scatter Plot Data Generator")
        print("="*60)
        print(f"Embeddings directory: {self.embeddings_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"ZCA threshold: {self.zca_threshold}")
        print("="*60)
        
        # Step 1: Load embeddings
        embeddings = self.load_embeddings()
        if not embeddings:
            print("❌ No embeddings loaded. Exiting.")
            return
        
        # Step 2: Load predictions
        predictions = self.load_predictions()
        
        # Step 3: Compute MDS positions
        positions = self.compute_mds_positions(embeddings, n_components=2)
        
        # Step 4: Generate CSV files
        df = self.generate_scatter_plot_csvs(embeddings, positions, predictions)
        
        # Step 5: Generate metadata
        metadata = self.generate_metadata(positions)
        
        print("\n" + "="*60)
        print("✅ Scatter Plot Data Generation Complete")
        print("="*60)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        print("  1. scatter_plot_data_all_layers.csv - Combined data for all layers")
        print("  2. scatter_plot_data_by_layer/ - Individual CSV files per layer")
        print("  3. scatter_plot_metadata.json - Metadata for consistent plotting")
        print("\nEach CSV contains:")
        print("  - real_id: Original dataset index")
        print("  - x, y: 2D MDS coordinates")
        print("  - color_value: Prediction value for coloring")
        print("  - layer: Layer name")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Scatter Plot Data CSV Files")
    parser.add_argument('--embeddings_dir', type=str, required=True,
                       help='Directory containing pre-saved embeddings')
    parser.add_argument('--output_dir', type=str, 
                       default='../../frontend/public/data',
                       help='Output directory for CSV files (default: frontend/public/data)')
    parser.add_argument('--zca_threshold', type=float, default=1e-5,
                       help='Threshold for ZCA whitening (not used in MDS, but kept for compatibility)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for MDS (default: 42)')
    
    args = parser.parse_args()
    
    # Resolve output directory path
    script_dir = Path(__file__).parent
    if args.output_dir.startswith('../') or args.output_dir.startswith('../../'):
        output_dir = (script_dir / args.output_dir).resolve()
    else:
        output_dir = Path(args.output_dir)
    
    # Create generator and run
    generator = ScatterPlotDataGenerator(
        embeddings_dir=args.embeddings_dir,
        output_dir=output_dir,
        zca_threshold=args.zca_threshold,
        random_state=args.random_state
    )
    
    generator.run()


if __name__ == "__main__":
    main()

