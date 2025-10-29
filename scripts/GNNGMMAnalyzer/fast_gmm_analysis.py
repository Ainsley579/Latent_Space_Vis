#!/usr/bin/env python3
"""
Fast GNN GMM Analysis Script

Optimized version for quick testing and analysis with:
- GPU acceleration (when available)
- Smaller sample sizes by default
- Efficient embedding extraction
- Quick GMM fitting and visualization

Usage:
    python fast_gmm_analysis.py --model_type dimenet-10conf --sample_frac 0.01
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Enable GPU if available (remove CPU-only restriction)
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    device = torch.device('cuda')
else:
    print("Using CPU")
    device = torch.device('cpu')

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from model_setup import dimenet10conf, dimenet_model

class FastGNNGMMAnalyzer:
    """Fast GMM analysis for GNN embeddings"""
    
    def __init__(self, model_type="dimenet-10conf", checkpoint_index=0, 
                 output_dir="../GNNGMMAnalyzerResults/fast_gmm_results", 
                 sample_frac=0.01, max_components=10):
        self.model_type = model_type
        self.checkpoint_index = checkpoint_index
        self.output_dir = Path(output_dir)
        self.sample_frac = sample_frac
        self.max_components = max_components
        self.model = None
        self.data = None
        self.layer_embeddings = {}
        self.gmm_results = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        
    def load_model_and_data(self):
        """Load the GNN model and data efficiently"""
        print(f"Loading {self.model_type} model (checkpoint {self.checkpoint_index})...")
        
        if self.model_type == "dimenet-10conf":
            self.model, self.data, _ = dimenet10conf()
        else:
            self.model, self.data, _ = dimenet_model(self.model_type, self.checkpoint_index)
        
        print(f"Model loaded. Original data size: {len(self.data)} samples")
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Convert DataLoader to list for sampling
        data_list = list(self.data)
        
        # Sample data for speed
        if self.sample_frac < 1.0:
            n_samples = int(len(data_list) * self.sample_frac)
            indices = np.random.choice(len(data_list), n_samples, replace=False)
            self.data = [data_list[i] for i in indices]
            print(f"Sampled {n_samples} samples for fast analysis")
        else:
            self.data = data_list
        
        print(f"Final dataset size: {len(self.data)} samples")
    
    def extract_embeddings_fast(self):
        """Fast embedding extraction using single forward pass"""
        print("Extracting embeddings from all layers (fast method)...")
        
        # Define layers to extract
        layers = [self.model.gnn.emb] + list(self.model.gnn.output_blocks)
        layer_names = ["emb"] + [f"output_block_{i}" for i in range(len(self.model.gnn.output_blocks))]
        
        # Initialize storage
        all_embeddings = {name: [] for name in layer_names}
        
        # Process batches efficiently
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data):
                if batch_idx % 100 == 0:
                    print(f"  Processing batch {batch_idx}/{len(self.data)}")
                
                # Move batch to device
                if hasattr(batch, 'x'):
                    x = batch.x.to(device)
                else:
                    x = batch.to(device)
                
                # Forward pass through all layers
                x_emb = self.model.gnn.emb(x)
                all_embeddings["emb"].append(x_emb.sum(dim=0).cpu())
                
                # Pass through output blocks
                x_current = x_emb
                for i, output_block in enumerate(self.model.gnn.output_blocks):
                    x_current = output_block(x_current)
                    all_embeddings[f"output_block_{i}"].append(x_current.sum(dim=0).cpu())
        
        # Convert to tensors
        for layer_name, embeddings_list in all_embeddings.items():
            embeddings_tensor = torch.stack(embeddings_list)
            self.layer_embeddings[layer_name] = embeddings_tensor.numpy()
            print(f"  {layer_name}: {embeddings_tensor.shape}")
        
        print(f"Extracted embeddings from {len(layers)} layers")
        return self.layer_embeddings
    
    def fit_gmms_fast(self, n_components=None):
        """Fast GMM fitting with reduced component testing"""
        if n_components is None:
            n_components = self.max_components
        
        print(f"Fitting GMMs with {n_components} components...")
        
        for layer_name, embeddings in self.layer_embeddings.items():
            print(f"  Fitting GMM for layer: {layer_name}")
            
            # Standardize embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # Fit GMM
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=42,
                max_iter=50  # Reduced iterations for speed
            )
            gmm.fit(embeddings_scaled)
            
            # Store results
            self.gmm_results[layer_name] = {
                'gmm': gmm,
                'scaler': scaler,
                'embeddings_scaled': embeddings_scaled,
                'labels': gmm.predict(embeddings_scaled),
                'probabilities': gmm.predict_proba(embeddings_scaled)
            }
        
        print(f"Fitted GMMs to {len(self.layer_embeddings)} layers")
    
    def create_parallel_coordinates_fast(self):
        """Create parallel coordinates plot efficiently"""
        print("Creating parallel coordinates plot...")
        
        # Use PCA for dimensionality reduction
        n_layers = len(self.layer_embeddings)
        
        # Create DataFrame with PCA components
        data_dict = {}
        for i, (layer_name, embeddings) in enumerate(self.layer_embeddings.items()):
            # Use PCA to reduce to 2 components for visualization
            pca = PCA(n_components=2, random_state=42)
            embeddings_pca = pca.fit_transform(embeddings)
            
            data_dict[f'{layer_name}_PC1'] = embeddings_pca[:, 0]
            data_dict[f'{layer_name}_PC2'] = embeddings_pca[:, 1]
        
        df = pd.DataFrame(data_dict)
        
        # Sample data for visualization
        if len(df) > 500:
            sample_indices = np.random.choice(len(df), 500, replace=False)
            df_sample = df.iloc[sample_indices]
        else:
            df_sample = df
        
        # Create parallel coordinates plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot parallel coordinates with transparency
        for i in range(len(df_sample)):
            ax.plot(range(len(df_sample.columns)), df_sample.iloc[i], 
                   alpha=0.3, linewidth=0.5, color='blue')
        
        ax.set_xticks(range(len(df_sample.columns)))
        ax.set_xticklabels(df_sample.columns, rotation=45, ha='right')
        ax.set_ylabel('PCA Component Value')
        ax.set_title('Fast GMM Analysis: Parallel Coordinates Plot')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fast_parallel_coordinates.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Parallel coordinates plot saved")
    
    def create_layer_evolution_fast(self):
        """Create fast layer evolution visualization"""
        print("Creating layer evolution visualization...")
        
        n_layers = len(self.layer_embeddings)
        n_cols = min(3, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_layers == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (layer_name, embeddings) in enumerate(self.layer_embeddings.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Use PCA for 2D projection
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # Get GMM labels
            labels = self.gmm_results[layer_name]['labels']
            
            # Plot with GMM cluster colors
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=labels, cmap='tab10', alpha=0.7, s=20)
            
            ax.set_title(f'Layer: {layer_name}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_layers, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fast_layer_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Layer evolution visualization saved")
    
    def run_fast_analysis(self):
        """Run the complete fast analysis pipeline"""
        print("Starting FAST GNN GMM analysis...")
        start_time = time.time()
        
        # Step 1: Load model and data
        self.load_model_and_data()
        
        # Step 2: Extract embeddings
        self.extract_embeddings_fast()
        
        # Step 3: Fit GMMs
        self.fit_gmms_fast()
        
        # Step 4: Create visualizations
        self.create_parallel_coordinates_fast()
        self.create_layer_evolution_fast()
        
        total_time = time.time() - start_time
        print(f"\nFast analysis completed in {total_time:.2f} seconds")
        print(f"Results saved in: {self.output_dir}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Model: {self.model_type}")
        print(f"- Sample size: {len(self.data)} molecules")
        print(f"- Layers analyzed: {len(self.layer_embeddings)}")
        print(f"- GMM components: {self.max_components}")


def main():
    parser = argparse.ArgumentParser(description="Fast GNN GMM Analysis")
    parser.add_argument('--model_type', type=str, default='dimenet-10conf',
                       choices=['dimenet-10conf', 'dimenet-100conf', 'dimenet-noaug'],
                       help='Type of DimeNet model to analyze')
    parser.add_argument('--checkpoint_index', type=int, default=0,
                       help='Index of checkpoint to load (0-based)')
    parser.add_argument('--output_dir', type=str, default='../GNNGMMAnalyzerResults/fast_gmm_results',
                       help='Output directory for results')
    parser.add_argument('--sample_frac', type=float, default=0.01,
                       help='Fraction of data to sample (default: 0.01 for speed)')
    parser.add_argument('--max_components', type=int, default=10,
                       help='Maximum number of GMM components (default: 10)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = FastGNNGMMAnalyzer(
        model_type=args.model_type,
        checkpoint_index=args.checkpoint_index,
        output_dir=args.output_dir,
        sample_frac=args.sample_frac,
        max_components=args.max_components
    )
    
    # Run analysis
    analyzer.run_fast_analysis()


if __name__ == "__main__":
    main()
