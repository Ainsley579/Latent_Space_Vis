#!/usr/bin/env python3
"""
GNN Layer-wise GMM Analysis Script with Layer-Specific Components

This script performs comprehensive GMM analysis on GNN embeddings across different layers
with INDIVIDUAL optimal component selection for each layer (not shared across layers):

1. Layer-specific GMM fitting with individual BIC/AIC for optimal components per layer
2. Layer-wise trajectory analysis with different component counts
3. Visualization of evolution across layers with varying component counts
4. Quantification of cluster sharpening per layer
5. Boundary analysis for ambiguous molecules
6. Fuzzy clustering and parallel coordinates plots
7. Sankey diagrams showing transitions between layers with different component counts

Usage:
    python gnn_gmm_analysis_layer_specific.py --model_type dimenet-10conf --checkpoint_index 0
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
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib.patches import Ellipse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import plotly for Sankey diagrams
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available, Sankey diagrams will be skipped")
    PLOTLY_AVAILABLE = False


import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Go up to latent-spaces directory
sys.path.append(parent_dir)

try:
    from latentlib import save_embedding
    LATENTLIB_AVAILABLE = True
except ImportError:
    print("Warning: latentlib not available, using alternative embedding extraction")
    LATENTLIB_AVAILABLE = False

from model_setup import dimenet10conf, dimenet_model

class GNNGMMAnalyzerLayerSpecific:
    """Comprehensive GMM analysis for GNN embeddings across layers with layer-specific components"""
    
    def __init__(self, model_type="dimenet-10conf", checkpoint_index=0, 
                 output_dir="../GNNGMMAnalyzerResults/gnn_gmm_layer_specific_results", sample_frac=1.0,
                 embeddings_dir=None):
        self.model_type = model_type
        self.checkpoint_index = checkpoint_index
        self.output_dir = Path(output_dir)
        self.sample_frac = sample_frac
        self.embeddings_dir = embeddings_dir
        self.model = None
        self.data = None
        self.smiles = None
        self.layer_embeddings = {}
        self.gmm_results = {}
        self.layer_optimal_components = {}  # Store optimal components for each layer
        self.component_persistence = {}  # Track component persistence across layers
        self.point_transitions = {}      # Track individual point transitions
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def adaptive_zca_whitening(self, X, threshold=1e-5, epsilon=1e-5):
        """Applies an adaptive ZCA whitening to X.
        Args:
            X: Matrix to reduce
            threshold: the minimum value a singular value (column) should have to be kept.
            epsilon: Noise to add to singular values to avoid division by zero.

        Returns:
            The whitened matrix and the whitening transformation matrix.
        """
        # Calculate covariance matrix
        sigma = np.cov(X, rowvar=False)
        U, S, V = np.linalg.svd(sigma)

        # Find number of features to keep
        k = 0
        for v in S:
            if v > threshold:
                k += 1
        print(f"    ZCA Whitening: {k} features kept out of {len(S)} (threshold={threshold})")

        # Keep only the top k components
        top_U = U[:, :k]
        top_S = S[:k]

        # Create whitening transformation matrix
        red_zca = np.dot(np.diag(1.0 / np.sqrt(top_S + epsilon)), top_U.T)
        
        # Apply whitening transformation
        X_whitened = X @ red_zca.T
        
        return X_whitened, red_zca
        
    def load_pre_saved_embeddings(self, embeddings_dir=None):
        """Load pre-saved embeddings from directory"""
        if embeddings_dir is None:
            embeddings_dir = self.embeddings_dir
        
        if embeddings_dir is None:
            raise ValueError("No embeddings directory specified")
        
        embeddings_path = Path(embeddings_dir)
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_path}")
        
        print(f"Loading pre-saved embeddings from: {embeddings_path}")
        
        # Define expected layer files and their names
        # Use unique_cas files if available, then unique files, then original files
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
        
        # Load embeddings for each layer (9 layers total)
        for layer_name in ['emb', 'output_block_0', 'output_block_1', 'output_block_2', 'output_block_3', 'output_block_4', 
                           'u_mlp_input', 'u_mlp_l1', 'u_mlp_l2']:
            # Try unique_cas file first, then unique file, then fallback to original
            unique_cas_filename = layer_mappings[layer_name]['unique_cas']
            unique_filename = layer_mappings[layer_name]['unique']
            original_filename = layer_mappings[layer_name]['original']
            
            # Check for unique_cas file first
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
                # Load tensor and convert to numpy
                embeddings_tensor = torch.load(filepath, map_location='cpu')
                embeddings_np = embeddings_tensor.detach().cpu().numpy()
            
                # Reshape if needed (flatten extra dimensions)
                if embeddings_np.ndim > 2:
                    original_shape = embeddings_np.shape
                    # Flatten all dimensions except the first (batch dimension)
                    embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)
                    print(f"    Shape: {original_shape} -> {embeddings_np.shape} (reshaped)")
                else:
                    print(f"    Shape: {embeddings_np.shape}")
                
                self.layer_embeddings[layer_name] = embeddings_np
            except Exception as e:
                    print(f"    Error loading {filename}: {e}")
        
        if not self.layer_embeddings:
            raise ValueError("No embeddings were successfully loaded")
        
        print(f"Successfully loaded embeddings from {len(self.layer_embeddings)} layers")
        return self.layer_embeddings
    
    def find_optimal_components_for_layer(self, embeddings, layer_name, max_components=20, zca_threshold=1e-5):
        """Find optimal number of components for a specific layer using BIC and AIC with ZCA whitening"""
        print(f"Finding optimal components for layer: {layer_name}")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply ZCA whitening
        print(f"  Applying ZCA whitening for component selection...")
        embeddings_whitened, _ = self.adaptive_zca_whitening(embeddings_scaled, threshold=zca_threshold)
        
        # Test different numbers of components
        n_components_range = range(2, max_components + 1)
        bic_scores = []
        aic_scores = []
        silhouette_scores = []
        
        for n_components in n_components_range:
            print(f"  Testing {n_components} components...")
            
            try:
                # Fit GMM on whitened embeddings
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42,
                    max_iter=100
                )
                gmm.fit(embeddings_whitened)
                
                # Calculate scores
                bic_scores.append(gmm.bic(embeddings_whitened))
                aic_scores.append(gmm.aic(embeddings_whitened))
                
                # Calculate silhouette score
                labels = gmm.predict(embeddings_whitened)
                sil_score = silhouette_score(embeddings_whitened, labels)
                silhouette_scores.append(sil_score)
                
            except Exception as e:
                print(f"    Warning: Failed to fit {n_components} components: {e}")
                bic_scores.append(float('inf'))
                aic_scores.append(float('inf'))
                silhouette_scores.append(0)
        
        # Find optimal components
        optimal_bic = n_components_range[np.argmin(bic_scores)]
        optimal_aic = n_components_range[np.argmin(aic_scores)]
        optimal_sil = n_components_range[np.argmax(silhouette_scores)]
        
        print(f"  Optimal components for {layer_name} - BIC: {optimal_bic}, AIC: {optimal_aic}, Silhouette: {optimal_sil}")
        
        # Use AIC as primary criterion (often more conservative)
        optimal_components = optimal_aic
        
        # Store the optimal components for this layer
        self.layer_optimal_components[layer_name] = optimal_components
        
        return optimal_components, n_components_range, bic_scores, aic_scores, silhouette_scores
    
    def fit_layer_gmms_with_specific_components(self, zca_threshold=1e-5):
        """Fit GMM to embeddings from each layer with their own optimal component counts and ZCA whitening"""
        print(f"Fitting GMMs to all layers with layer-specific optimal components...")
        
        # First, find optimal components for each layer
        for layer_name, embeddings in self.layer_embeddings.items():
            print(f"\n=== Processing layer: {layer_name} ===")
            optimal_components = self.find_optimal_components_for_layer(
                embeddings, layer_name, max_components=30, zca_threshold=zca_threshold
            )[0]  # Only get the optimal components count
            print(f"Layer {layer_name} will use {optimal_components} components")
        
        # Now fit GMMs with layer-specific components
        for layer_name, embeddings in self.layer_embeddings.items():
            optimal_components = self.layer_optimal_components[layer_name]
            print(f"\nFitting GMM for layer: {layer_name} (n_components={optimal_components})")
            
            # Standardize embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # Apply ZCA whitening
            print(f"    Applying ZCA whitening...")
            embeddings_whitened, zca_transform = self.adaptive_zca_whitening(
                embeddings_scaled, threshold=zca_threshold
            )
            
            # Fit GMM on whitened embeddings with layer-specific components
            gmm = GaussianMixture(
                n_components=optimal_components,
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
            gmm.fit(embeddings_whitened)
            
            # Store results
            self.gmm_results[layer_name] = {
                'gmm': gmm,
                'scaler': scaler,
                'zca_transform': zca_transform,
                'embeddings_scaled': embeddings_scaled,
                'embeddings_whitened': embeddings_whitened,
                'embeddings_original': embeddings,
                'means': gmm.means_,
                'covariances': gmm.covariances_,
                'weights': gmm.weights_,
                'labels': gmm.predict(embeddings_whitened),
                'probabilities': gmm.predict_proba(embeddings_whitened),
                'n_components': optimal_components
            }
        
        print(f"\nFitted GMMs to {len(self.layer_embeddings)} layers with layer-specific component counts")
        print("Component counts per layer:")
        for layer_name, n_comp in self.layer_optimal_components.items():
            print(f"  {layer_name}: {n_comp} components")
    
    def create_component_count_evolution_plot(self):
        """Create visualization showing how component count changes across layers"""
        print("Creating component count evolution plot...")
        
        layer_names = list(self.layer_optimal_components.keys())
        component_counts = list(self.layer_optimal_components.values())
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top plot: Component count evolution
        x_pos = np.arange(len(layer_names))
        bars = ax1.bar(x_pos, component_counts, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1.5)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(layer_names, rotation=45, ha='right')
        ax1.set_ylabel('Number of Components')
        ax1.set_title('Evolution of Optimal Component Count Across Layers (Layer-Specific)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, component_counts)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Bottom plot: Component count changes between consecutive layers
        if len(component_counts) > 1:
            changes = []
            change_labels = []
            
            for i in range(len(component_counts) - 1):
                change = component_counts[i+1] - component_counts[i]
                changes.append(change)
                change_labels.append(f'{layer_names[i]} → {layer_names[i+1]}')
            
            colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in changes]
            bars2 = ax2.bar(range(len(changes)), changes, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            ax2.set_xticks(range(len(changes)))
            ax2.set_xticklabels(change_labels, rotation=45, ha='right')
            ax2.set_ylabel('Component Count Change')
            ax2.set_title('Component Count Changes Between Consecutive Layers', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, change) in enumerate(zip(bars2, changes)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + (0.1 if height > 0 else -0.3),
                        f'{change:+d}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_count_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Component count evolution plot saved")
    
    def visualize_layer_evolution(self):
        """Create 2D visualizations showing evolution across layers"""
        print("Creating layer evolution visualizations...")
        
        # Create subplots for each layer
        n_layers = len(self.layer_embeddings)
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_layers == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Get molecule classes if available
        molecule_classes = self._get_molecule_classes()
        
        for i, (layer_name, embeddings) in enumerate(self.layer_embeddings.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Fit PCA for this specific layer
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # Get GMM results
            gmm_result = self.gmm_results[layer_name]
            # Transform means from whitened space back to original space, then to 2D
            # First, transform means back from whitened space to scaled space
            means_scaled = gmm_result['means'] @ gmm_result['zca_transform']
            # Then inverse transform from scaled space to original space
            means_original = gmm_result['scaler'].inverse_transform(means_scaled)
            # Finally, transform to 2D using PCA
            means_2d = pca.transform(means_original)
            
            # Plot points
            if molecule_classes is not None:
                scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=molecule_classes, cmap='tab10', alpha=0.6, s=20)
            else:
                scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   alpha=0.6, s=20)
            
            # Plot GMM means (skip ellipses for now due to dimension issues with ZCA whitening)
            # self._plot_gmm_ellipses(ax, means_2d, gmm_result['covariances'], 
            #                       gmm_result['scaler'], pca, gmm_result['zca_transform'])
            
            # Plot GMM means as points
            ax.scatter(means_2d[:, 0], means_2d[:, 1], 
                      c='red', marker='x', s=100, linewidth=3, label=f'GMM Centers (n={gmm_result["n_components"]})')
            
            ax.set_title(f'Layer: {layer_name}\n({gmm_result["n_components"]} components)')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Remove empty subplots
        for i in range(n_layers, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Layer evolution visualization saved")
    
    def _get_molecule_classes(self):
        """Extract molecule classes from SMILES data if available"""
        if self.smiles is None:
            return None
        
        try:
            # Extract CAS numbers or other identifiers
            if isinstance(self.smiles, tuple):
                smiles_df = self.smiles[0]  # Use train smiles
            else:
                smiles_df = self.smiles
            
            if 'ligand_1_cas' in smiles_df.columns:
                # Use CAS numbers as classes
                classes = smiles_df['ligand_1_cas'].values
                # Convert to numeric labels
                unique_classes = np.unique(classes)
                class_labels = np.array([np.where(unique_classes == c)[0][0] for c in classes])
                return class_labels
            else:
                return None
        except Exception as e:
            print(f"Warning: Could not extract molecule classes: {e}")
            return None
    
    def quantify_cluster_sharpening(self):
        """Plot average determinant/trace of covariance matrices across layers"""
        print("Quantifying cluster sharpening...")
        
        layers = list(self.layer_embeddings.keys())
        determinants = []
        traces = []
        
        # Store individual component data for detailed analysis
        # Use log-determinants for numerical stability in per-component plots
        individual_determinants = {}  # {layer_name: [logdet1, logdet2, ...]}
        individual_traces = {}       # {layer_name: [trace1, trace2, ...]}
        
        for layer_name in layers:
            gmm_result = self.gmm_results[layer_name]
            covariances = gmm_result['covariances']
            
            # Calculate individual component metrics
            # Use slogdet to avoid underflow; for positive-definite covariances, sign should be +1
            layer_determinants = []
            for cov in covariances:
                sign, logabsdet = np.linalg.slogdet(cov)
                # If sign <= 0 due to numerical issues, fall back to very small value
                if sign <= 0:
                    logabsdet = -1e12
                layer_determinants.append(logabsdet)
            layer_traces = [np.trace(cov) for cov in covariances]
            
            individual_determinants[layer_name] = layer_determinants
            individual_traces[layer_name] = layer_traces
            
            # Calculate average determinant and trace
            avg_det = float(np.exp(np.mean(layer_determinants)))
            avg_trace = np.mean(layer_traces)
            
            determinants.append(avg_det)
            traces.append(avg_trace)
        
        # Plot average results (original plot)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Determinant plot
        axes[0].plot(range(len(layers)), determinants, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Layer Index')
        axes[0].set_ylabel('Average Determinant')
        axes[0].set_title('Cluster Sharpening: Average Determinant')
        axes[0].set_yscale('log')
        axes[0].set_xticks(range(len(layers)))
        axes[0].set_xticklabels(layers, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Trace plot
        axes[1].plot(range(len(layers)), traces, 'o-', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Layer Index')
        axes[1].set_ylabel('Average Trace')
        axes[1].set_title('Cluster Sharpening: Average Trace')
        axes[1].set_xticks(range(len(layers)))
        axes[1].set_xticklabels(layers, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_sharpening.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual component plots
        self._plot_individual_component_sharpening(individual_determinants, individual_traces, layers)
        
        print("Cluster sharpening analysis saved")
    
    def _plot_individual_component_sharpening(self, individual_determinants, individual_traces, layers):
        """Create plots for individual component sharpening across layers"""
        print("Creating individual component sharpening plots...")
        
        # Plot 1: Individual component log-determinants across layers
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get maximum number of components across all layers
        max_components = max(self.layer_optimal_components.values())
        
        # Create data matrix: rows = components, columns = layers
        det_matrix = np.full((max_components, len(layers)), np.nan)
        for i, layer_name in enumerate(layers):
            n_comp = self.layer_optimal_components[layer_name]
            det_matrix[:n_comp, i] = individual_determinants[layer_name]
        
        # Plot each component as a separate line
        colors = plt.cm.tab10(np.linspace(0, 1, max_components))
        for comp_idx in range(max_components):
            # Only plot if this component exists in at least one layer
            valid_data = det_matrix[comp_idx, :]
            if not np.all(np.isnan(valid_data)):
                ax.plot(range(len(layers)), valid_data, 
                       'o-', linewidth=2, markersize=6, color=colors[comp_idx],
                       label=f'Component {comp_idx}')
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('log|Determinant|')
        ax.set_title('Individual Component log-Determinants Across Layers (Layer-Specific Components)')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_sharpening_individual_determinants.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Individual component traces across layers
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create data matrix: rows = components, columns = layers
        trace_matrix = np.full((max_components, len(layers)), np.nan)
        for i, layer_name in enumerate(layers):
            n_comp = self.layer_optimal_components[layer_name]
            trace_matrix[:n_comp, i] = individual_traces[layer_name]
        
        # Plot each component as a separate line
        for comp_idx in range(max_components):
            # Only plot if this component exists in at least one layer
            valid_data = trace_matrix[comp_idx, :]
            if not np.all(np.isnan(valid_data)):
                ax.plot(range(len(layers)), valid_data, 
                       'o-', linewidth=2, markersize=6, color=colors[comp_idx],
                       label=f'Component {comp_idx}')
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Trace')
        ax.set_title('Individual Component Traces Across Layers (Layer-Specific Components)')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_sharpening_individual_traces.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Individual component sharpening plots saved")
    
    def analyze_boundary_molecules(self, n_molecules=10):
        """Analyze molecules with highest assignment entropy"""
        print(f"Analyzing {n_molecules} most ambiguous molecules...")
        
        # Get final layer GMM
        final_layer = list(self.gmm_results.keys())[-1]
        final_gmm = self.gmm_results[final_layer]
        probabilities = final_gmm['probabilities']
        
        # Calculate entropy for each molecule
        entropies = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        
        # Find molecules with highest entropy
        top_indices = np.argsort(entropies)[-n_molecules:]
        
        # Create analysis
        boundary_analysis = {
            'indices': top_indices,
            'entropies': entropies[top_indices],
            'probabilities': probabilities[top_indices],
            'smiles': []
        }
        
        # Extract SMILES if available
        if self.smiles is not None:
            try:
                if isinstance(self.smiles, tuple):
                    smiles_df = self.smiles[0]
                else:
                    smiles_df = self.smiles
                
                if 'ligand_1_smiles' in smiles_df.columns:
                    boundary_analysis['smiles'] = smiles_df['ligand_1_smiles'].iloc[top_indices].values
            except Exception as e:
                print(f"Warning: Could not extract SMILES for boundary molecules: {e}")
        
        # Save analysis
        np.save(self.output_dir / 'boundary_analysis.npy', boundary_analysis)
        
        # Create visualization
        self._plot_boundary_analysis(boundary_analysis)
        
        print("Boundary analysis completed")
        return boundary_analysis
    
    def _plot_boundary_analysis(self, boundary_analysis):
        """Plot boundary molecule analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Entropy distribution
        axes[0, 0].hist(boundary_analysis['entropies'], bins=10, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Assignment Entropy')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Entropy Distribution of Boundary Molecules')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Probability heatmap
        im = axes[0, 1].imshow(boundary_analysis['probabilities'], cmap='viridis', aspect='auto')
        axes[0, 1].set_xlabel('Component')
        axes[0, 1].set_ylabel('Molecule Index')
        axes[0, 1].set_title('Assignment Probabilities')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Entropy vs index
        axes[1, 0].scatter(range(len(boundary_analysis['entropies'])), boundary_analysis['entropies'])
        axes[1, 0].set_xlabel('Molecule Index')
        axes[1, 0].set_ylabel('Assignment Entropy')
        axes[1, 0].set_title('Entropy by Molecule Index')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Component weights
        final_layer = list(self.gmm_results.keys())[-1]
        weights = self.gmm_results[final_layer]['weights']
        axes[1, 1].bar(range(len(weights)), weights)
        axes[1, 1].set_xlabel('Component')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].set_title('GMM Component Weights')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'boundary_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_parallel_coordinates_plot(self):
        """Create parallel coordinates plot for layer-wise trajectory"""
        print("Creating parallel coordinates plot...")
        
        # Prepare data for parallel coordinates
        n_layers = len(self.layer_embeddings)
        n_samples = len(list(self.layer_embeddings.values())[0])
        
        # Create DataFrame with layer embeddings
        data_dict = {}
        for i, (layer_name, embeddings) in enumerate(self.layer_embeddings.items()):
            # Use first few principal components for visualization
            pca = PCA(n_components=min(3, embeddings.shape[1]))
            embeddings_pca = pca.fit_transform(embeddings)
            
            for j in range(embeddings_pca.shape[1]):
                data_dict[f'{layer_name}_PC{j+1}'] = embeddings_pca[:, j]
        
        df = pd.DataFrame(data_dict)
        
        # Sample data for visualization (too many points can be overwhelming)
        if len(df) > 1000:
            sample_indices = np.random.choice(len(df), 1000, replace=False)
            df_sample = df.iloc[sample_indices]
        else:
            df_sample = df
        
        # Create original parallel coordinates plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot parallel coordinates
        for i in range(len(df_sample)):
            ax.plot(range(len(df_sample.columns)), df_sample.iloc[i], 
                   alpha=0.1, linewidth=0.5, color='blue')
        
        ax.set_xticks(range(len(df_sample.columns)))
        ax.set_xticklabels(df_sample.columns, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title('Layer-wise Trajectory: Parallel Coordinates Plot (Layer-Specific Components)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parallel_coordinates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate plots for PC1, PC2, PC3 across layers
        self._create_separate_pc_plots(df_sample)
        
        print("Parallel coordinates plot saved")
    
    def _create_separate_pc_plots(self, df_sample):
        """Create separate plots for PC1, PC2, PC3 across all layers"""
        print("Creating separate PC plots...")
        
        # Get layer names
        layer_names = []
        for col in df_sample.columns:
            layer_name = col.split('_PC')[0]
            if layer_name not in layer_names:
                layer_names.append(layer_name)
        
        # Create plots for each principal component
        for pc_num in [1, 2, 3]:
            pc_cols = [f'{layer}_PC{pc_num}' for layer in layer_names if f'{layer}_PC{pc_num}' in df_sample.columns]
            
            if not pc_cols:
                continue
                
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot trajectories for each molecule
            for i in range(len(df_sample)):
                values = df_sample[pc_cols].iloc[i].values
                ax.plot(range(len(values)), values, alpha=0.1, linewidth=0.5, color='blue')
            
            ax.set_xticks(range(len(pc_cols)))
            ax.set_xticklabels([col.replace('_PC' + str(pc_num), '') for col in pc_cols], 
                              rotation=45, ha='right')
            ax.set_ylabel(f'PC{pc_num} Value')
            ax.set_title(f'Principal Component {pc_num} Evolution Across GNN Layers (Layer-Specific Components)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'PC{pc_num}_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  PC{pc_num} evolution plot saved")
        
        print("Separate PC plots completed")
    
    def create_fuzzy_clustering_analysis(self):
        """Create fuzzy clustering analysis and visualizations"""
        print("Creating fuzzy clustering analysis...")
        
        # Analyze fuzzy clustering for each layer
        fuzzy_results = {}
        
        for layer_name, gmm_result in self.gmm_results.items():
            probabilities = gmm_result['probabilities']
            
            # Calculate fuzzy clustering metrics
            max_prob = np.max(probabilities, axis=1)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
            
            fuzzy_results[layer_name] = {
                'probabilities': probabilities,
                'max_probability': max_prob,
                'entropy': entropy,
                'confidence': max_prob,  # Confidence = max probability
                'ambiguity': entropy     # Ambiguity = entropy
            }
        
        # Create visualizations
        self._plot_fuzzy_clustering(fuzzy_results)
        
        # Save results
        np.save(self.output_dir / 'fuzzy_clustering_results.npy', fuzzy_results)
        
        print("Fuzzy clustering analysis completed")
        return fuzzy_results
    
    def _plot_fuzzy_clustering(self, fuzzy_results):
        """Plot fuzzy clustering analysis"""
        n_layers = len(fuzzy_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        layers = list(fuzzy_results.keys())
        
        # Confidence across layers
        confidences = [fuzzy_results[layer]['confidence'] for layer in layers]
        axes[0, 0].boxplot(confidences, labels=layers)
        axes[0, 0].set_ylabel('Confidence (Max Probability)')
        axes[0, 0].set_title('Confidence Distribution Across Layers (Layer-Specific Components)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ambiguity across layers
        ambiguities = [fuzzy_results[layer]['ambiguity'] for layer in layers]
        axes[0, 1].boxplot(ambiguities, labels=layers)
        axes[0, 1].set_ylabel('Ambiguity (Entropy)')
        axes[0, 1].set_title('Ambiguity Distribution Across Layers (Layer-Specific Components)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average confidence over layers
        avg_confidence = [np.mean(fuzzy_results[layer]['confidence']) for layer in layers]
        axes[1, 0].plot(range(len(layers)), avg_confidence, 'o-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Average Confidence')
        axes[1, 0].set_title('Average Confidence Across Layers')
        axes[1, 0].set_xticks(range(len(layers)))
        axes[1, 0].set_xticklabels(layers, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Average ambiguity over layers
        avg_ambiguity = [np.mean(fuzzy_results[layer]['ambiguity']) for layer in layers]
        axes[1, 1].plot(range(len(layers)), avg_ambiguity, 'o-', linewidth=2, markersize=8, color='orange')
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('Average Ambiguity')
        axes[1, 1].set_title('Average Ambiguity Across Layers')
        axes[1, 1].set_xticks(range(len(layers)))
        axes[1, 1].set_xticklabels(layers, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fuzzy_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_sankey_diagram(self):
        """Create Sankey diagram showing component transitions across layers with different component counts"""
        if not PLOTLY_AVAILABLE:
            print("Skipping Sankey diagram: plotly not available")
            return
        
        print("Creating Sankey diagram for component transitions with layer-specific components...")
        
        layer_names = list(self.gmm_results.keys())
        
        # Prepare data for Sankey diagram
        source_nodes = []
        target_nodes = []
        values = []
        labels = []
        
        # Create node labels for each layer with their component counts
        node_id = 0
        layer_node_mapping = {}
        
        for layer_idx, layer_name in enumerate(layer_names):
            n_components = self.layer_optimal_components[layer_name]
            layer_node_mapping[layer_name] = {}
            for comp_idx in range(n_components):
                node_label = f"{layer_name}\nComponent {comp_idx}\n({n_components} total)"
                labels.append(node_label)
                layer_node_mapping[layer_name][comp_idx] = node_id
                node_id += 1
        
        # Calculate transitions between consecutive layers
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i+1]
            
            # Get cluster assignments
            current_labels = self.gmm_results[current_layer]['labels']
            next_labels = self.gmm_results[next_layer]['labels']
            
            current_n_components = self.layer_optimal_components[current_layer]
            next_n_components = self.layer_optimal_components[next_layer]
            
            # Calculate transition counts
            for src_comp in range(current_n_components):
                src_mask = current_labels == src_comp
                if np.sum(src_mask) > 0:
                    next_clusters = next_labels[src_mask]
                    for dst_comp in range(next_n_components):
                        count = np.sum(next_clusters == dst_comp)
                        if count > 0:  # Only include non-zero transitions
                            source_nodes.append(layer_node_mapping[current_layer][src_comp])
                            target_nodes.append(layer_node_mapping[next_layer][dst_comp])
                            values.append(count)
        
        # Create color palette for different layers
        colors = px.colors.qualitative.Set3
        node_colors = []
        for layer_idx, layer_name in enumerate(layer_names):
            layer_color = colors[layer_idx % len(colors)]
            n_components = self.layer_optimal_components[layer_name]
            for comp_idx in range(n_components):
                node_colors.append(layer_color)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=source_nodes,
                target=target_nodes,
                value=values,
                color='rgba(0,0,0,0.2)'  # Semi-transparent links
            )
        )])
        
        fig.update_layout(
            title_text=f"Component Transitions Across GNN Layers (Layer-Specific Components)<br><sub>Each flow represents data points moving between components with different counts per layer</sub>",
            font_size=10,
            width=1200,
            height=600
        )
        
        # Save as HTML
        sankey_path = self.output_dir / 'component_transitions_sankey.html'
        fig.write_html(str(sankey_path))
        
        # Also save as static image if possible
        try:
            sankey_png_path = self.output_dir / 'component_transitions_sankey.png'
            fig.write_image(str(sankey_png_path), width=1200, height=600, scale=2)
            print(f"Sankey diagram saved as PNG: {sankey_png_path}")
        except Exception as e:
            print(f"Could not save PNG version: {e}")
        
        print(f"Sankey diagram saved as HTML: {sankey_path}")
        
        # Create a simplified version focusing on major transitions
        self._create_simplified_sankey_diagram(layer_names, layer_node_mapping)
    
    def _create_simplified_sankey_diagram(self, layer_names, layer_node_mapping):
        """Create a simplified Sankey diagram showing only major transitions (>10% of points)"""
        print("Creating simplified Sankey diagram (major transitions only)...")
        
        source_nodes = []
        target_nodes = []
        values = []
        labels = []
        
        # Create node labels
        node_id = 0
        for layer_idx, layer_name in enumerate(layer_names):
            n_components = self.layer_optimal_components[layer_name]
            for comp_idx in range(n_components):
                node_label = f"{layer_name}\nC{comp_idx}\n({n_components})"
                labels.append(node_label)
                node_id += 1
        
        # Calculate major transitions only (>10% threshold)
        threshold = 0.1  # 10% threshold
        
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i+1]
            
            current_labels = self.gmm_results[current_layer]['labels']
            next_labels = self.gmm_results[next_layer]['labels']
            
            current_n_components = self.layer_optimal_components[current_layer]
            next_n_components = self.layer_optimal_components[next_layer]
            
            for src_comp in range(current_n_components):
                src_mask = current_labels == src_comp
                src_count = np.sum(src_mask)
                if src_count > 0:
                    next_clusters = next_labels[src_mask]
                    for dst_comp in range(next_n_components):
                        count = np.sum(next_clusters == dst_comp)
                        proportion = count / src_count
                        
                        if proportion > threshold:  # Only major transitions
                            source_nodes.append(layer_node_mapping[current_layer][src_comp])
                            target_nodes.append(layer_node_mapping[next_layer][dst_comp])
                            values.append(count)
        
        # Create simplified Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color='lightblue'
            ),
            link=dict(
                source=source_nodes,
                target=target_nodes,
                value=values,
                color='rgba(0,0,0,0.3)'
            )
        )])
        
        fig.update_layout(
            title_text=f"Major Component Transitions (>10% threshold) - Layer-Specific Components<br><sub>Showing only significant flows between components with different counts per layer</sub>",
            font_size=10,
            width=1200,
            height=600
        )
        
        # Save simplified version
        simplified_path = self.output_dir / 'component_transitions_sankey_simplified.html'
        fig.write_html(str(simplified_path))
        
        try:
            simplified_png_path = self.output_dir / 'component_transitions_sankey_simplified.png'
            fig.write_image(str(simplified_png_path), width=1200, height=600, scale=2)
            print(f"Simplified Sankey diagram saved as PNG: {simplified_png_path}")
        except Exception as e:
            print(f"Could not save simplified PNG version: {e}")
        
        print(f"Simplified Sankey diagram saved as HTML: {simplified_path}")
    
    def export_numerical_data_to_csv(self):
        """Export all numerical data used in plots to CSV files"""
        print("Exporting numerical data to CSV files...")
        
        # 1. Export cluster assignments for each layer
        layer_names = list(self.layer_embeddings.keys())
        cluster_data = {}
        for layer_name in layer_names:
            gmm_result = self.gmm_results[layer_name]
            cluster_data[f'{layer_name}_cluster'] = gmm_result['labels']
            cluster_data[f'{layer_name}_prob_max'] = np.max(gmm_result['probabilities'], axis=1)
            cluster_data[f'{layer_name}_entropy'] = -np.sum(gmm_result['probabilities'] * np.log(gmm_result['probabilities'] + 1e-10), axis=1)
        
        df_clusters = pd.DataFrame(cluster_data)
        df_clusters.to_csv(self.output_dir / 'cluster_assignments_and_metrics.csv', index=False)
        
        # 2. Export component sharpening metrics
        sharpening_data = []
        for layer_name in layer_names:
            gmm_result = self.gmm_results[layer_name]
            covariances = gmm_result['covariances']
            
            # Calculate individual component metrics
            layer_determinants = []
            layer_traces = []
            for i, cov in enumerate(covariances):
                sign, logabsdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    logabsdet = -1e12
                layer_determinants.append(logabsdet)
                layer_traces.append(np.trace(cov))
            
            for comp_idx in range(len(covariances)):
                sharpening_data.append({
                    'layer': layer_name,
                    'component': comp_idx,
                    'log_determinant': layer_determinants[comp_idx],
                    'trace': layer_traces[comp_idx],
                    'weight': gmm_result['weights'][comp_idx]
                })
        
        df_sharpening = pd.DataFrame(sharpening_data)
        df_sharpening.to_csv(self.output_dir / 'component_sharpening_metrics.csv', index=False)
        
        # 3. Export layer-specific component counts
        component_counts_data = []
        for layer_name, n_components in self.layer_optimal_components.items():
            component_counts_data.append({
                'layer': layer_name,
                'optimal_components': n_components,
                'method': 'AIC'
            })
        
        df_component_counts = pd.DataFrame(component_counts_data)
        df_component_counts.to_csv(self.output_dir / 'layer_specific_component_counts.csv', index=False)
        
        # 4. Export boundary molecule analysis
        final_layer = list(self.gmm_results.keys())[-1]
        final_gmm = self.gmm_results[final_layer]
        probabilities = final_gmm['probabilities']
        entropies = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        
        boundary_data = []
        for i in range(len(entropies)):
            boundary_data.append({
                'molecule_index': i,
                'entropy': entropies[i],
                'max_probability': np.max(probabilities[i]),
                'assigned_cluster': final_gmm['labels'][i]
            })
        
        df_boundary = pd.DataFrame(boundary_data)
        df_boundary.to_csv(self.output_dir / 'boundary_molecule_analysis.csv', index=False)
        
        # 5. Export component selection metrics for each layer
        # This would need to be stored during the find_optimal_components_for_layer calls
        # For now, create a summary
        summary_data = []
        for layer_name, n_components in self.layer_optimal_components.items():
            summary_data.append({
                'layer': layer_name,
                'optimal_components': n_components,
                'method': 'AIC'
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(self.output_dir / 'analysis_summary_metrics.csv', index=False)
        
        print("CSV export completed:")
        print("  - cluster_assignments_and_metrics.csv")
        print("  - component_sharpening_metrics.csv") 
        print("  - layer_specific_component_counts.csv")
        print("  - boundary_molecule_analysis.csv")
        print("  - analysis_summary_metrics.csv")
    
    def run_complete_analysis(self, use_pre_saved_embeddings=False, zca_threshold=1e-5):
        """Run the complete GMM analysis pipeline with layer-specific components and ZCA whitening"""
        print("Starting complete GNN GMM analysis with layer-specific components and ZCA whitening...")
        start_time = time.time()
        
        if use_pre_saved_embeddings:
            # Step 1: Load pre-saved embeddings
            self.load_pre_saved_embeddings()
        else:
            raise NotImplementedError("This script is designed for pre-saved embeddings only")
        
        # Step 2: Fit GMMs to all layers with layer-specific optimal components and ZCA whitening
        self.fit_layer_gmms_with_specific_components(zca_threshold=zca_threshold)
        
        # Step 3: Create component count evolution visualization
        self.create_component_count_evolution_plot()
        
        # Step 4: Visualize layer evolution
        self.visualize_layer_evolution()
        
        # Step 5: Quantify cluster sharpening
        self.quantify_cluster_sharpening()
        
        # Step 6: Analyze boundary molecules
        self.analyze_boundary_molecules()
        
        # Step 7: Create parallel coordinates plot
        self.create_parallel_coordinates_plot()
        
        # Step 8: Create fuzzy clustering analysis
        self.create_fuzzy_clustering_analysis()
        
        # Step 9: Create Sankey diagram for component transitions
        self.create_sankey_diagram()
        
        # Step 10: Export numerical data to CSV files
        self.export_numerical_data_to_csv()
        
        # Step 11: Generate summary report
        self._generate_summary_report()
        
        total_time = time.time() - start_time
        print(f"\nComplete analysis finished in {total_time:.2f} seconds")
        print(f"Results saved in: {self.output_dir}")
        print(f"\nComponent counts per layer:")
        for layer_name, n_comp in self.layer_optimal_components.items():
            print(f"  {layer_name}: {n_comp} components")
    
    def _generate_summary_report(self):
        """Generate a summary report of the analysis"""
        report_path = self.output_dir / 'analysis_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("GNN GMM Analysis Summary (Layer-Specific Components)\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Checkpoint Index: {self.checkpoint_index}\n")
            f.write(f"Sample Fraction: {self.sample_frac}\n\n")
            
            f.write("Layer-Specific Component Counts:\n")
            for layer_name, n_components in self.layer_optimal_components.items():
                f.write(f"  {layer_name}: {n_components} components\n")
            
            f.write(f"\nLayer Information:\n")
            for i, (layer_name, embeddings) in enumerate(self.layer_embeddings.items()):
                n_components = self.layer_optimal_components[layer_name]
                f.write(f"  Layer {i}: {layer_name} - Shape: {embeddings.shape} - Components: {n_components}\n")
            
            f.write(f"\nAnalysis completed successfully!\n")
            f.write(f"Total layers analyzed: {len(self.layer_embeddings)}\n")
            f.write(f"Output directory: {self.output_dir}\n")
        
        print(f"Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="GNN Layer-wise GMM Analysis with Layer-Specific Components")
    parser.add_argument('--model_type', type=str, default='dimenet-10conf',
                       choices=['dimenet-10conf', 'dimenet-100conf', 'dimenet-noaug'],
                       help='Type of DimeNet model to analyze')
    parser.add_argument('--checkpoint_index', type=int, default=0,
                       help='Index of checkpoint to load (0-based)')
    parser.add_argument('--output_dir', type=str, default='../GNNGMMAnalyzerResults/gnn_gmm_layer_specific_results',
                       help='Output directory for results')
    parser.add_argument('--sample_frac', type=float, default=1.0,
                       help='Fraction of data to sample (1.0 = use all)')
    parser.add_argument('--max_components', type=int, default=30,
                       help='Maximum number of components to test for BIC/AIC per layer')
    parser.add_argument('--embeddings_dir', type=str, default=None,
                       help='Directory containing pre-saved embeddings (.pt files)')
    parser.add_argument('--use_pre_saved', action='store_true', default=True,
                       help='Use pre-saved embeddings instead of extracting from model')
    parser.add_argument('--zca_threshold', type=float, default=1e-5,
                       help='Threshold for ZCA whitening (default: 1e-5)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = GNNGMMAnalyzerLayerSpecific(
        model_type=args.model_type,
        checkpoint_index=args.checkpoint_index,
        output_dir=args.output_dir,
        sample_frac=args.sample_frac,
        embeddings_dir=args.embeddings_dir
    )
    
    # Run analysis
    analyzer.run_complete_analysis(use_pre_saved_embeddings=args.use_pre_saved, zca_threshold=args.zca_threshold)


def run_analysis_with_pre_saved_embeddings_layer_specific(embeddings_dir, output_dir=None, max_components=30, 
                                                          model_type="dimenet-10conf", checkpoint_index=0, 
                                                          sample_frac=1.0, experiment_name=None, zca_threshold=1e-5):
    """
    Convenience function to run GMM analysis with pre-saved embeddings, layer-specific components, and ZCA whitening
    
    Args:
        embeddings_dir (str): Path to directory containing .pt embedding files
        output_dir (str): Output directory for results (optional)
        max_components (int): Maximum number of components to test for BIC/AIC per layer
        model_type (str): Type of model (for naming purposes)
        checkpoint_index (int): Checkpoint index (for naming purposes)
        sample_frac (float): Fraction of data to use (1.0 = use all)
        experiment_name (str): Custom name for this experiment (optional)
        zca_threshold (float): Threshold for ZCA whitening (default: 1e-5)
    
    Returns:
        GNNGMMAnalyzerLayerSpecific: The analyzer object with results
    """
    if output_dir is None:
        # Create default output directory with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            output_dir = f"../GNNGMMAnalyzerResults/{experiment_name}_{timestamp}"
        else:
            output_dir = f"../GNNGMMAnalyzerResults/{model_type}_checkpoint{checkpoint_index}_layer_specific_{timestamp}"
    
    print(f"Starting GMM analysis with pre-saved embeddings and layer-specific components...")
    print(f"Embeddings directory: {embeddings_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max components per layer: {max_components}")
    
    # Create analyzer
    analyzer = GNNGMMAnalyzerLayerSpecific(
        model_type=model_type,
        checkpoint_index=checkpoint_index,
        output_dir=output_dir,
        sample_frac=sample_frac,
        embeddings_dir=embeddings_dir
    )
    
    # Run analysis with pre-saved embeddings, layer-specific components, and ZCA whitening
    analyzer.run_complete_analysis(use_pre_saved_embeddings=True, zca_threshold=zca_threshold)
    
    return analyzer


if __name__ == "__main__":
    main()
