#!/usr/bin/env python3
"""
GNN Layer-wise GMM Analysis Script

This script performs comprehensive GMM analysis on GNN embeddings across different layers:
1. Baseline GMM fitting with BIC/AIC for optimal components
2. Layer-wise trajectory analysis
3. Visualization of evolution across layers
4. Quantification of cluster sharpening
5. Boundary analysis for ambiguous molecules
6. Fuzzy clustering and parallel coordinates plots

Usage:
    python gnn_gmm_analysis.py --model_type dimenet-10conf --checkpoint_index 0
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

class GNNGMMAnalyzer:
    """Comprehensive GMM analysis for GNN embeddings across layers"""
    
    def __init__(self, model_type="dimenet-10conf", checkpoint_index=0, 
                 output_dir="../GNNGMMAnalyzerResults/gnn_gmm_results", sample_frac=1.0,
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
        self.optimal_components = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_model_and_data(self):
        """Load the GNN model and data"""
        print(f"Loading {self.model_type} model (checkpoint {self.checkpoint_index})...")
        
        if self.model_type == "dimenet-10conf":
            self.model, self.data, self.smiles = dimenet10conf()
        else:
            self.model, self.data, self.smiles = dimenet_model(self.model_type, self.checkpoint_index)
        
        print("Model loaded successfully. Data size: {} samples".format(len(self.data)))
        
        # Sample data if needed
        if self.sample_frac < 1.0:
            # Convert DataLoader to list first
            data_list = list(self.data)
            n_samples = int(len(data_list) * self.sample_frac)
            indices = np.random.choice(len(data_list), n_samples, replace=False)
            self.data = [data_list[i] for i in indices]
            print("Sampled {} samples from {} total".format(n_samples, len(data_list)))
    
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
        layer_files = {
            'emb.pt': 'emb',
            'output0.pt': 'output_block_0',
            'output1.pt': 'output_block_1', 
            'output2.pt': 'output_block_2',
            'output3.pt': 'output_block_3',
            'output4.pt': 'output_block_4'
        }
        
        # Load embeddings for each layer
        for filename, layer_name in layer_files.items():
            filepath = embeddings_path / filename
            if filepath.exists():
                print(f"  Loading {filename} -> {layer_name}")
                try:
                    # Load tensor and convert to numpy
                    embeddings_tensor = torch.load(filepath, map_location='cpu')
                    embeddings_np = embeddings_tensor.detach().cpu().numpy()
                    self.layer_embeddings[layer_name] = embeddings_np
                    print(f"    Shape: {embeddings_np.shape}")
                except Exception as e:
                    print(f"    Error loading {filename}: {e}")
            else:
                print(f"  Warning: {filename} not found")
        
        if not self.layer_embeddings:
            raise ValueError("No embeddings were successfully loaded")
        
        print(f"Successfully loaded embeddings from {len(self.layer_embeddings)} layers")
        return self.layer_embeddings
    
    def extract_layer_embeddings(self):
        """Extract embeddings from all GNN layers efficiently"""
        print("Extracting embeddings from all GNN layers...")
        
        # Define layers to extract
        layers = [self.model.gnn.emb] + list(self.model.gnn.output_blocks)
        layer_names = ["emb"] + [f"output_block_{i}" for i in range(len(self.model.gnn.output_blocks))]
        
        # Create embeddings directory
        embeddings_dir = self.output_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        
        # Extract embeddings for each layer
        for i, (layer, name) in enumerate(zip(layers, layer_names)):
            print(f"  Extracting layer {i}: {name}")
            
            if LATENTLIB_AVAILABLE:
                # Use latentlib if available
                _, emb = save_embedding(self.model, layer, self.data)
                s_emb = torch.cat([y.sum(dim=0) for y in emb])
            else:
                # Alternative extraction method
                s_emb = self._extract_layer_embedding_alternative(layer)
            
            # Convert to numpy and save
            embeddings_np = s_emb.detach().cpu().numpy()
            self.layer_embeddings[name] = embeddings_np
            
            # Save embeddings
            torch.save(s_emb, embeddings_dir / f"layer_{i}_{name}.pt")
            np.save(embeddings_dir / f"layer_{i}_{name}.npy", embeddings_np)
            
            print(f"    Shape: {embeddings_np.shape}")
        
        print(f"Extracted embeddings from {len(layers)} layers")
        return self.layer_embeddings
    
    def _extract_layer_embedding_alternative(self, layer):
        """Alternative embedding extraction when latentlib is not available"""
        embeddings = []
        
        with torch.no_grad():
            for batch in self.data:
                # Forward pass through the layer
                if hasattr(batch, 'x'):
                    x = batch.x
                else:
                    x = batch
                
                # Apply layer
                if hasattr(layer, '__call__'):
                    output = layer(x)
                else:
                    output = x
                
                # Sum across nodes for graph-level embedding
                if hasattr(output, 'sum'):
                    emb = output.sum(dim=0)
                else:
                    emb = output
                
                embeddings.append(emb)
        
        return torch.stack(embeddings)
    
    def find_optimal_components(self, embeddings, max_components=20):
        """Find optimal number of components using BIC and AIC"""
        print("Finding optimal number of components using BIC/AIC...")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Test different numbers of components
        n_components_range = range(2, max_components + 1)
        bic_scores = []
        aic_scores = []
        silhouette_scores = []
        
        for n_components in n_components_range:
            print(f"  Testing {n_components} components...")
            
            # Fit GMM
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=42,
                max_iter=100
            )
            gmm.fit(embeddings_scaled)
            
            # Calculate scores
            bic_scores.append(gmm.bic(embeddings_scaled))
            aic_scores.append(gmm.aic(embeddings_scaled))
            
            # Calculate silhouette score
            labels = gmm.predict(embeddings_scaled)
            sil_score = silhouette_score(embeddings_scaled, labels)
            silhouette_scores.append(sil_score)
        
        # Find optimal components
        optimal_bic = n_components_range[np.argmin(bic_scores)]
        optimal_aic = n_components_range[np.argmin(aic_scores)]
        optimal_sil = n_components_range[np.argmax(silhouette_scores)]
        
        print(f"  Optimal components - BIC: {optimal_bic}, AIC: {optimal_aic}, Silhouette: {optimal_sil}")
        
        # Use AIC as primary criterion (often more conservative)
        self.optimal_components = optimal_aic
        
        # Plot the results
        self._plot_component_selection(n_components_range, bic_scores, aic_scores, silhouette_scores)
        
        return self.optimal_components
    
    def _plot_component_selection(self, n_components_range, bic_scores, aic_scores, silhouette_scores):
        """Plot component selection results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # BIC plot
        axes[0].plot(n_components_range, bic_scores, 'o-', label='BIC')
        axes[0].axvline(n_components_range[np.argmin(bic_scores)], color='red', linestyle='--', 
                       label=f'Optimal: {n_components_range[np.argmin(bic_scores)]}')
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('BIC Score')
        axes[0].set_title('BIC vs Number of Components')
        axes[0].legend()
        axes[0].grid(True)
        
        # AIC plot
        axes[1].plot(n_components_range, aic_scores, 'o-', label='AIC')
        axes[1].axvline(n_components_range[np.argmin(aic_scores)], color='red', linestyle='--',
                       label=f'Optimal: {n_components_range[np.argmin(aic_scores)]}')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('AIC Score')
        axes[1].set_title('AIC vs Number of Components')
        axes[1].legend()
        axes[1].grid(True)
        
        # Silhouette plot
        axes[2].plot(n_components_range, silhouette_scores, 'o-', label='Silhouette')
        axes[2].axvline(n_components_range[np.argmax(silhouette_scores)], color='red', linestyle='--',
                       label=f'Optimal: {n_components_range[np.argmax(silhouette_scores)]}')
        axes[2].set_xlabel('Number of Components')
        axes[2].set_ylabel('Silhouette Score')
        axes[2].set_title('Silhouette Score vs Number of Components')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_selection.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def fit_layer_gmms(self, n_components=None):
        """Fit GMM to embeddings from each layer with cluster alignment"""
        if n_components is None:
            n_components = self.optimal_components
        
        print(f"Fitting GMMs to all layers with {n_components} components...")
        
        # First, fit GMMs to all layers without alignment
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
                max_iter=200
            )
            gmm.fit(embeddings_scaled)
            
            # Store results
            self.gmm_results[layer_name] = {
                'gmm': gmm,
                'scaler': scaler,
                'embeddings_scaled': embeddings_scaled,
                'embeddings_original': embeddings,
                'means': gmm.means_,
                'covariances': gmm.covariances_,
                'weights': gmm.weights_,
                'labels': gmm.predict(embeddings_scaled),
                'probabilities': gmm.predict_proba(embeddings_scaled)
            }
        
        # Now align cluster indices across layers
        print("Aligning cluster indices across layers...")
        self._align_cluster_indices()
        
        print(f"Fitted GMMs to {len(self.layer_embeddings)} layers with aligned cluster indices")
    
    def _align_cluster_indices(self):
        """Align cluster indices across layers to maximize diagonal elements in transition matrices"""
        layer_names = list(self.gmm_results.keys())
        n_components = self.optimal_components
        
        # Start with the first layer as reference
        reference_layer = layer_names[0]
        print(f"  Using {reference_layer} as reference layer")
        
        # For each subsequent layer, find the best permutation of cluster indices
        for i in range(1, len(layer_names)):
            current_layer = layer_names[i]
            previous_layer = layer_names[i-1]
            
            print(f"  Aligning {current_layer} with {previous_layer}")
            
            # Get current cluster assignments
            current_labels = self.gmm_results[current_layer]['labels'].copy()
            previous_labels = self.gmm_results[previous_layer]['labels'].copy()
            
            # Create transition matrix to find best alignment
            transition_matrix = np.zeros((n_components, n_components))
            for prev_cluster in range(n_components):
                prev_mask = previous_labels == prev_cluster
                if np.sum(prev_mask) > 0:
                    current_clusters = current_labels[prev_mask]
                    for curr_cluster in range(n_components):
                        transition_matrix[prev_cluster, curr_cluster] = np.sum(current_clusters == curr_cluster)
            
            # Find the permutation that maximizes diagonal elements
            best_permutation = self._find_best_cluster_permutation(transition_matrix)
            
            # Apply the permutation to current layer
            self._apply_cluster_permutation(current_layer, best_permutation)
            
            print(f"    Applied permutation: {best_permutation}")
    
    def _find_best_cluster_permutation(self, transition_matrix):
        """Find the permutation of cluster indices that maximizes diagonal elements"""
        from itertools import permutations
        
        n_components = transition_matrix.shape[0]
        best_permutation = None
        best_diagonal_sum = -1
        
        # Try all possible permutations
        for perm in permutations(range(n_components)):
            diagonal_sum = 0
            for i in range(n_components):
                diagonal_sum += transition_matrix[i, perm[i]]
            
            if diagonal_sum > best_diagonal_sum:
                best_diagonal_sum = diagonal_sum
                best_permutation = list(perm)
        
        return best_permutation
    
    def _apply_cluster_permutation(self, layer_name, permutation):
        """Apply cluster index permutation to a layer's GMM results"""
        n_components = len(permutation)
        
        # Create mapping from old to new cluster indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(permutation)}
        
        # Update labels
        old_labels = self.gmm_results[layer_name]['labels'].copy()
        new_labels = np.array([old_to_new[label] for label in old_labels])
        self.gmm_results[layer_name]['labels'] = new_labels
        
        # Update probabilities
        old_probs = self.gmm_results[layer_name]['probabilities'].copy()
        new_probs = old_probs[:, permutation]
        self.gmm_results[layer_name]['probabilities'] = new_probs
        
        # Update means, covariances, and weights
        old_means = self.gmm_results[layer_name]['means'].copy()
        old_covariances = self.gmm_results[layer_name]['covariances'].copy()
        old_weights = self.gmm_results[layer_name]['weights'].copy()
        
        new_means = old_means[permutation]
        new_covariances = old_covariances[permutation]
        new_weights = old_weights[permutation]
        
        self.gmm_results[layer_name]['means'] = new_means
        self.gmm_results[layer_name]['covariances'] = new_covariances
        self.gmm_results[layer_name]['weights'] = new_weights
        
        # Update the GMM object itself
        gmm = self.gmm_results[layer_name]['gmm']
        gmm.means_ = new_means
        gmm.covariances_ = new_covariances
        gmm.weights_ = new_weights
    
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
            means_2d = pca.transform(gmm_result['scaler'].inverse_transform(gmm_result['means']))
            
            # Plot points
            if molecule_classes is not None:
                scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=molecule_classes, cmap='tab10', alpha=0.6, s=20)
            else:
                scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   alpha=0.6, s=20)
            
            # Plot GMM ellipses
            self._plot_gmm_ellipses(ax, means_2d, gmm_result['covariances'], 
                                  gmm_result['scaler'], pca)
            
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
        plt.savefig(self.output_dir / 'layer_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Layer evolution visualization saved")
    
    def _plot_gmm_ellipses(self, ax, means_2d, covariances, scaler, pca):
        """Plot GMM ellipses on 2D projection"""
        for i, cov in enumerate(covariances):
            mean_2d = means_2d[i]
            
            # Transform covariance to 2D space
            cov_2d = pca.components_ @ scaler.inverse_transform(cov) @ pca.components_.T
            
            # Calculate ellipse parameters
            eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 2 * np.sqrt(eigenvals)
            
            # Plot ellipse
            ellipse = Ellipse(mean_2d, width, height, angle=angle, 
                           alpha=0.3, facecolor='red', edgecolor='red')
            ax.add_patch(ellipse)
    
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
        
        n_components = self.optimal_components
        
        # Plot 1: Individual component log-determinants across layers
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create data matrix: rows = components, columns = layers
        det_matrix = np.zeros((n_components, len(layers)))
        for i, layer_name in enumerate(layers):
            det_matrix[:, i] = individual_determinants[layer_name]
        
        # Plot each component as a separate line
        colors = plt.cm.tab10(np.linspace(0, 1, n_components))
        for comp_idx in range(n_components):
            ax.plot(range(len(layers)), det_matrix[comp_idx, :], 
                   'o-', linewidth=2, markersize=6, color=colors[comp_idx],
                   label=f'Component {comp_idx}')
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('log|Determinant|')
        ax.set_title('Individual Component log-Determinants Across Layers')
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
        trace_matrix = np.zeros((n_components, len(layers)))
        for i, layer_name in enumerate(layers):
            trace_matrix[:, i] = individual_traces[layer_name]
        
        # Plot each component as a separate line
        for comp_idx in range(n_components):
            ax.plot(range(len(layers)), trace_matrix[comp_idx, :], 
                   'o-', linewidth=2, markersize=6, color=colors[comp_idx],
                   label=f'Component {comp_idx}')
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Trace')
        ax.set_title('Individual Component Traces Across Layers')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_sharpening_individual_traces.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Heatmap of log-determinants (components vs layers)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(det_matrix, cmap='viridis', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(layers)))
        ax.set_yticks(range(n_components))
        ax.set_xticklabels(layers, rotation=45)
        ax.set_yticklabels([f'Component {i}' for i in range(n_components)])
        
        ax.set_xlabel('GNN Layer')
        ax.set_ylabel('GMM Component')
        ax.set_title('log|Determinant| Heatmap: Components vs Layers')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log|Determinant|')
        
        # Add text annotations
        for i in range(n_components):
            for j in range(len(layers)):
                text = ax.text(j, i, f'{det_matrix[i, j]:.2f}',
                             ha="center", va="center", 
                             color="white" if det_matrix[i, j] < np.median(det_matrix) else "black")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_sharpening_determinant_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Heatmap of traces (components vs layers)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(trace_matrix, cmap='plasma', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(layers)))
        ax.set_yticks(range(n_components))
        ax.set_xticklabels(layers, rotation=45)
        ax.set_yticklabels([f'Component {i}' for i in range(n_components)])
        
        ax.set_xlabel('GNN Layer')
        ax.set_ylabel('GMM Component')
        ax.set_title('Trace Heatmap: Components vs Layers')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Trace Value')
        
        # Add text annotations
        for i in range(n_components):
            for j in range(len(layers)):
                text = ax.text(j, i, f'{trace_matrix[i, j]:.2e}',
                             ha="center", va="center", 
                             color="white" if trace_matrix[i, j] < np.median(trace_matrix) else "black")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_sharpening_trace_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 5: Box plots for each layer showing component distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Determinant box plots
        det_data = [individual_determinants[layer] for layer in layers]
        bp1 = ax1.boxplot(det_data, labels=layers, patch_artist=True)
        ax1.set_xlabel('GNN Layer')
        ax1.set_ylabel('Determinant')
        ax1.set_title('Determinant Distribution Across Components (per Layer)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(layers)))
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        
        # Trace box plots
        trace_data = [individual_traces[layer] for layer in layers]
        bp2 = ax2.boxplot(trace_data, labels=layers, patch_artist=True)
        ax2.set_xlabel('GNN Layer')
        ax2.set_ylabel('Trace')
        ax2.set_title('Trace Distribution Across Components (per Layer)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Color the boxes
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_sharpening_component_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Individual component sharpening plots saved")
    
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
        
        # 3. Export transition matrices
        transition_data = []
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i+1]
            previous_layer = layer_names[i]
            
            current_labels = self.gmm_results[current_layer]['labels']
            previous_labels = self.gmm_results[previous_layer]['labels']
            
            # Create transition matrix
            n_components = self.optimal_components
            transition_matrix = np.zeros((n_components, n_components))
            for prev_cluster in range(n_components):
                prev_mask = previous_labels == prev_cluster
                if np.sum(prev_mask) > 0:
                    current_clusters = current_labels[prev_mask]
                    for curr_cluster in range(n_components):
                        transition_matrix[prev_cluster, curr_cluster] = np.sum(current_clusters == curr_cluster)
            
            # Normalize by row sums
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix_norm = transition_matrix / (row_sums + 1e-10)
            
            # Export as long format
            for prev_cluster in range(n_components):
                for curr_cluster in range(n_components):
                    transition_data.append({
                        'from_layer': previous_layer,
                        'to_layer': current_layer,
                        'from_cluster': prev_cluster,
                        'to_cluster': curr_cluster,
                        'count': transition_matrix[prev_cluster, curr_cluster],
                        'probability': transition_matrix_norm[prev_cluster, curr_cluster]
                    })
        
        df_transitions = pd.DataFrame(transition_data)
        df_transitions.to_csv(self.output_dir / 'cluster_transition_matrices.csv', index=False)
        
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
        
        # 5. Export component selection metrics
        # This would need to be stored during the find_optimal_components call
        # For now, create a summary
        summary_data = [{
            'metric': 'optimal_components',
            'value': self.optimal_components,
            'method': 'AIC'
        }]
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(self.output_dir / 'analysis_summary_metrics.csv', index=False)
        
        print("CSV export completed:")
        # print("  - cluster_assignments_and_metrics.csv")
        # print("  - component_sharpening_metrics.csv") 
        # print("  - cluster_transition_matrices.csv")
        # print("  - boundary_molecule_analysis.csv")
        # print("  - analysis_summary_metrics.csv")
    
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
        ax.set_title('Layer-wise Trajectory: Parallel Coordinates Plot')
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
            
            # Add mean trajectory
            # mean_values = df_sample[pc_cols].mean().values
            # ax.plot(range(len(mean_values)), mean_values, 
                #    color='red', linewidth=3, alpha=0.8, label='Mean trajectory')
            
            # Add standard deviation bands
            # std_values = df_sample[pc_cols].std().values
            # upper_bound = mean_values + std_values
            # lower_bound = mean_values - std_values
            # ax.fill_between(range(len(mean_values)), lower_bound, upper_bound, 
            #                alpha=0.2, color='red', label='±1 std')
            
            ax.set_xticks(range(len(pc_cols)))
            ax.set_xticklabels([col.replace('_PC' + str(pc_num), '') for col in pc_cols], 
                              rotation=45, ha='right')
            ax.set_ylabel(f'PC{pc_num} Value')
            ax.set_title(f'Principal Component {pc_num} Evolution Across GNN Layers')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'PC{pc_num}_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  PC{pc_num} evolution plot saved")
        
        print("Separate PC plots completed")
    
    def create_cluster_tracking_parallel_plot(self):
        """Create parallel coordinates plot tracking cluster assignments across layers"""
        print("Creating cluster tracking parallel coordinates plot...")
        
        # Prepare data for cluster tracking
        n_layers = len(self.layer_embeddings)
        n_samples = len(list(self.layer_embeddings.values())[0])
        
        # Create DataFrame with cluster assignments for each layer
        cluster_data = {}
        layer_names = list(self.layer_embeddings.keys())
        
        for layer_name in layer_names:
            gmm_result = self.gmm_results[layer_name]
            cluster_data[f'{layer_name}_cluster'] = gmm_result['labels']
        
        df_clusters = pd.DataFrame(cluster_data)
        
        # Use all data for cluster tracking (since clusters are discrete, overlap is manageable)
        df_sample = df_clusters
        sample_indices_original = np.arange(len(df_clusters))
        
        # Create the cluster tracking plot
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot each molecule's cluster trajectory
        print(f"  Plotting {len(df_sample)} molecule trajectories...")
        for i, (idx, row) in enumerate(df_sample.iterrows()):
            molecule_id = sample_indices_original[i]
            cluster_values = row.values
            
            # Add small random jitter to make overlapping lines visible
            jittered_values = cluster_values + np.random.normal(0, 0.05, len(cluster_values))
            
            # Plot the trajectory with better visibility
            ax.plot(range(len(jittered_values)), jittered_values, 
                   alpha=0.6, linewidth=0.8, color='gray')
        
        # Mean cluster trajectory removed as requested
        
        # Add cluster distribution bars for each layer
        for i, layer_name in enumerate(layer_names):
            cluster_col = f'{layer_name}_cluster'
            cluster_counts = df_sample[cluster_col].value_counts().sort_index()
            
            # Add small bars showing cluster distribution
            for cluster_id, count in cluster_counts.items():
                bar_height = count / len(df_sample) * 0.5  # Scale to fit
                ax.bar(i, bar_height, bottom=cluster_id, alpha=0.3, 
                      color=plt.cm.tab10(cluster_id / (self.optimal_components - 1)))
        
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_ylabel('GMM Cluster Assignment')
        ax.set_xlabel('GNN Layer')
        ax.set_title('Cluster Assignment Tracking Across GNN Layers\n(Each line = one molecule\'s cluster trajectory)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar for cluster IDs - use only the colors we need
        from matplotlib.colors import ListedColormap
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, self.optimal_components))
        cluster_cmap = ListedColormap(cluster_colors)
        
        sm = plt.cm.ScalarMappable(cmap=cluster_cmap, 
                                 norm=plt.Normalize(vmin=0, vmax=self.optimal_components-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Cluster ID')
        cbar.set_ticks(range(self.optimal_components))
        cbar.set_ticklabels([f'Cluster {i}' for i in range(self.optimal_components)])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_tracking_parallel.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a more detailed version with cluster transition matrix
        self._create_cluster_transition_analysis(df_sample, layer_names)
        
        print("Cluster tracking parallel coordinates plot saved")
    
    def _create_cluster_transition_analysis(self, df_sample, layer_names):
        """Create cluster transition analysis and visualization"""
        print("Creating cluster transition analysis...")
        
        # Create transition matrices between consecutive layers
        transition_matrices = []
        
        for i in range(len(layer_names) - 1):
            current_layer = f'{layer_names[i]}_cluster'
            next_layer = f'{layer_names[i+1]}_cluster'
            
            # Create transition matrix
            transition_matrix = pd.crosstab(
                df_sample[current_layer], 
                df_sample[next_layer], 
                normalize='index'
            ).fillna(0)
            
            transition_matrices.append({
                'from_layer': layer_names[i],
                'to_layer': layer_names[i+1],
                'matrix': transition_matrix
            })
        
        # Create visualization
        n_transitions = len(transition_matrices)
        fig, axes = plt.subplots(1, n_transitions, figsize=(5*n_transitions, 5))
        
        if n_transitions == 1:
            axes = [axes]
        
        for i, trans_data in enumerate(transition_matrices):
            ax = axes[i]
            matrix = trans_data['matrix']
            
            # Plot heatmap
            im = ax.imshow(matrix.values, cmap='Blues', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(len(matrix.columns)))
            ax.set_yticks(range(len(matrix.index)))
            ax.set_xticklabels([f'C{j}' for j in matrix.columns])
            ax.set_yticklabels([f'C{j}' for j in matrix.index])
            
            ax.set_xlabel(f'{trans_data["to_layer"]} Cluster')
            ax.set_ylabel(f'{trans_data["from_layer"]} Cluster')
            ax.set_title(f'{trans_data["from_layer"]} → {trans_data["to_layer"]}')
            
            # Add text annotations
            for row in range(len(matrix.index)):
                for col in range(len(matrix.columns)):
                    text = ax.text(col, row, f'{matrix.iloc[row, col]:.2f}',
                                 ha="center", va="center", color="black" if matrix.iloc[row, col] < 0.5 else "white")
            
            plt.colorbar(im, ax=ax, label='Transition Probability')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_transition_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create cluster stability plot
        self._create_cluster_stability_plot(df_sample, layer_names)
        
        print("Cluster transition analysis completed")
    
    def _create_cluster_stability_plot(self, df_sample, layer_names):
        """Create cluster stability analysis across layers"""
        print("Creating cluster stability plot...")
        
        # Calculate cluster stability metrics
        stability_metrics = []
        
        for i in range(len(layer_names) - 1):
            current_layer = f'{layer_names[i]}_cluster'
            next_layer = f'{layer_names[i+1]}_cluster'
            
            # Calculate how many molecules stay in the same cluster
            same_cluster = (df_sample[current_layer] == df_sample[next_layer]).sum()
            total_molecules = len(df_sample)
            stability = same_cluster / total_molecules
            
            stability_metrics.append({
                'transition': f'{layer_names[i]} → {layer_names[i+1]}',
                'stability': stability,
                'same_cluster': same_cluster,
                'total': total_molecules
            })
        
        # Create stability plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Stability bar plot
        transitions = [m['transition'] for m in stability_metrics]
        stabilities = [m['stability'] for m in stability_metrics]
        
        bars = ax1.bar(range(len(transitions)), stabilities, color='skyblue', alpha=0.7)
        ax1.set_xticks(range(len(transitions)))
        ax1.set_xticklabels(transitions, rotation=45, ha='right')
        ax1.set_ylabel('Cluster Stability (Fraction staying in same cluster)')
        ax1.set_title('Cluster Stability Across Layer Transitions')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, stability) in enumerate(zip(bars, stabilities)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{stability:.3f}', ha='center', va='bottom')
        
        # Overall stability trend
        ax2.plot(range(len(stabilities)), stabilities, 'o-', linewidth=2, markersize=8)
        ax2.set_xticks(range(len(transitions)))
        ax2.set_xticklabels(transitions, rotation=45, ha='right')
        ax2.set_ylabel('Cluster Stability')
        ax2.set_title('Cluster Stability Trend')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Cluster stability plot saved")
    
    def create_molecule_cluster_heatmap(self):
        """Create heatmap showing cluster assignments for each molecule across layers"""
        print("Creating molecule cluster heatmap...")
        
        # Prepare data for heatmap
        n_layers = len(self.layer_embeddings)
        n_samples = len(list(self.layer_embeddings.values())[0])
        layer_names = list(self.layer_embeddings.keys())
        
        # Create matrix: rows = molecules, columns = layers
        cluster_matrix = np.zeros((n_samples, n_layers))
        
        for i, layer_name in enumerate(layer_names):
            gmm_result = self.gmm_results[layer_name]
            cluster_matrix[:, i] = gmm_result['labels']
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 16))
        
        # Create custom colormap with only the colors we need
        from matplotlib.colors import ListedColormap
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, self.optimal_components))
        cluster_cmap = ListedColormap(cluster_colors)
        
        # Create heatmap
        im = ax.imshow(cluster_matrix, cmap=cluster_cmap, aspect='auto', 
                      vmin=0, vmax=self.optimal_components-1)
        
        # Set labels
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_yticks(range(0, n_samples, 100))  # Show every 100th molecule
        ax.set_yticklabels(range(0, n_samples, 100))
        
        ax.set_xlabel('GNN Layer')
        ax.set_ylabel('Molecule Index')
        ax.set_title('Cluster Assignment Heatmap\n(Each row = one molecule, each column = one layer)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cluster ID')
        cbar.set_ticks(range(self.optimal_components))
        cbar.set_ticklabels([f'Cluster {i}' for i in range(self.optimal_components)])
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'molecule_cluster_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a more detailed version with cluster change indicators
        self._create_cluster_change_heatmap(cluster_matrix, layer_names)
        
        print("Molecule cluster heatmap saved")
    
    def _create_cluster_change_heatmap(self, cluster_matrix, layer_names):
        """Create heatmap highlighting cluster changes"""
        print("Creating cluster change heatmap...")
        
        n_samples, n_layers = cluster_matrix.shape
        
        # Create change matrix: 1 = changed cluster, 0 = same cluster
        change_matrix = np.zeros((n_samples, n_layers-1))
        
        for i in range(n_layers - 1):
            change_matrix[:, i] = (cluster_matrix[:, i] != cluster_matrix[:, i+1]).astype(int)
        
        # Create the change heatmap
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
        
        # Create custom colormap with only the colors we need
        from matplotlib.colors import ListedColormap
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, self.optimal_components))
        cluster_cmap = ListedColormap(cluster_colors)
        
        # Original cluster assignments
        im1 = ax1.imshow(cluster_matrix, cmap=cluster_cmap, aspect='auto', 
                        vmin=0, vmax=self.optimal_components-1)
        ax1.set_xticks(range(n_layers))
        ax1.set_xticklabels(layer_names, rotation=45, ha='right')
        ax1.set_yticks(range(0, n_samples, 100))
        ax1.set_yticklabels(range(0, n_samples, 100))
        ax1.set_xlabel('GNN Layer')
        ax1.set_ylabel('Molecule Index')
        ax1.set_title('Cluster Assignments Across Layers')
        ax1.grid(True, alpha=0.3)
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Cluster ID')
        cbar1.set_ticks(range(self.optimal_components))
        cbar1.set_ticklabels([f'Cluster {i}' for i in range(self.optimal_components)])
        
        # Cluster changes (red = changed, white = same)
        im2 = ax2.imshow(change_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(n_layers-1))
        ax2.set_xticklabels([f'{layer_names[i]} → {layer_names[i+1]}' for i in range(n_layers-1)], 
                           rotation=45, ha='right')
        ax2.set_yticks(range(0, n_samples, 100))
        ax2.set_yticklabels(range(0, n_samples, 100))
        ax2.set_xlabel('Layer Transition')
        ax2.set_ylabel('Molecule Index')
        ax2.set_title('Cluster Changes (Red = Changed Cluster, White = Same Cluster)')
        ax2.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Changed (1) / Same (0)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_change_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics
        self._create_cluster_change_summary(cluster_matrix, change_matrix, layer_names)
        
        print("Cluster change heatmap saved")
    
    def _create_cluster_change_summary(self, cluster_matrix, change_matrix, layer_names):
        """Create summary statistics of cluster changes"""
        print("Creating cluster change summary...")
        
        n_samples, n_layers = cluster_matrix.shape
        
        # Calculate statistics
        total_changes = np.sum(change_matrix, axis=0)
        change_percentages = (total_changes / n_samples) * 100
        
        # Molecules that never change clusters
        never_change = np.sum(np.sum(change_matrix, axis=1) == 0)
        
        # Molecules that change at least once
        ever_change = n_samples - never_change
        
        # Create summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Change percentage per transition
        transitions = [f'{layer_names[i]} → {layer_names[i+1]}' for i in range(n_layers-1)]
        bars = ax1.bar(range(len(transitions)), change_percentages, color='skyblue', alpha=0.7)
        ax1.set_xticks(range(len(transitions)))
        ax1.set_xticklabels(transitions, rotation=45, ha='right')
        ax1.set_ylabel('Percentage of Molecules Changing Clusters')
        ax1.set_title('Cluster Change Rate per Layer Transition')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, pct) in enumerate(zip(bars, change_percentages)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        # Distribution of number of changes per molecule
        changes_per_molecule = np.sum(change_matrix, axis=1)
        unique_changes, counts = np.unique(changes_per_molecule, return_counts=True)
        
        ax2.bar(unique_changes, counts, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Number of Cluster Changes')
        ax2.set_ylabel('Number of Molecules')
        ax2.set_title('Distribution of Cluster Changes per Molecule')
        ax2.grid(True, alpha=0.3)
        
        # Overall stability summary
        stability_data = [never_change, ever_change]
        labels = ['Never Change\nClusters', 'Change Clusters\nAt Least Once']
        colors = ['lightgreen', 'lightcoral']
        
        ax3.pie(stability_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Overall Cluster Stability')
        
        # Cluster distribution in final layer
        final_clusters = cluster_matrix[:, -1]
        unique_clusters, cluster_counts = np.unique(final_clusters, return_counts=True)
        
        ax4.bar(unique_clusters, cluster_counts, color='gold', alpha=0.7)
        ax4.set_xlabel('Final Cluster ID')
        ax4.set_ylabel('Number of Molecules')
        ax4.set_title('Final Cluster Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_change_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Cluster change summary saved")
    
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
        axes[0, 0].set_title('Confidence Distribution Across Layers')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ambiguity across layers
        ambiguities = [fuzzy_results[layer]['ambiguity'] for layer in layers]
        axes[0, 1].boxplot(ambiguities, labels=layers)
        axes[0, 1].set_ylabel('Ambiguity (Entropy)')
        axes[0, 1].set_title('Ambiguity Distribution Across Layers')
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
    
    def run_complete_analysis(self, use_pre_saved_embeddings=False):
        """Run the complete GMM analysis pipeline"""
        print("Starting complete GNN GMM analysis...")
        start_time = time.time()
        
        if use_pre_saved_embeddings:
            # Step 1: Load pre-saved embeddings
            self.load_pre_saved_embeddings()
        else:
            # Step 1: Load model and data
            self.load_model_and_data()
            
            # Step 2: Extract layer embeddings
            self.extract_layer_embeddings()
        
        # Step 3: Find optimal components using final layer
        final_layer = list(self.layer_embeddings.keys())[-1]
        self.find_optimal_components(self.layer_embeddings[final_layer])
        
        # Step 4: Fit GMMs to all layers
        self.fit_layer_gmms()
        
        # Step 5: Visualize layer evolution
        self.visualize_layer_evolution()
        
        # Step 6: Quantify cluster sharpening
        self.quantify_cluster_sharpening()
        
        # Step 7: Analyze boundary molecules
        self.analyze_boundary_molecules()
        
        # Step 8: Create parallel coordinates plot
        self.create_parallel_coordinates_plot()
        
        # Step 8.5: Create cluster tracking parallel coordinates plot
        self.create_cluster_tracking_parallel_plot()
        
        # Step 8.6: Create molecule cluster heatmap
        self.create_molecule_cluster_heatmap()
        
        # Step 9: Create fuzzy clustering analysis
        self.create_fuzzy_clustering_analysis()
        
        # Step 10: Export numerical data to CSV files
        self.export_numerical_data_to_csv()
        
        # Step 11: Generate summary report
        self._generate_summary_report()
        
        total_time = time.time() - start_time
        print(f"\nComplete analysis finished in {total_time:.2f} seconds")
        print(f"Results saved in: {self.output_dir}")
    
    def _generate_summary_report(self):
        """Generate a summary report of the analysis"""
        report_path = self.output_dir / 'analysis_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("GNN GMM Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Checkpoint Index: {self.checkpoint_index}\n")
            f.write(f"Sample Fraction: {self.sample_frac}\n")
            f.write(f"Optimal Components: {self.optimal_components}\n\n")
            
            f.write("Layer Information:\n")
            for i, (layer_name, embeddings) in enumerate(self.layer_embeddings.items()):
                f.write(f"  Layer {i}: {layer_name} - Shape: {embeddings.shape}\n")
            
            f.write(f"\nAnalysis completed successfully!\n")
            f.write(f"Total layers analyzed: {len(self.layer_embeddings)}\n")
            f.write(f"Output directory: {self.output_dir}\n")
        
        print(f"Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="GNN Layer-wise GMM Analysis")
    parser.add_argument('--model_type', type=str, default='dimenet-10conf',
                       choices=['dimenet-10conf', 'dimenet-100conf', 'dimenet-noaug'],
                       help='Type of DimeNet model to analyze')
    parser.add_argument('--checkpoint_index', type=int, default=0,
                       help='Index of checkpoint to load (0-based)')
    parser.add_argument('--output_dir', type=str, default='../GNNGMMAnalyzerResults/gnn_gmm_results',
                       help='Output directory for results')
    parser.add_argument('--sample_frac', type=float, default=1.0,
                       help='Fraction of data to sample (1.0 = use all)')
    parser.add_argument('--max_components', type=int, default=20,
                       help='Maximum number of components to test for BIC/AIC')
    parser.add_argument('--embeddings_dir', type=str, default=None,
                       help='Directory containing pre-saved embeddings (.pt files)')
    parser.add_argument('--use_pre_saved', action='store_true',
                       help='Use pre-saved embeddings instead of extracting from model')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = GNNGMMAnalyzer(
        model_type=args.model_type,
        checkpoint_index=args.checkpoint_index,
        output_dir=args.output_dir,
        sample_frac=args.sample_frac,
        embeddings_dir=args.embeddings_dir
    )
    
    # Run analysis
    analyzer.run_complete_analysis(use_pre_saved_embeddings=args.use_pre_saved)


def run_analysis_with_pre_saved_embeddings(embeddings_dir, output_dir=None, max_components=20, 
                                          model_type="dimenet-10conf", checkpoint_index=0, 
                                          sample_frac=1.0, experiment_name=None):
    """
    Convenience function to run GMM analysis with pre-saved embeddings
    
    Args:
        embeddings_dir (str): Path to directory containing .pt embedding files
        output_dir (str): Output directory for results (optional)
        max_components (int): Maximum number of components to test for BIC/AIC
        model_type (str): Type of model (for naming purposes)
        checkpoint_index (int): Checkpoint index (for naming purposes)
        sample_frac (float): Fraction of data to use (1.0 = use all)
        experiment_name (str): Custom name for this experiment (optional)
    
    Returns:
        GNNGMMAnalyzer: The analyzer object with results
    """
    if output_dir is None:
        # Create default output directory with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            output_dir = f"../GNNGMMAnalyzerResults/{experiment_name}_{timestamp}"
        else:
            output_dir = f"../GNNGMMAnalyzerResults/{model_type}_checkpoint{checkpoint_index}_{timestamp}"
    
    print(f"Starting GMM analysis with pre-saved embeddings...")
    print(f"Embeddings directory: {embeddings_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max components: {max_components}")
    
    # Create analyzer
    analyzer = GNNGMMAnalyzer(
        model_type=model_type,
        checkpoint_index=checkpoint_index,
        output_dir=output_dir,
        sample_frac=sample_frac,
        embeddings_dir=embeddings_dir
    )
    
    # Run analysis with pre-saved embeddings
    analyzer.run_complete_analysis(use_pre_saved_embeddings=True)
    
    return analyzer


if __name__ == "__main__":
    main()
