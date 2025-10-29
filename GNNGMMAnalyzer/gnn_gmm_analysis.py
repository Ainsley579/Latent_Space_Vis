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
    
    def find_optimal_components(self, embeddings, max_components=20, zca_threshold=1e-5):
        """Find optimal number of components using BIC and AIC with ZCA whitening"""
        print("Finding optimal number of components using BIC/AIC with ZCA whitening...")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply ZCA whitening
        print("  Applying ZCA whitening for component selection...")
        embeddings_whitened, _ = self.adaptive_zca_whitening(embeddings_scaled, threshold=zca_threshold)
        
        # Test different numbers of components
        n_components_range = range(2, max_components + 1)
        bic_scores = []
        aic_scores = []
        silhouette_scores = []
        
        for n_components in n_components_range:
            print(f"  Testing {n_components} components...")
            
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
    
    def fit_layer_gmms(self, n_components=None, zca_threshold=1e-5):
        """Fit GMM to embeddings from each layer with cluster alignment and ZCA whitening"""
        if n_components is None:
            n_components = self.optimal_components
        
        print(f"Fitting GMMs to all layers with {n_components} components...")
        
        # First, fit GMMs to all layers without alignment
        for layer_name, embeddings in self.layer_embeddings.items():
            print(f"  Fitting GMM for layer: {layer_name}")
            
            # Standardize embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # Apply ZCA whitening
            print(f"    Applying ZCA whitening...")
            embeddings_whitened, zca_transform = self.adaptive_zca_whitening(
                embeddings_scaled, threshold=zca_threshold
            )
            
            # Fit GMM on whitened embeddings
            gmm = GaussianMixture(
                n_components=n_components,
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
                'probabilities': gmm.predict_proba(embeddings_whitened)
            }
        
        # Now align cluster indices across layers (for visualization purposes only)
        print("Aligning cluster indices across layers...")
        self._align_cluster_indices()
        
        print(f"Fitted GMMs to {len(self.layer_embeddings)} layers with aligned cluster indices")
    
    def _align_cluster_indices(self):
        """Align cluster indices across layers using K-NN based approach"""
        layer_names = list(self.gmm_results.keys())
        n_components = self.optimal_components
        
        # Start with the first layer as reference
        reference_layer = layer_names[0]
        print(f"  Using {reference_layer} as reference layer")
        
        # For each subsequent layer, use K-NN to find best alignment
        for i in range(1, len(layer_names)):
            current_layer = layer_names[i]
            previous_layer = layer_names[i-1]
            
            print(f"  Aligning {current_layer} with {previous_layer} using K-NN approach")
            
            # Get component features (means) from both layers
            current_means = self.gmm_results[current_layer]['means']
            previous_means = self.gmm_results[previous_layer]['means']
            
            # Use K-NN to find best alignment
            best_permutation = self._find_best_alignment_with_knn(
                current_means, previous_means, n_components
            )
            
            # Apply the permutation to current layer
            self._apply_cluster_permutation(current_layer, best_permutation)
            
            print(f"    Applied permutation: {best_permutation}")
    
    def _find_best_alignment_with_knn(self, current_means, previous_means, n_components, k=1):
        """Find best component alignment using K-NN approach"""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA
        
        # Handle different dimensions by projecting to common space
        if current_means.shape[1] != previous_means.shape[1]:
            # Use PCA to project to common dimension
            min_dim = min(current_means.shape[1], previous_means.shape[1])
            # Ensure PCA components don't exceed number of samples
            min_dim = min(min_dim, current_means.shape[0], previous_means.shape[0])
            
            # For different dimensions, we need to handle them separately
            # Project each to the smaller dimension
            if current_means.shape[1] > previous_means.shape[1]:
                # Current has more dimensions, project it down
                pca_current = PCA(n_components=min_dim)
                current_means_proj = pca_current.fit_transform(current_means)
                # Also project previous means to the same dimension
                pca_previous = PCA(n_components=min_dim)
                previous_means_proj = pca_previous.fit_transform(previous_means)
            else:
                # Previous has more dimensions, project it down
                pca_previous = PCA(n_components=min_dim)
                previous_means_proj = pca_previous.fit_transform(previous_means)
                # Also project current means to the same dimension
                pca_current = PCA(n_components=min_dim)
                current_means_proj = pca_current.fit_transform(current_means)
        else:
            current_means_proj = current_means
            previous_means_proj = previous_means
        
        # Create K-NN model on previous layer means
        knn = NearestNeighbors(n_neighbors=min(k, n_components), algorithm='auto')
        knn.fit(previous_means_proj)
        
        # Find nearest neighbors for each current component
        distances, indices = knn.kneighbors(current_means_proj)
        
        # Create alignment mapping: current_component -> previous_component
        alignment_map = {}
        used_previous_indices = set()
        
        # First pass: exact matches (distance = 0)
        for curr_idx in range(n_components):
            if distances[curr_idx, 0] < 1e-10:  # Essentially identical
                prev_idx = indices[curr_idx, 0]
                if prev_idx not in used_previous_indices:
                    alignment_map[curr_idx] = prev_idx
                    used_previous_indices.add(prev_idx)
        
        # Second pass: remaining components using closest available
        for curr_idx in range(n_components):
            if curr_idx not in alignment_map:
                for neighbor_idx in indices[curr_idx]:
                    if neighbor_idx not in used_previous_indices:
                        alignment_map[curr_idx] = neighbor_idx
                        used_previous_indices.add(neighbor_idx)
                        break
        
        # Create permutation: new_index -> original_index
        # We want current_component to be mapped to the position of previous_component
        permutation = list(range(n_components))
        for curr_idx, prev_idx in alignment_map.items():
            permutation[prev_idx] = curr_idx
        
        # Validate and fix any missing mappings
        unused_indices = set(range(n_components)) - set(alignment_map.values())
        unused_current = set(range(n_components)) - set(alignment_map.keys())
        
        for curr_idx, prev_idx in zip(sorted(unused_current), sorted(unused_indices)):
            permutation[prev_idx] = curr_idx
        
        return permutation
    
    def _find_optimal_alignment_with_knn(self, transition_matrix, current_means, previous_means, n_components, k=1):
        """Find optimal alignment for transition matrix using K-NN approach"""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA
        
        # Handle different dimensions by projecting to common space
        if current_means.shape[1] != previous_means.shape[1]:
            # Use PCA to project to common dimension
            min_dim = min(current_means.shape[1], previous_means.shape[1])
            # Ensure PCA components don't exceed number of samples
            min_dim = min(min_dim, current_means.shape[0], previous_means.shape[0])
            
            # For different dimensions, we need to handle them separately
            # Project each to the smaller dimension
            if current_means.shape[1] > previous_means.shape[1]:
                # Current has more dimensions, project it down
                pca_current = PCA(n_components=min_dim)
                current_means_proj = pca_current.fit_transform(current_means)
                # Also project previous means to the same dimension
                pca_previous = PCA(n_components=min_dim)
                previous_means_proj = pca_previous.fit_transform(previous_means)
            else:
                # Previous has more dimensions, project it down
                pca_previous = PCA(n_components=min_dim)
                previous_means_proj = pca_previous.fit_transform(previous_means)
                # Also project current means to the same dimension
                pca_current = PCA(n_components=min_dim)
                current_means_proj = pca_current.fit_transform(current_means)
        else:
            current_means_proj = current_means
            previous_means_proj = previous_means
        
        # Create K-NN model on previous layer means
        knn = NearestNeighbors(n_neighbors=min(k, n_components), algorithm='auto')
        knn.fit(previous_means_proj)
        
        # Find nearest neighbors for each current component
        distances, indices = knn.kneighbors(current_means_proj)
        
        # Create alignment mapping based on nearest neighbors
        alignment_map = {}
        used_previous_indices = set()
        
        # First pass: exact matches
        for curr_idx in range(n_components):
            if distances[curr_idx, 0] < 1e-10:
                prev_idx = indices[curr_idx, 0]
                if prev_idx not in used_previous_indices:
                    alignment_map[curr_idx] = prev_idx
                    used_previous_indices.add(prev_idx)
        
        # Second pass: remaining components
        for curr_idx in range(n_components):
            if curr_idx not in alignment_map:
                for neighbor_idx in indices[curr_idx]:
                    if neighbor_idx not in used_previous_indices:
                        alignment_map[curr_idx] = neighbor_idx
                        used_previous_indices.add(neighbor_idx)
                        break
        
        # Create permutation
        permutation = list(range(n_components))
        for curr_idx, prev_idx in alignment_map.items():
            permutation[prev_idx] = curr_idx
        
        # Fill any remaining gaps
        unused_indices = set(range(n_components)) - set(alignment_map.values())
        unused_current = set(range(n_components)) - set(alignment_map.keys())
        
        for curr_idx, prev_idx in zip(sorted(unused_current), sorted(unused_indices)):
            permutation[prev_idx] = curr_idx
        
        return permutation
    
    def _find_best_cluster_permutation(self, transition_matrix):
        """Find the permutation of cluster indices that makes diagonal elements the maximum for each row"""
        from itertools import permutations
        
        n_components = transition_matrix.shape[0]
        best_permutation = None
        best_score = -1
        
        # Try all possible permutations of columns (target clusters)
        for perm in permutations(range(n_components)):
            # Check if diagonal elements are the maximum for each row
            diagonal_elements = []
            row_maxima = []
            
            for i in range(n_components):
                diagonal_val = transition_matrix[i, perm[i]]
                row_max = np.max(transition_matrix[i, :])
                diagonal_elements.append(diagonal_val)
                row_maxima.append(row_max)
            
            # Score: number of rows where diagonal is the maximum
            score = sum(1 for i in range(n_components) if diagonal_elements[i] == row_maxima[i])
            
            if score > best_score:
                best_score = score
                best_permutation = list(perm)
            elif score == best_score:
                # If tied, prefer the one with higher diagonal sum
                diagonal_sum = sum(diagonal_elements)
                current_diagonal_sum = sum(transition_matrix[i, best_permutation[i]] for i in range(n_components))
                if diagonal_sum > current_diagonal_sum:
                    best_permutation = list(perm)
        
        print(f"      Diagonal sum before alignment: {np.trace(transition_matrix):.3f}")
        permuted_matrix = transition_matrix[:, best_permutation]
        print(f"      Diagonal sum after alignment: {np.trace(permuted_matrix):.3f}")
        print(f"      Rows with diagonal as maximum: {best_score}/{n_components}")
        
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
            
            # Plot GMM ellipses (skip for now due to dimension issues with ZCA whitening)
            # self._plot_gmm_ellipses(ax, means_2d, gmm_result['covariances'], 
            #                       gmm_result['scaler'], pca, gmm_result['zca_transform'])
            
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
    
    def _plot_gmm_ellipses(self, ax, means_2d, covariances, scaler, pca, zca_transform):
        """Plot GMM ellipses on 2D projection"""
        for i, cov in enumerate(covariances):
            mean_2d = means_2d[i]
            
            # Transform covariance from whitened space back to original space, then to 2D
            # First, transform covariance back from whitened space to scaled space
            cov_scaled = zca_transform @ cov @ zca_transform.T
            # Apply inverse scaling transformation manually for covariance matrix
            # For covariance: if X_scaled = (X - mean) / scale, then Cov(X) = scale * Cov(X_scaled) * scale
            scale_matrix = np.diag(scaler.scale_)
            cov_original = scale_matrix @ cov_scaled @ scale_matrix
            # Finally, transform to 2D using PCA
            cov_2d = pca.components_ @ cov_original @ pca.components_.T
            
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
    
    def analyze_component_persistence(self):
        """Analyze component persistence across layers based on majority overlap"""
        print("Analyzing component persistence across layers...")
        
        layer_names = list(self.gmm_results.keys())
        self.component_persistence = {}
        self.point_transitions = {}
        
        # Track transitions between consecutive layers
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i+1]
            
            print(f"  Analyzing persistence: {current_layer} -> {next_layer}")
            
            # Get cluster assignments
            current_labels = self.gmm_results[current_layer]['labels']
            next_labels = self.gmm_results[next_layer]['labels']
            
            n_components = self.optimal_components
            
            # Create transition matrix
            transition_matrix = np.zeros((n_components, n_components))
            for prev_cluster in range(n_components):
                prev_mask = current_labels == prev_cluster
                if np.sum(prev_mask) > 0:
                    next_clusters = next_labels[prev_mask]
                    for next_cluster in range(n_components):
                        transition_matrix[prev_cluster, next_cluster] = np.sum(next_clusters == next_cluster)
            
            # Normalize by row sums to get transition probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_probs = transition_matrix / (row_sums + 1e-10)
            
            # Find persistent components (majority overlap > threshold)
            persistence_threshold = 0.5  # 50% majority threshold
            persistence_map = {}
            persistence_scores = {}
            
            for prev_cluster in range(n_components):
                # Find the next cluster with maximum overlap
                max_overlap_idx = np.argmax(transition_probs[prev_cluster])
                max_overlap_prob = transition_probs[prev_cluster, max_overlap_idx]
                
                persistence_scores[prev_cluster] = {
                    'max_overlap_prob': max_overlap_prob,
                    'target_cluster': max_overlap_idx,
                    'is_persistent': max_overlap_prob > persistence_threshold,
                    'transition_probs': transition_probs[prev_cluster].copy()
                }
                
                if max_overlap_prob > persistence_threshold:
                    persistence_map[prev_cluster] = max_overlap_idx
                else:
                    persistence_map[prev_cluster] = None  # No clear persistence
            
            # Store persistence analysis
            transition_key = f"{current_layer}_to_{next_layer}"
            self.component_persistence[transition_key] = {
                'transition_matrix': transition_matrix,
                'transition_probs': transition_probs,
                'persistence_map': persistence_map,
                'persistence_scores': persistence_scores,
                'threshold': persistence_threshold
            }
            
            # Track individual point transitions
            self.point_transitions[transition_key] = {
                'current_labels': current_labels,
                'next_labels': next_labels,
                'n_points': len(current_labels)
            }
            
            # Print persistence summary
            persistent_count = sum(1 for v in persistence_map.values() if v is not None)
            print(f"    {persistent_count}/{n_components} components show persistence (>{persistence_threshold*100}%)")
            
            for prev_cluster, persistence_data in persistence_scores.items():
                status = "PERSISTENT" if persistence_data['is_persistent'] else "CHANGED"
                target = persistence_data['target_cluster'] if persistence_data['is_persistent'] else "N/A"
                prob = persistence_data['max_overlap_prob']
                print(f"      Component {prev_cluster}: {status} -> {target} ({prob:.1%})")
        
        print("Component persistence analysis completed")
    
    def create_persistence_visualizations(self):
        """Create visualizations for component persistence"""
        print("Creating component persistence visualizations...")
        
        layer_names = list(self.gmm_results.keys())
        
        # Create persistence summary plot
        self._create_persistence_summary_plot()
        
        # Create detailed persistence plots for each layer transition
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i+1]
            transition_key = f"{current_layer}_to_{next_layer}"
            
            if transition_key in self.component_persistence:
                self._create_layer_persistence_plot(current_layer, next_layer, transition_key)
        
        print("Persistence visualizations completed")
    
    def _create_persistence_summary_plot(self):
        """Create a summary plot showing persistence across all layer transitions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        layer_names = list(self.gmm_results.keys())
        n_transitions = len(layer_names) - 1
        transitions = []
        persistence_counts = []
        avg_persistence_scores = []
        component_stability = {i: [] for i in range(self.optimal_components)}
        
        for i in range(n_transitions):
            current_layer = layer_names[i]
            next_layer = layer_names[i+1]
            transition_key = f"{current_layer}_to_{next_layer}"
            
            if transition_key in self.component_persistence:
                persistence_data = self.component_persistence[transition_key]
                persistent_count = sum(1 for v in persistence_data['persistence_map'].values() if v is not None)
                avg_score = np.mean([data['max_overlap_prob'] for data in persistence_data['persistence_scores'].values()])
                
                transitions.append(f"{current_layer}\n→ {next_layer}")
                persistence_counts.append(persistent_count)
                avg_persistence_scores.append(avg_score)
                
                # Track stability of each component
                for comp_idx in range(self.optimal_components):
                    component_stability[comp_idx].append(
                        persistence_data['persistence_scores'][comp_idx]['max_overlap_prob']
                    )
        
        # Plot 1: Number of persistent components per transition
        axes[0, 0].bar(range(n_transitions), persistence_counts, color='skyblue', alpha=0.7)
        axes[0, 0].set_xticks(range(n_transitions))
        axes[0, 0].set_xticklabels(transitions, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Number of Persistent Components')
        axes[0, 0].set_title('Component Persistence Count per Transition')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Average persistence score per transition
        axes[0, 1].bar(range(n_transitions), avg_persistence_scores, color='lightgreen', alpha=0.7)
        axes[0, 1].set_xticks(range(n_transitions))
        axes[0, 1].set_xticklabels(transitions, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Average Persistence Score')
        axes[0, 1].set_title('Average Component Persistence Score per Transition')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Component stability over transitions
        colors = plt.cm.tab10(np.linspace(0, 1, self.optimal_components))
        for comp_idx in range(self.optimal_components):
            axes[1, 0].plot(range(n_transitions), component_stability[comp_idx], 
                           'o-', color=colors[comp_idx], label=f'Component {comp_idx}', 
                           linewidth=2, markersize=6)
        axes[1, 0].set_xticks(range(n_transitions))
        axes[1, 0].set_xticklabels(transitions, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Persistence Score')
        axes[1, 0].set_title('Individual Component Stability Across Transitions')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Persistence distribution
        all_scores = []
        for transition_key, persistence_data in self.component_persistence.items():
            scores = [data['max_overlap_prob'] for data in persistence_data['persistence_scores'].values()]
            all_scores.extend(scores)
        
        axes[1, 1].hist(all_scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Persistence Threshold')
        axes[1, 1].set_xlabel('Persistence Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Component Persistence Scores')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_persistence_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_layer_persistence_plot(self, current_layer, next_layer, transition_key):
        """Create detailed persistence plot for a specific layer transition"""
        persistence_data = self.component_persistence[transition_key]
        transition_probs = persistence_data['transition_probs']
        persistence_scores = persistence_data['persistence_scores']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Transition probability matrix
        im1 = axes[0, 0].imshow(transition_probs, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        axes[0, 0].set_xlabel(f'{next_layer} Component')
        axes[0, 0].set_ylabel(f'{current_layer} Component')
        axes[0, 0].set_title(f'Transition Probabilities: {current_layer} → {next_layer}')
        
        # Add text annotations
        for i in range(transition_probs.shape[0]):
            for j in range(transition_probs.shape[1]):
                text = axes[0, 0].text(j, i, f'{transition_probs[i, j]:.2f}',
                                     ha="center", va="center", 
                                     color="white" if transition_probs[i, j] > 0.5 else "black")
        
        plt.colorbar(im1, ax=axes[0, 0], label='Transition Probability')
        
        # Plot 2: Persistence scores
        components = list(range(self.optimal_components))
        max_overlap_probs = [persistence_scores[i]['max_overlap_prob'] for i in components]
        is_persistent = [persistence_scores[i]['is_persistent'] for i in components]
        
        colors = ['green' if persistent else 'red' for persistent in is_persistent]
        bars = axes[0, 1].bar(components, max_overlap_probs, color=colors, alpha=0.7)
        axes[0, 1].axhline(0.5, color='red', linestyle='--', linewidth=2, label='Persistence Threshold')
        axes[0, 1].set_xlabel('Component')
        axes[0, 1].set_ylabel('Max Overlap Probability')
        axes[0, 1].set_title(f'Component Persistence Scores: {current_layer} → {next_layer}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, prob in zip(bars, max_overlap_probs):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{prob:.2f}', ha='center', va='bottom')
        
        # Plot 3: Component mapping visualization
        mapping_data = []
        mapping_colors = []
        for prev_comp, persistence_info in persistence_scores.items():
            if persistence_info['is_persistent']:
                mapping_data.append((prev_comp, persistence_info['target_cluster'], 
                                   persistence_info['max_overlap_prob']))
                mapping_colors.append('green')
            else:
                mapping_data.append((prev_comp, -1, persistence_info['max_overlap_prob']))
                mapping_colors.append('red')
        
        if mapping_data:
            prev_comps, next_comps, probs = zip(*mapping_data)
            axes[1, 0].scatter(prev_comps, next_comps, c=mapping_colors, s=[p*200 for p in probs], 
                             alpha=0.7, edgecolors='black')
            
            # Add arrows for persistent mappings
            for prev_comp, next_comp, prob in mapping_data:
                if next_comp >= 0:  # Persistent component
                    axes[1, 0].annotate('', xy=(next_comp, prev_comp), xytext=(prev_comp, prev_comp),
                                       arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
        
        axes[1, 0].set_xlabel(f'{current_layer} Component')
        axes[1, 0].set_ylabel(f'{next_layer} Component')
        axes[1, 0].set_title(f'Component Mapping: {current_layer} → {next_layer}')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(-0.5, self.optimal_components - 0.5)
        axes[1, 0].set_ylim(-0.5, self.optimal_components - 0.5)
        
        # Plot 4: Persistence statistics
        persistent_count = sum(1 for info in persistence_scores.values() if info['is_persistent'])
        changed_count = len(persistence_scores) - persistent_count
        avg_persistence = np.mean(max_overlap_probs)
        
        stats_text = f"""Persistence Statistics:
        
Persistent Components: {persistent_count}/{len(persistence_scores)}
Changed Components: {changed_count}/{len(persistence_scores)}
Average Persistence: {avg_persistence:.3f}

Component Details:"""
        
        for comp_idx, info in persistence_scores.items():
            status = "PERSISTENT" if info['is_persistent'] else "CHANGED"
            target = info['target_cluster'] if info['is_persistent'] else "N/A"
            stats_text += f"\n  Component {comp_idx}: {status} → {target} ({info['max_overlap_prob']:.1%})"
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'persistence_{current_layer}_to_{next_layer}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
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
            
            # Use K-NN approach for better alignment
            optimal_permutation = self._find_optimal_alignment_with_knn(
                transition_matrix_norm, self.gmm_results[current_layer]['means'], 
                self.gmm_results[previous_layer]['means'], self.optimal_components
            )
            
            # Apply permutation to columns (relabel target clusters)
            transition_matrix_norm_aligned = transition_matrix_norm[:, optimal_permutation]
            transition_matrix_aligned = transition_matrix[:, optimal_permutation]
            
            # Export as long format (using aligned matrices)
            # The aligned matrices already have the columns permuted, so we need to map back to original column indices
            for prev_cluster in range(n_components):
                for curr_cluster in range(n_components):
                    # curr_cluster is the position in the permuted matrix
                    # optimal_permutation[curr_cluster] is the original column index
                    original_col_idx = optimal_permutation[curr_cluster]
                    transition_data.append({
                        'from_layer': previous_layer,
                        'to_layer': current_layer,
                        'from_cluster': prev_cluster,
                        'to_cluster': curr_cluster,  # Use the new (permuted) column index
                        'count': transition_matrix_aligned[prev_cluster, curr_cluster],
                        'probability': transition_matrix_norm_aligned[prev_cluster, curr_cluster]
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
        
        # 6. Export component persistence data
        persistence_data = []
        for transition_key, persistence_info in self.component_persistence.items():
            for comp_idx, scores in persistence_info['persistence_scores'].items():
                persistence_data.append({
                    'transition': transition_key,
                    'component': comp_idx,
                    'max_overlap_prob': scores['max_overlap_prob'],
                    'target_cluster': scores['target_cluster'],
                    'is_persistent': scores['is_persistent'],
                    'threshold': persistence_info['threshold']
                })
        
        df_persistence = pd.DataFrame(persistence_data)
        df_persistence.to_csv(self.output_dir / 'component_persistence_analysis.csv', index=False)
        
        print("CSV export completed:")
        print("  - cluster_assignments_and_metrics.csv")
        print("  - component_sharpening_metrics.csv") 
        print("  - cluster_transition_matrices.csv")
        print("  - boundary_molecule_analysis.csv")
        print("  - component_persistence_analysis.csv")
        print("  - analysis_summary_metrics.csv")
    
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
        
        # Create continuity and fragmentation summary
        self._create_continuity_fragmentation_summary(df_sample, layer_names)
        
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
            
            # Use K-NN approach for better alignment
            optimal_permutation = self._find_optimal_alignment_with_knn(
                transition_matrix.values, self.gmm_results[layer_names[i+1]]['means'], 
                self.gmm_results[layer_names[i]]['means'], self.optimal_components
            )
            
            # Apply permutation to column labels only (for visualization)
            permuted_columns = transition_matrix.columns[optimal_permutation]
            transition_matrix_aligned = transition_matrix[permuted_columns]
            
            transition_matrices.append({
                'from_layer': layer_names[i],
                'to_layer': layer_names[i+1],
                'matrix': transition_matrix_aligned
            })
        
        # Create visualization with 2 rows, 4 columns each
        n_transitions = len(transition_matrices)
        n_rows = 2
        n_cols = 4
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for i, trans_data in enumerate(transition_matrices):
            ax = axes_flat[i]
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
        
        # Hide unused subplots
        for i in range(n_transitions, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_transition_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create cluster continuity and fragmentation visualization
        self._create_cluster_continuity_visualization(df_sample, layer_names)
        
        # Create cluster stability plot
        self._create_cluster_stability_plot(df_sample, layer_names)
        
        print("Cluster transition analysis completed")
    
    def _create_cluster_continuity_visualization(self, df_sample, layer_names):
        """Create visualization showing cluster continuity and fragmentation across layers"""
        print("Creating cluster continuity and fragmentation visualization...")
        
        n_components = self.optimal_components
        n_transitions = len(layer_names) - 1
        
        # Create a horizontal visualization showing flow of data points
        fig, axes = plt.subplots(1, n_transitions, figsize=(6 * n_transitions, 12))
        
        if n_transitions == 1:
            axes = [axes]
        
        for i in range(n_transitions):
            ax = axes[i]
            current_layer = f'{layer_names[i]}_cluster'
            next_layer = f'{layer_names[i+1]}_cluster'
            
            # Calculate transition counts
            transition_counts = pd.crosstab(
                df_sample[current_layer],
                df_sample[next_layer]
            )
            
            # For each component in current layer, calculate:
            # - Total points in the component
            # - Points that stay together (go to same component in next layer)
            # - Points that fragment (go to different components)
            
            continuity_data = []
            for src_component in range(n_components):
                total_points = transition_counts.loc[src_component].sum() if src_component in transition_counts.index else 0
                
                if total_points == 0:
                    continue
                
                # Find where points go
                destinations = {}
                for dst_component in range(n_components):
                    if dst_component in transition_counts.columns and src_component in transition_counts.index:
                        count = transition_counts.loc[src_component, dst_component]
                        if count > 0:
                            destinations[dst_component] = count
                
                # Find the majority destination
                if destinations:
                    majority_dest = max(destinations, key=destinations.get)
                    majority_count = destinations[majority_dest]
                    majority_pct = (majority_count / total_points) * 100
                    fragmented_count = total_points - majority_count
                    fragmented_pct = (fragmented_count / total_points) * 100
                    
                    continuity_data.append({
                        'component': src_component,
                        'total': total_points,
                        'majority_dest': majority_dest,
                        'stayed_together': majority_count,
                        'stayed_together_pct': majority_pct,
                        'fragmented': fragmented_count,
                        'fragmented_pct': fragmented_pct,
                        'destinations': destinations
                    })
            
            # Create horizontal stacked bar chart
            y_pos = np.arange(len(continuity_data))
            stayed = [d['stayed_together'] for d in continuity_data]
            fragmented = [d['fragmented'] for d in continuity_data]
            
            bars1 = ax.barh(y_pos, stayed, label='Stayed Together', color='#2ecc71', alpha=0.8)
            bars2 = ax.barh(y_pos, fragmented, left=stayed, label='Fragmented', color='#e74c3c', alpha=0.8)
            
            # Add labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f'Component {d["component"]}' for d in continuity_data])
            ax.set_xlabel('Number of Data Points')
            ax.set_title(f'{layer_names[i]} → {layer_names[i+1]}', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(axis='x', alpha=0.3)
            
            # Add detailed annotations
            for idx, d in enumerate(continuity_data):
                # Stayed together annotation
                if d['stayed_together'] > 0:
                    ax.text(d['stayed_together']/2, idx, 
                           f"{d['stayed_together_pct']:.1f}%\n→C{d['majority_dest']}", 
                           ha='center', va='center', fontsize=8, fontweight='bold', color='white')
                
                # Fragmented annotation with details
                if d['fragmented'] > 0:
                    # Create detailed fragmentation text
                    frag_details = []
                    for dest, count in d['destinations'].items():
                        if dest != d['majority_dest'] and count > 0:
                            frag_details.append(f"C{dest}:{count}")
                    
                    if frag_details:
                        frag_text = f"{d['fragmented_pct']:.1f}%\n" + "\n".join(frag_details[:3])  # Limit to 3 destinations
                        if len(frag_details) > 3:
                            frag_text += "\n..."
                    else:
                        frag_text = f"{d['fragmented_pct']:.1f}%\nsplit"
                    
                    ax.text(d['stayed_together'] + d['fragmented']/2, idx,
                           frag_text,
                           ha='center', va='center', fontsize=7, style='italic', color='white')
                
                # Total count annotation
                ax.text(d['total'] + 2, idx, f"n={d['total']}", 
                       ha='left', va='center', fontsize=8, fontweight='bold')
            
            # Add a text box with summary statistics
            avg_continuity = np.mean([d['stayed_together_pct'] for d in continuity_data])
            textstr = f'Average Continuity: {avg_continuity:.1f}%'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_continuity_fragmentation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Cluster continuity and fragmentation visualization saved")
    
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
    
    def _create_continuity_fragmentation_summary(self, df_sample, layer_names):
        """Create summary statistics of cluster continuity and fragmentation"""
        print("Creating continuity and fragmentation summary...")
        
        n_transitions = len(layer_names) - 1
        
        # Calculate continuity and fragmentation metrics for each transition
        continuity_data = []
        fragmentation_data = []
        avg_continuities = []
        avg_fragmentations = []
        
        for i in range(n_transitions):
            current_layer = f'{layer_names[i]}_cluster'
            next_layer = f'{layer_names[i+1]}_cluster'
            
            # Calculate transition counts
            transition_counts = pd.crosstab(
                df_sample[current_layer],
                df_sample[next_layer]
            )
            
            # Calculate continuity and fragmentation for each component
            component_continuities = []
            component_fragmentations = []
            
            for src_component in range(self.optimal_components):
                total_points = transition_counts.loc[src_component].sum() if src_component in transition_counts.index else 0
                
                if total_points == 0:
                    continue
                
                # Find majority destination
                destinations = {}
                for dst_component in range(self.optimal_components):
                    if dst_component in transition_counts.columns and src_component in transition_counts.index:
                        count = transition_counts.loc[src_component, dst_component]
                        if count > 0:
                            destinations[dst_component] = count
                
                if destinations:
                    majority_dest = max(destinations, key=destinations.get)
                    majority_count = destinations[majority_dest]
                    continuity = (majority_count / total_points) * 100
                    fragmentation = ((total_points - majority_count) / total_points) * 100
                    
                    component_continuities.append(continuity)
                    component_fragmentations.append(fragmentation)
            
            avg_continuities.append(np.mean(component_continuities))
            avg_fragmentations.append(np.mean(component_fragmentations))
            continuity_data.append(component_continuities)
            fragmentation_data.append(component_fragmentations)
        
        # Create summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Average continuity per transition
        transitions = [f'{layer_names[i]} → {layer_names[i+1]}' for i in range(n_transitions)]
        bars1 = ax1.bar(range(len(transitions)), avg_continuities, color='#2ecc71', alpha=0.8)
        ax1.set_xticks(range(len(transitions)))
        ax1.set_xticklabels(transitions, rotation=45, ha='right')
        ax1.set_ylabel('Average Continuity (%)')
        ax1.set_title('Average Cluster Continuity per Layer Transition')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for i, (bar, continuity) in enumerate(zip(bars1, avg_continuities)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{continuity:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Average fragmentation per transition
        bars2 = ax2.bar(range(len(transitions)), avg_fragmentations, color='#e74c3c', alpha=0.8)
        ax2.set_xticks(range(len(transitions)))
        ax2.set_xticklabels(transitions, rotation=45, ha='right')
        ax2.set_ylabel('Average Fragmentation (%)')
        ax2.set_title('Average Cluster Fragmentation per Layer Transition')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for i, (bar, fragmentation) in enumerate(zip(bars2, avg_fragmentations)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{fragmentation:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Continuity vs Fragmentation scatter
        ax3.scatter(avg_continuities, avg_fragmentations, s=100, alpha=0.7, c=range(len(transitions)), 
                   cmap='viridis', edgecolors='black', linewidth=2)
        ax3.set_xlabel('Average Continuity (%)')
        ax3.set_ylabel('Average Fragmentation (%)')
        ax3.set_title('Continuity vs Fragmentation by Transition')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 100)
        ax3.set_ylim(0, 100)
        
        # Add diagonal line (continuity + fragmentation = 100%)
        ax3.plot([0, 100], [100, 0], 'r--', alpha=0.5, linewidth=2, label='Perfect balance')
        
        # Add transition labels
        for i, (cont, frag) in enumerate(zip(avg_continuities, avg_fragmentations)):
            ax3.annotate(f'{i+1}', (cont, frag), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold')
        
        ax3.legend()
        
        # Overall summary statistics
        overall_continuity = np.mean(avg_continuities)
        overall_fragmentation = np.mean(avg_fragmentations)
        
        # Create pie chart showing overall balance
        stability_categories = []
        stability_counts = []
        colors = []
        
        for i, (cont, frag) in enumerate(zip(avg_continuities, avg_fragmentations)):
            if cont >= 70:
                stability_categories.append('High Continuity\n(≥70%)')
                stability_counts.append(1)
                colors.append('#2ecc71')
            elif cont >= 50:
                stability_categories.append('Medium Continuity\n(50-70%)')
                stability_counts.append(1)
                colors.append('#f39c12')
            else:
                stability_categories.append('Low Continuity\n(<50%)')
                stability_counts.append(1)
                colors.append('#e74c3c')
        
        # Count categories
        from collections import Counter
        category_counts = Counter(stability_categories)
        
        if category_counts:
            ax4.pie(category_counts.values(), labels=category_counts.keys(), 
                   colors=['#2ecc71', '#f39c12', '#e74c3c'], autopct='%1.1f%%', startangle=90)
            ax4.set_title('Overall Cluster Stability Distribution')
        
        # Add summary text box
        summary_text = f'Overall Statistics:\n'
        summary_text += f'Average Continuity: {overall_continuity:.1f}%\n'
        summary_text += f'Average Fragmentation: {overall_fragmentation:.1f}%\n'
        summary_text += f'Total Transitions: {n_transitions}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left', bbox=props)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'continuity_fragmentation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Continuity and fragmentation summary saved")
    
    def create_sankey_diagram(self):
        """Create Sankey diagram showing component transitions across layers"""
        if not PLOTLY_AVAILABLE:
            print("Skipping Sankey diagram: plotly not available")
            return
        
        print("Creating Sankey diagram for component transitions...")
        
        layer_names = list(self.gmm_results.keys())
        n_components = self.optimal_components
        
        # Prepare data for Sankey diagram
        source_nodes = []
        target_nodes = []
        values = []
        labels = []
        
        # Create node labels for each layer
        node_id = 0
        layer_node_mapping = {}
        
        for layer_idx, layer_name in enumerate(layer_names):
            layer_node_mapping[layer_name] = {}
            for comp_idx in range(n_components):
                node_label = f"{layer_name}\nComponent {comp_idx}"
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
            
            # Calculate transition counts
            for src_comp in range(n_components):
                src_mask = current_labels == src_comp
                if np.sum(src_mask) > 0:
                    next_clusters = next_labels[src_mask]
                    for dst_comp in range(n_components):
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
            title_text=f"Component Transitions Across GNN Layers<br><sub>Each flow represents data points moving between components</sub>",
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
        self._create_simplified_sankey_diagram(layer_names, n_components, layer_node_mapping)
    
    def _create_simplified_sankey_diagram(self, layer_names, n_components, layer_node_mapping):
        """Create a simplified Sankey diagram showing only major transitions (>10% of points)"""
        print("Creating simplified Sankey diagram (major transitions only)...")
        
        source_nodes = []
        target_nodes = []
        values = []
        labels = []
        
        # Create node labels
        node_id = 0
        for layer_idx, layer_name in enumerate(layer_names):
            for comp_idx in range(n_components):
                node_label = f"{layer_name}\nC{comp_idx}"
                labels.append(node_label)
                node_id += 1
        
        # Calculate major transitions only (>10% threshold)
        threshold = 0.1  # 10% threshold
        
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i+1]
            
            current_labels = self.gmm_results[current_layer]['labels']
            next_labels = self.gmm_results[next_layer]['labels']
            
            for src_comp in range(n_components):
                src_mask = current_labels == src_comp
                src_count = np.sum(src_mask)
                if src_count > 0:
                    next_clusters = next_labels[src_mask]
                    for dst_comp in range(n_components):
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
            title_text=f"Major Component Transitions (>10% threshold)<br><sub>Showing only significant flows between components</sub>",
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
    
    def run_complete_analysis(self, use_pre_saved_embeddings=False, zca_threshold=1e-5):
        """Run the complete GMM analysis pipeline with ZCA whitening"""
        print("Starting complete GNN GMM analysis with ZCA whitening...")
        start_time = time.time()
        
        if use_pre_saved_embeddings:
            # Step 1: Load pre-saved embeddings
            self.load_pre_saved_embeddings()
        else:
            # Step 1: Load model and data
            self.load_model_and_data()
            
            # Step 2: Extract layer embeddings
            self.extract_layer_embeddings()
        
        # Step 3: Find optimal components using final layer with ZCA whitening
        final_layer = list(self.layer_embeddings.keys())[-1]
        self.find_optimal_components(self.layer_embeddings[final_layer], zca_threshold=zca_threshold)
        
        # Step 4: Fit GMMs to all layers with ZCA whitening
        self.fit_layer_gmms(zca_threshold=zca_threshold)
        
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
        
        # Step 9.5: Create Sankey diagram for component transitions
        self.create_sankey_diagram()
        
        # Step 10: Analyze component persistence across layers
        self.analyze_component_persistence()
        
        # Step 11: Create persistence visualizations
        self.create_persistence_visualizations()
        
        # Step 12: Export numerical data to CSV files
        self.export_numerical_data_to_csv()
        
        # Step 13: Generate summary report
        self._generate_summary_report()
        
        total_time = time.time() - start_time
        print(f"\nComplete analysis finished in {total_time:.2f} seconds")
        print(f"Results saved in: {self.output_dir}")
    
    def run_layer_specific_component_analysis(self, max_components=20):
        """Run analysis where each layer can have different optimal number of components"""
        print("Starting layer-specific component analysis...")
        start_time = time.time()
        
        # Step 1: Find optimal components for each layer separately
        layer_optimal_components = {}
        
        print("Finding optimal components for each layer separately...")
        for layer_name, embeddings in self.layer_embeddings.items():
            print(f"  Analyzing layer: {layer_name}")
            optimal_components = self._find_optimal_components_for_layer(embeddings, max_components)
            layer_optimal_components[layer_name] = optimal_components
            print(f"    Optimal components: {optimal_components}")
        
        # Step 2: Fit GMMs with layer-specific components
        print("Fitting GMMs with layer-specific optimal components...")
        layer_gmm_results = {}
        
        for layer_name, embeddings in self.layer_embeddings.items():
            print(f"  Fitting GMM for layer: {layer_name} (n_components={layer_optimal_components[layer_name]})")
            
            # Standardize embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # Apply ZCA whitening
            print(f"    Applying ZCA whitening...")
            embeddings_whitened, zca_transform = self.adaptive_zca_whitening(
                embeddings_scaled, threshold=1e-5
            )
            
            # Fit GMM with layer-specific components on whitened embeddings
            gmm = GaussianMixture(
                n_components=layer_optimal_components[layer_name],
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
            gmm.fit(embeddings_whitened)
            
            # Store results
            layer_gmm_results[layer_name] = {
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
                'n_components': layer_optimal_components[layer_name]
            }
        
        # Step 3: Create visualization of component count evolution
        self._create_component_count_evolution_plot(layer_optimal_components)
        
        # Step 4: Create detailed component analysis
        self._create_layer_specific_component_analysis(layer_gmm_results, layer_optimal_components)
        
        total_time = time.time() - start_time
        print(f"Layer-specific component analysis finished in {total_time:.2f} seconds")
        print(f"Results saved in: {self.output_dir}")
        
        return layer_gmm_results, layer_optimal_components
    
    def _find_optimal_components_for_layer(self, embeddings, max_components):
        """Find optimal number of components for a specific layer with ZCA whitening"""
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply ZCA whitening
        embeddings_whitened, _ = self.adaptive_zca_whitening(embeddings_scaled, threshold=1e-5)
        
        # Test different numbers of components
        n_components_range = range(2, max_components + 1)
        bic_scores = []
        aic_scores = []
        silhouette_scores = []
        
        for n_components in n_components_range:
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
        
        # Use AIC as primary criterion (often more conservative)
        return optimal_aic
    
    def _create_component_count_evolution_plot(self, layer_optimal_components):
        """Create visualization showing how component count changes across layers"""
        print("Creating component count evolution plot...")
        
        layer_names = list(layer_optimal_components.keys())
        component_counts = list(layer_optimal_components.values())
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top plot: Component count evolution
        x_pos = np.arange(len(layer_names))
        bars = ax1.bar(x_pos, component_counts, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1.5)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(layer_names, rotation=45, ha='right')
        ax1.set_ylabel('Number of Components')
        ax1.set_title('Evolution of Optimal Component Count Across Layers', fontsize=14, fontweight='bold')
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
    
    def _create_layer_specific_component_analysis(self, layer_gmm_results, layer_optimal_components):
        """Create detailed analysis of layer-specific components"""
        print("Creating layer-specific component analysis...")
        
        # Create summary statistics
        summary_data = []
        
        for layer_name in layer_gmm_results.keys():
            gmm_result = layer_gmm_results[layer_name]
            n_components = layer_optimal_components[layer_name]
            
            # Calculate component statistics
            labels = gmm_result['labels']
            component_sizes = [np.sum(labels == i) for i in range(n_components)]
            
            # Calculate component sharpening metrics
            covariances = gmm_result['covariances']
            log_determinants = []
            traces = []
            
            for cov in covariances:
                sign, logabsdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    logabsdet = -1e12
                log_determinants.append(logabsdet)
                traces.append(np.trace(cov))
            
            summary_data.append({
                'layer': layer_name,
                'n_components': n_components,
                'component_sizes': component_sizes,
                'avg_component_size': np.mean(component_sizes),
                'std_component_size': np.std(component_sizes),
                'min_component_size': np.min(component_sizes),
                'max_component_size': np.max(component_sizes),
                'avg_log_determinant': np.mean(log_determinants),
                'avg_trace': np.mean(traces)
            })
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        layer_names = [d['layer'] for d in summary_data]
        n_components = [d['n_components'] for d in summary_data]
        avg_sizes = [d['avg_component_size'] for d in summary_data]
        std_sizes = [d['std_component_size'] for d in summary_data]
        
        # Component count vs average component size
        scatter = ax1.scatter(n_components, avg_sizes, s=100, alpha=0.7, c=range(len(layer_names)), 
                             cmap='viridis', edgecolors='black', linewidth=2)
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Average Component Size')
        ax1.set_title('Component Count vs Average Component Size')
        ax1.grid(True, alpha=0.3)
        
        # Add layer labels
        for i, layer in enumerate(layer_names):
            ax1.annotate(layer, (n_components[i], avg_sizes[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Component size distribution
        ax2.boxplot([d['component_sizes'] for d in summary_data], labels=layer_names)
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.set_ylabel('Component Size')
        ax2.set_title('Component Size Distribution by Layer')
        ax2.grid(True, alpha=0.3)
        
        # Component count vs sharpening metrics
        avg_log_dets = [d['avg_log_determinant'] for d in summary_data]
        avg_traces = [d['avg_trace'] for d in summary_data]
        
        ax3.scatter(n_components, avg_log_dets, s=100, alpha=0.7, c='red', 
                   edgecolors='black', linewidth=2, label='Log Determinant')
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Average Log Determinant')
        ax3.set_title('Component Count vs Cluster Sharpening (Log Determinant)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Component count vs trace
        ax4.scatter(n_components, avg_traces, s=100, alpha=0.7, c='blue', 
                   edgecolors='black', linewidth=2, label='Trace')
        ax4.set_xlabel('Number of Components')
        ax4.set_ylabel('Average Trace')
        ax4.set_title('Component Count vs Cluster Sharpening (Trace)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_specific_component_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary data to CSV
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(self.output_dir / 'layer_specific_component_summary.csv', index=False)
        
        print("Layer-specific component analysis saved")
    
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
    parser.add_argument('--zca_threshold', type=float, default=1e-5,
                       help='Threshold for ZCA whitening (default: 1e-5)')
    
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
    analyzer.run_complete_analysis(use_pre_saved_embeddings=args.use_pre_saved, zca_threshold=args.zca_threshold)


def run_analysis_with_pre_saved_embeddings(embeddings_dir, output_dir=None, max_components=20, 
                                          model_type="dimenet-10conf", checkpoint_index=0, 
                                          sample_frac=1.0, experiment_name=None, zca_threshold=1e-5):
    """
    Convenience function to run GMM analysis with pre-saved embeddings and ZCA whitening
    
    Args:
        embeddings_dir (str): Path to directory containing .pt embedding files
        output_dir (str): Output directory for results (optional)
        max_components (int): Maximum number of components to test for BIC/AIC
        model_type (str): Type of model (for naming purposes)
        checkpoint_index (int): Checkpoint index (for naming purposes)
        sample_frac (float): Fraction of data to use (1.0 = use all)
        experiment_name (str): Custom name for this experiment (optional)
        zca_threshold (float): Threshold for ZCA whitening (default: 1e-5)
    
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
    
    # Run analysis with pre-saved embeddings and ZCA whitening
    analyzer.run_complete_analysis(use_pre_saved_embeddings=True, zca_threshold=zca_threshold)
    
    return analyzer


if __name__ == "__main__":
    main()
