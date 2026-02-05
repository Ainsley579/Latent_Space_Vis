#!/usr/bin/env python3
"""
Core utilities for embedding processing pipeline.

This module contains general functions that can be used with any embeddings,
regardless of their source or format.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.manifold import MDS
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import torch


def load_embeddings_from_dict(embeddings_dict):
    """
    Load embeddings from a dictionary.
    
    Args:
        embeddings_dict: Dictionary mapping layer names to numpy arrays or torch tensors
        
    Returns:
        Dictionary mapping layer names to numpy arrays (2D: n_samples x n_features)
    """
    processed = {}
    
    for layer_name, data in embeddings_dict.items():
        if isinstance(data, torch.Tensor):
            data = data.squeeze().cpu().numpy()
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Ensure 2D array
        if data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        elif data.ndim == 1:
            data = data.reshape(-1, 1)
        
        processed[layer_name] = data
    
    return processed


def load_embeddings_from_files(embeddings_dir, layer_files):
    """
    Load embeddings from files.
    
    Args:
        embeddings_dir: Directory containing embedding files
        layer_files: Dictionary mapping layer names to filenames
        
    Returns:
        Dictionary mapping layer names to numpy arrays
    """
    embeddings_dir = Path(embeddings_dir)
    embeddings = {}
    
    for layer_name, filename in layer_files.items():
        filepath = embeddings_dir / filename
        
        if not filepath.exists():
            print(f"  ⚠ {filename} not found, skipping {layer_name}")
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
    
    return embeddings


def adaptive_zca_whitening(X, threshold=1e-5, epsilon=1e-5):
    """
    Apply adaptive ZCA whitening to X.
    
    Args:
        X: Matrix to whiten (n_samples x n_features)
        threshold: Minimum singular value to keep
        epsilon: Small value to avoid division by zero
    
    Returns:
        X_whitened: Whitened matrix
        zca_transform: ZCA transformation matrix
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
    
    return X_whitened, red_zca


def fit_gmm(embeddings, n_components, zca_threshold=1e-5, random_state=42, max_iter=200):
    """
    Fit GMM with specified number of components.
    
    Args:
        embeddings: Embedding matrix (n_samples x n_features)
        n_components: Number of GMM components
        zca_threshold: Threshold for ZCA whitening
        random_state: Random state for GMM
        max_iter: Maximum iterations for GMM
        
    Returns:
        Dictionary containing:
            - 'gmm': Fitted GMM model
            - 'labels': Cluster assignments (n_samples,)
            - 'probabilities': Cluster probabilities (n_samples x n_components)
            - 'means': Cluster means
            - 'covariances': Cluster covariances
            - 'weights': Cluster weights
            - 'embeddings_processed': Processed embeddings (after scaling and ZCA)
    """
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Apply ZCA whitening
    embeddings_processed, zca_transform = adaptive_zca_whitening(
        embeddings_scaled, threshold=zca_threshold
    )
    
    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state,
        max_iter=max_iter
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
        'embeddings_processed': embeddings_processed,
        'scaler': scaler,
        'zca_transform': zca_transform
    }


def find_optimal_cluster_alignment(transition_matrix, n_components):
    """
    Find optimal cluster alignment from transition matrix using Hungarian algorithm.
    
    Args:
        transition_matrix: Transition matrix (n_components x n_components)
        n_components: Number of components
        
    Returns:
        Permutation array mapping old cluster indices to new aligned indices
    """
    # Normalize transition matrix
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix_norm = transition_matrix / (row_sums + 1e-10)
    
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


def compute_mds_positions(embeddings_dict, n_components=2, random_state=42):
    """
    Compute MDS positions for all layers.
    
    Args:
        embeddings_dict: Dictionary mapping layer names to embedding arrays
        n_components: Number of MDS dimensions (default: 2)
        random_state: Random state for MDS
        
    Returns:
        Dictionary mapping layer names to MDS positions (n_samples x n_components)
    """
    def get_euclidean_distances(X):
        """Compute pairwise Euclidean distances."""
        X_squared = np.sum(X**2, axis=1, keepdims=True)
        distances_squared = X_squared + X_squared.T - 2 * X @ X.T
        distances_squared = np.maximum(distances_squared, 0)
        distances = np.sqrt(distances_squared)
        # Ensure symmetry to avoid floating-point precision issues
        # MDS requires a symmetric distance matrix
        distances = (distances + distances.T) / 2.0
        return distances
    
    positions = {}
    reducer = MDS(
        n_components=n_components,
        dissimilarity='precomputed',
        random_state=random_state,
        n_init=1
    )
    
    for layer_name, embedding in embeddings_dict.items():
        # Compute pairwise distances
        distances = get_euclidean_distances(embedding)
        
        # Fit MDS
        pos = reducer.fit_transform(distances)
        positions[layer_name] = pos
    
    return positions


def map_embedding_indices_to_real_ids(embedding_indices, id_mapping=None):
    """
    Map embedding indices to real IDs (original dataset indices).
    
    Args:
        embedding_indices: Array or list of embedding indices
        id_mapping: Dictionary mapping embedding indices to real IDs (optional)
        
    Returns:
        Array of real IDs (same length as embedding_indices)
    """
    if id_mapping is None:
        return np.array(embedding_indices)
    
    real_ids = []
    for idx in embedding_indices:
        if idx in id_mapping:
            real_ids.append(id_mapping[idx])
        else:
            real_ids.append(idx)  # Fallback to original index
    
    return np.array(real_ids)

