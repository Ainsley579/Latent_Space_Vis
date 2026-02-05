#!/usr/bin/env python3
"""
Backend analyzer for GMM clustering and dimensionality reduction.

Provides simplified interface for:
1. GMM fitting on embeddings
2. PCA/UMAP 降维
3. Cluster assignment and metrics
4. Sankey flow analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple, List

try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict


class EmbeddingAnalyzer(object):
    """Analyze embeddings and compute GMM clustering."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def fit_gmm(self, embeddings, n_components_range=(2, 10)):
        """
        Fit GMM and find optimal number of components.
        
        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            n_components_range: tuple (min_components, max_components)
        
        Returns:
            dict with optimal_gmm, optimal_n_components, bic_scores, aic_scores
        """
        # Scale embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        min_comp, max_comp = n_components_range
        bic_scores = []
        aic_scores = []
        gmm_models = []
        
        for n_comp in range(min_comp, max_comp + 1):
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type='full',
                random_state=self.random_state
            )
            gmm.fit(embeddings_scaled)
            gmm_models.append(gmm)
            bic_scores.append(gmm.bic(embeddings_scaled))
            aic_scores.append(gmm.aic(embeddings_scaled))
        
        # Find optimal
        optimal_idx = np.argmin(bic_scores)
        optimal_n_components = min_comp + optimal_idx
        optimal_gmm = gmm_models[optimal_idx]
        
        return {
            'optimal_gmm': optimal_gmm,
            'optimal_n_components': int(optimal_n_components),
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'n_components_range': n_components_range
        }
    
    def get_cluster_assignments(self, embeddings, gmm):
        """Get cluster assignments from fitted GMM."""
        embeddings_scaled = self.scaler.transform(embeddings)
        labels = gmm.predict(embeddings_scaled)
        probabilities = gmm.predict_proba(embeddings_scaled)
        return {
            'labels': labels,
            'probabilities': probabilities
        }
    
    def compute_silhouette(self, embeddings, labels):
        """Compute silhouette score."""
        if len(np.unique(labels)) < 2:
            return -1.0
        embeddings_scaled = self.scaler.transform(embeddings)
        return silhouette_score(embeddings_scaled, labels)


class DimensionalityReducer(object):
    """Reduce embeddings to 2D for visualization."""
    
    def __init__(self, method='pca', random_state=42):
        """
        Args:
            method: 'pca' or 'umap'
            random_state: random seed
        """
        self.method = method.lower()
        self.random_state = random_state
        self.reducer = None
    
    def fit_reduce(self, embeddings):
        """Fit reducer and return 2D coordinates."""
        if self.method == 'pca':
            self.reducer = PCA(n_components=2, random_state=self.random_state)
        elif self.method == 'umap':
            try:
                import umap
                self.reducer = umap.UMAP(
                    n_components=2,
                    random_state=self.random_state,
                    verbose=0
                )
            except ImportError:
                print("Warning: umap not available, falling back to PCA")
                self.method = 'pca'
                self.reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")
        
        reduced = self.reducer.fit_transform(embeddings)
        return reduced


class SankeyFlowAnalyzer(object):
    """Analyze cluster transitions across layers (for Sankey diagram)."""
    
    @staticmethod
    def compute_transitions(labels_from, labels_to):
        """
        Compute transitions between two clustering results.
        
        Args:
            labels_from: cluster labels from previous layer
            labels_to: cluster labels in next layer
            
        Returns:
            dict with transition probabilities
        """
        n_from = len(np.unique(labels_from))
        n_to = len(np.unique(labels_to))
        
        transition_matrix = np.zeros((n_from, n_to))
        
        for i in range(n_from):
            mask = labels_from == i
            if not mask.any():
                continue
            labels_to_subset = labels_to[mask]
            for j in range(n_to):
                count = np.sum(labels_to_subset == j)
                transition_matrix[i, j] = count
        
        # Convert to probability
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        prob_matrix = transition_matrix / row_sums
        
        # Build output format
        transitions = []
        for i in range(n_from):
            for j in range(n_to):
                if transition_matrix[i, j] > 0:
                    transitions.append({
                        'from_components': i,  # This will be updated by caller
                        'to_components': j,
                        'from_cluster': i,
                        'to_cluster': j,
                        'count': int(transition_matrix[i, j]),
                        'probability': float(prob_matrix[i, j])
                    })
        
        return transitions


class ClusterRepresentativesExtractor(object):
    """Extract representative samples (closest/farthest to cluster centers)."""
    
    @staticmethod
    def get_top_k(embeddings, labels, k=5):
        """
        Get top-k closest and farthest samples per cluster.
        
        Args:
            embeddings: numpy array of embeddings
            labels: cluster labels
            k: number of top samples to return
            
        Returns:
            dict with top_k_closest and top_k_farthest per cluster
        """
        from scipy.spatial.distance import cdist
        
        n_clusters = len(np.unique(labels))
        representatives = {}
        
        # Compute cluster centers
        centers = np.array([
            embeddings[labels == c].mean(axis=0)
            for c in range(n_clusters)
        ])
        
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_embeddings = embeddings[mask]
            cluster_indices = np.where(mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Distances to center
            center = centers[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            
            # Get top-k closest and farthest
            closest_idx = np.argsort(distances)[:k]
            farthest_idx = np.argsort(distances)[-k:][::-1]
            
            representatives[cluster_id] = {
                'top_k_closest': [int(cluster_indices[i]) for i in closest_idx],
                'top_k_farthest': [int(cluster_indices[i]) for i in farthest_idx],
                'k': k
            }
        
        return representatives
