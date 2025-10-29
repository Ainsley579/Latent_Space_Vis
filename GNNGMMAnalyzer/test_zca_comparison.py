#!/usr/bin/env python3
"""
Quick test to compare component counts with and without ZCA whitening
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def adaptive_zca_whitening(X, threshold=1e-5, epsilon=1e-5):
    """ZCA whitening function"""
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

def find_optimal_components(embeddings, use_zca=True, max_components=20, threshold=1e-5):
    """Find optimal components with or without ZCA whitening"""
    print(f"Testing with ZCA whitening: {use_zca}")
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    if use_zca:
        # Apply ZCA whitening
        embeddings_processed, _ = adaptive_zca_whitening(embeddings_scaled, threshold=threshold)
        print(f"  ZCA Whitening: {embeddings_processed.shape[1]} features kept out of {embeddings_scaled.shape[1]}")
    else:
        # Use scaled embeddings without ZCA
        embeddings_processed = embeddings_scaled
    
    # Test different numbers of components
    n_components_range = range(2, max_components + 1)
    bic_scores = []
    aic_scores = []
    silhouette_scores = []
    
    for n_components in n_components_range:
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=42,
                max_iter=100
            )
            gmm.fit(embeddings_processed)
            
            bic_scores.append(gmm.bic(embeddings_processed))
            aic_scores.append(gmm.aic(embeddings_processed))
            
            labels = gmm.predict(embeddings_processed)
            sil_score = silhouette_score(embeddings_processed, labels)
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
    
    print(f"  Optimal components - BIC: {optimal_bic}, AIC: {optimal_aic}, Silhouette: {optimal_sil}")
    
    return optimal_aic, bic_scores, aic_scores, silhouette_scores

def main():
    """Compare component counts with and without ZCA whitening"""
    
    # Load embeddings
    embeddings_path = Path("/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings")
    
    # Test with the final layer (u_mlp_l2)
    layer_file = embeddings_path / "u_mlp_l2_cas.pt"
    
    if not layer_file.exists():
        print(f"File not found: {layer_file}")
        return
    
    # Load embeddings
    embeddings_tensor = torch.load(layer_file, map_location='cpu')
    embeddings = embeddings_tensor.detach().cpu().numpy()
    
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    print(f"Testing layer: u_mlp_l2")
    print(f"Embedding shape: {embeddings.shape}")
    print("=" * 50)
    
    # Test WITHOUT ZCA whitening
    print("\n1. WITHOUT ZCA Whitening:")
    optimal_no_zca, _, _, _ = find_optimal_components(embeddings, use_zca=False, max_components=30)
    
    # Test WITH ZCA whitening
    print("\n2. WITH ZCA Whitening:")
    optimal_with_zca, _, _, _ = find_optimal_components(embeddings, use_zca=True, max_components=30)
    
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS:")
    print(f"Without ZCA whitening: {optimal_no_zca} components")
    print(f"With ZCA whitening:    {optimal_with_zca} components")
    print(f"Difference:            {optimal_with_zca - optimal_no_zca} components")
    
    if optimal_with_zca > optimal_no_zca:
        print("✅ ZCA whitening INCREASES component count")
    elif optimal_with_zca < optimal_no_zca:
        print("✅ ZCA whitening DECREASES component count")
    else:
        print("✅ ZCA whitening has NO EFFECT on component count")

if __name__ == "__main__":
    main()
