#!/usr/bin/env python3
"""
Dimensionality reduction for latent space visualization.
Supports PCA, MDS, UMAP, and t-SNE. All return (n_samples, 2) arrays.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

_RANDOM_STATE = 42


def _standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def reduce_pca(
    X: np.ndarray,
    n_components: int = 2,
    standardize: bool = True,
    random_state: int = _RANDOM_STATE,
) -> np.ndarray:
    from sklearn.decomposition import PCA
    if standardize:
        X = _standardize(X)
    if X.shape[1] > 50:
        pca50 = PCA(n_components=50, random_state=random_state)
        X = pca50.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X)


def reduce_mds(
    X: np.ndarray,
    n_components: int = 2,
    standardize: bool = True,
    random_state: int = _RANDOM_STATE,
    n_jobs: int = 1,
    max_iter: int = 300,
) -> np.ndarray:
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import normalize
    if standardize:
        X = _standardize(X)
    if X.shape[1] > 50:
        from sklearn.decomposition import PCA
        X = PCA(n_components=50, random_state=random_state).fit_transform(X)
    # Use cosine distance: 1 - cosine similarity (as sklearn's 'cosine' metric).
    # We compute a precomputed dissimilarity matrix so the distance definition is explicit.
    Xn = normalize(X, norm="l2", axis=1)
    D = pairwise_distances(Xn, metric="cosine")
    mds = MDS(
        n_components=n_components,
        random_state=random_state,
        dissimilarity="precomputed",
        n_jobs=n_jobs,
        max_iter=max_iter,
    )
    return mds.fit_transform(D)


def reduce_umap(
    X: np.ndarray,
    n_components: int = 2,
    standardize: bool = True,
    random_state: int = _RANDOM_STATE,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
    if standardize:
        X = _standardize(X)
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        verbose=False,
    )
    return reducer.fit_transform(X)


def reduce_tsne(
    X: np.ndarray,
    n_components: int = 2,
    standardize: bool = True,
    random_state: int = _RANDOM_STATE,
    perplexity: float = 30.0,
    n_iter: int = 500,
) -> np.ndarray:
    from sklearn.manifold import TSNE
    if standardize:
        X = _standardize(X)
    if X.shape[1] > 50:
        from sklearn.decomposition import PCA
        X = PCA(n_components=50, random_state=random_state).fit_transform(X)
    n = X.shape[0]
    perp = min(perplexity, max(5, n // 4))
    iters = min(n_iter, max(250, n * 5))  # at least 250 (sklearn requirement), cap for speed
    # sklearn changed parameter name from n_iter -> max_iter in newer versions.
    # Try n_iter first for backward compatibility, then max_iter.
    try:
        tsne = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perp,
            n_iter=iters,
            init="pca",
        )
    except TypeError:
        tsne = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perp,
            max_iter=iters,
            init="pca",
        )
    return tsne.fit_transform(X)


def reduce_embedding(
    X: np.ndarray,
    method: str = "mds",
    random_state: int = _RANDOM_STATE,
    **kwargs,
) -> np.ndarray:
    """
    Single entry for 2D reduction. method in ('pca', 'mds', 'umap', 'tsne').
    When input has only 1 feature (or effective rank 1), returns 2D with second
    dimension spread (not a line) so scatter plots are visible.
    """
    n_samples, n_features = X.shape[0], X.shape[1]
    if n_samples < 2:
        out = np.zeros((n_samples, 2), dtype=np.float64)
        if n_features >= 1:
            out[:, 0] = X.ravel()
        return out
    if n_features < 2:
        # Single feature: use x = feature, y = spread (so not a single line)
        out = np.zeros((n_samples, 2), dtype=np.float64)
        out[:, 0] = X.ravel()
        out[:, 1] = np.linspace(-1.0, 1.0, n_samples)
        return out
    method = method.lower()
    result = None
    if method == "pca":
        result = reduce_pca(X, random_state=random_state, **kwargs)
    elif method == "mds":
        result = reduce_mds(X, random_state=random_state, **kwargs)
    elif method == "umap":
        result = reduce_umap(X, random_state=random_state, **kwargs)
    elif method == "tsne":
        result = reduce_tsne(X, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    # If second dimension has near-zero variance (rank-1 data), spread it so scatter isn't a line
    result = _ensure_2d_spread(result, random_state=random_state)
    return result


def _ensure_2d_spread(arr: np.ndarray, random_state: int = _RANDOM_STATE) -> np.ndarray:
    """If second column has negligible variance, replace with spread so plot is not a line."""
    if arr.shape[1] < 2:
        return arr
    col1 = arr[:, 1]
    var1 = np.var(col1)
    scale = max(1e-10, np.std(arr[:, 0]))
    if var1 < (0.01 * scale) ** 2:  # effectively degenerate
        rs = np.random.RandomState(random_state)
        arr = arr.copy()
        arr[:, 1] = np.linspace(-scale * 0.5, scale * 0.5, arr.shape[0]) + rs.randn(arr.shape[0]) * scale * 0.02
    return arr
