"""
Clustering methods for latent space: PGMeans-style Gaussian mixtures (with BIC/AIC),
HDBSCAN, Agglomerative, and KNN connected-components clustering.
Each returns labels array (non-negative integers; noise in HDBSCAN is remapped to last cluster).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def _flatten(X):
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X


def _fit_gaussian_mixture_for_k(X, k, random_state=42):
    """Fit one Gaussian mixture candidate with robust fallbacks."""
    from sklearn.mixture import GaussianMixture
    for cov_type in ("full", "diag"):
        for reg in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            try:
                model = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    random_state=random_state,
                    max_iter=300,
                    reg_covar=reg,
                    n_init=2,
                )
                model.fit(X)
                return model
            except Exception:
                continue
    return None


def fit_pgmeans(emb, k_min=2, k_max=15, random_state=42):
    """
    PGMeans-style Gaussian mixture selection using BIC/AIC sweeps.
    Returns labels, best_k, k_values, bic_list, aic_list.
    """
    # Use float64 to improve numerical stability for covariance estimation.
    X = StandardScaler().fit_transform(_flatten(emb)).astype(np.float64, copy=False)

    k_values = list(range(k_min, k_max + 1))
    bic_list, aic_list = [], []
    fitted_by_k = {}
    for k in k_values:
        model = _fit_gaussian_mixture_for_k(X, k, random_state=random_state)
        if model is not None:
            fitted_by_k[k] = model
            bic_list.append(float(model.bic(X)))
            aic_list.append(float(model.aic(X)))
        else:
            bic_list.append(float("inf"))
            aic_list.append(float("inf"))

    # If all candidates fail, return a single fallback cluster instead of raising.
    if not fitted_by_k:
        labels = np.zeros(X.shape[0], dtype=int)
        return {
            "labels": labels,
            "k": 1,
            "k_values": [1],
            "bic": [None],
            "aic": [None],
        }

    best_idx = int(np.argmin(bic_list))
    best_k = k_values[best_idx]
    model = fitted_by_k.get(best_k)
    if model is None:
        # np.argmin might point to a failed k if BIC ties/edge cases occur.
        best_k = min(fitted_by_k.keys())
        model = fitted_by_k[best_k]
    labels = model.predict(X)
    return {
        "labels": labels,
        "k": best_k,
        "k_values": k_values,
        "bic": bic_list,
        "aic": aic_list,
    }


def fit_pgmeans_at_k(emb, k, random_state=42, return_model=False):
    """
    Fit a single-k PGMeans candidate and return labels (and optional model).
    """
    X = StandardScaler().fit_transform(_flatten(emb)).astype(np.float64, copy=False)
    model = _fit_gaussian_mixture_for_k(X, int(k), random_state=random_state)
    if model is None:
        labels = np.zeros(X.shape[0], dtype=int)
        out = {
            "labels": labels,
            "k": 1,
            "bic": None,
            "aic": None,
        }
        if return_model:
            out["model"] = None
        return out
    labels = model.predict(X)
    out = {
        "labels": labels,
        "k": int(k),
        "bic": float(model.bic(X)),
        "aic": float(model.aic(X)),
    }
    if return_model:
        out["model"] = model
    return out


def fit_hdbscan(emb, min_cluster_size=5, min_samples=3, random_state=42):
    """
    HDBSCAN. Noise label -1 is remapped to max_label+1 so all labels are non-negative.
    Returns dict with labels, k (number of non-noise clusters), k_values=[k], bic=[None], aic=[None].
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("hdbscan is required. Install with: pip install hdbscan")
    X = StandardScaler().fit_transform(_flatten(emb))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    # Remap -1 (noise) to last cluster index so Sankey/representatives work
    uniq = np.unique(labels)
    if -1 in uniq:
        max_lab = int(labels.max())
        if max_lab >= 0:
            labels = np.where(labels == -1, max_lab + 1, labels)
        else:
            labels = np.zeros_like(labels, dtype=int)  # all noise -> single cluster 0
    labels = labels.astype(int)
    n_clusters = len(np.unique(labels))
    return {
        "labels": labels,
        "k": n_clusters,
        "k_values": [n_clusters],
        "bic": [None],
        "aic": [None],
    }


def fit_agglomerative(emb, n_clusters=None, k_min=2, k_max=15, random_state=42, linkage="ward"):
    """
    Agglomerative Clustering. If n_clusters is None, run PGMeans once to get best_k and use it.
    linkage: "ward" (default), "complete", or "average". Ward uses Euclidean only.
    Returns dict with labels, k, k_values=[k], bic=[None], aic=[None].
    """
    from sklearn.cluster import AgglomerativeClustering
    X = StandardScaler().fit_transform(_flatten(emb))
    if n_clusters is None:
        pg_out = fit_pgmeans(emb, k_min=k_min, k_max=k_max, random_state=random_state)
        n_clusters = pg_out["k"]
    n_clusters = max(2, min(n_clusters, len(X) - 1))
    link = (linkage or "ward").lower()
    if link not in ("ward", "complete", "average"):
        link = "ward"
    ac = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=link,
    )
    labels = ac.fit_predict(X)
    return {
        "labels": labels,
        "k": n_clusters,
        "k_values": [n_clusters],
        "bic": [None],
        "aic": [None],
    }


def fit_knn(emb, n_neighbors=10, random_state=42):
    """
    KNN graph clustering via connected components on a symmetric kNN graph.
    n_neighbors is the K in KNN.
    Returns dict with labels, k (#components), k_values, bic/aic as None.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    X = StandardScaler().fit_transform(_flatten(emb))
    n = int(X.shape[0])
    if n <= 1:
        labels = np.zeros(n, dtype=int)
        return {
            "labels": labels,
            "k": int(len(np.unique(labels))),
            "k_values": [int(len(np.unique(labels)))],
            "bic": [None],
            "aic": [None],
            "n_neighbors": 0,
        }

    k = int(n_neighbors) if n_neighbors is not None else 10
    k = max(1, min(k, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(X)
    idx = nbrs.kneighbors(X, return_distance=False)[:, 1:]  # drop self
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    graph = graph.maximum(graph.T)  # symmetric connectivity
    n_components, labels = connected_components(graph, directed=False)
    labels = labels.astype(int)
    return {
        "labels": labels,
        "k": int(n_components),
        "k_values": [int(n_components)],
        "bic": [None],
        "aic": [None],
        "n_neighbors": int(k),
    }


def fit_clustering(emb, method="pgmeans", k_min=2, k_max=15, random_state=42, **kwargs):
    """
    Single entry: method in ('pgmeans', 'hdbscan', 'agglomerative', 'knn').
    Returns dict with keys: labels, k, k_values, bic, aic.
    """
    method = (method or "pgmeans").lower()
    if method == "pgmeans":
        return fit_pgmeans(emb, k_min=k_min, k_max=k_max, random_state=random_state)
    if method == "hdbscan":
        return fit_hdbscan(
            emb,
            min_cluster_size=kwargs.get("min_cluster_size", 5),
            min_samples=kwargs.get("min_samples", 3),
            random_state=random_state,
        )
    if method == "agglomerative":
        return fit_agglomerative(
            emb,
            n_clusters=kwargs.get("n_clusters"),
            k_min=k_min,
            k_max=k_max,
            random_state=random_state,
            linkage=kwargs.get("linkage", "ward"),
        )
    if method == "knn":
        return fit_knn(
            emb,
            n_neighbors=kwargs.get("n_neighbors", 10),
            random_state=random_state,
        )
    raise ValueError("Unknown clustering method: {}".format(method))
