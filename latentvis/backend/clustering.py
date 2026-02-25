"""
Clustering methods for latent space: GMM (with BIC/AIC), HDBSCAN, Agglomerative.
Each returns labels array (non-negative integers; noise in HDBSCAN is remapped to last cluster).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def _flatten(X):
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X


def fit_gmm(emb, k_min=2, k_max=15, random_state=42):
    """
    GMM with BIC-based model selection. Returns labels, best_k, k_values, bic_list, aic_list.
    """
    from sklearn.mixture import GaussianMixture
    X = StandardScaler().fit_transform(_flatten(emb))
    k_values = list(range(k_min, k_max + 1))
    bic_list, aic_list = [], []
    for k in k_values:
        try:
            gmm = GaussianMixture(
                n_components=k, covariance_type="full", random_state=random_state, max_iter=200
            )
            gmm.fit(X)
            bic_list.append(float(gmm.bic(X)))
            aic_list.append(float(gmm.aic(X)))
        except Exception:
            bic_list.append(float("inf"))
            aic_list.append(float("inf"))
    best_k = k_values[int(np.argmin(bic_list))]
    gmm = GaussianMixture(
        n_components=best_k, covariance_type="full", random_state=random_state, max_iter=200
    )
    gmm.fit(X)
    labels = gmm.predict(X)
    return {
        "labels": labels,
        "k": best_k,
        "k_values": k_values,
        "bic": bic_list,
        "aic": aic_list,
    }


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
    Agglomerative Clustering. If n_clusters is None, run GMM once to get best_k and use it.
    linkage: "ward" (default), "complete", or "average". Ward uses Euclidean only.
    Returns dict with labels, k, k_values=[k], bic=[None], aic=[None].
    """
    from sklearn.cluster import AgglomerativeClustering
    X = StandardScaler().fit_transform(_flatten(emb))
    if n_clusters is None:
        gmm_out = fit_gmm(emb, k_min=k_min, k_max=k_max, random_state=random_state)
        n_clusters = gmm_out["k"]
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


def fit_clustering(emb, method="gmm", k_min=2, k_max=15, random_state=42, **kwargs):
    """
    Single entry: method in ('gmm', 'hdbscan', 'agglomerative').
    Returns dict with keys: labels, k, k_values, bic, aic.
    """
    method = (method or "gmm").lower()
    if method == "gmm":
        return fit_gmm(emb, k_min=k_min, k_max=k_max, random_state=random_state)
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
    raise ValueError("Unknown clustering method: {}".format(method))
