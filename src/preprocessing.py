import torch
import numpy as np
from sklearn.manifold import MDS
import pandas as pd


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


def get_euclidean_distances(X):
    """Compute pairwise Euclidean distances."""
    X_squared = np.sum(X**2, axis=1, keepdims=True)
    distances_squared = X_squared + X_squared.T - 2 * X @ X.T
    distances_squared = np.maximum(distances_squared, 0)
    return np.sqrt(distances_squared)


def generate_scatter_plot_data(embeddings, layer_order, id_mapping={}):
    n_samples = embeddings[0].shape[0]
    reducer = MDS(
        n_components=2,
        dissimilarity="precomputed",
        n_init=1,
    )

    data = []
    poses = {}
    for embedding, layer in zip(embeddings, layer_order):
        distances = get_euclidean_distances(embedding)
        poses[layer] = reducer.fit_transform(distances)

    for i in range(n_samples):
        d = []
        for layer in layer_order:
            p = poses[layer]
            d.append({"x": float(p[i, 0]), "y": float(p[i, 1])})
        groupID = id_mapping.get(i, 0)
        data.append({"positions": d, "groupID": groupID, "id": i})

    return data
