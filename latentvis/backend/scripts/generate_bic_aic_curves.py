#!/usr/bin/env python3
"""
Generate BIC/AIC curves per layer for dimenet10conf and dimenet_noaug.

Outputs one JSON file per model+layer in the target output directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _load_embeddings(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(str(path))
    if path.suffix == ".npz":
        data = np.load(str(path))
        if hasattr(data, "files") and data.files:
            return data[data.files[0]]
        return data
    if path.suffix == ".pt":
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to load .pt embeddings")
        tensor = torch.load(str(path), map_location="cpu")
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)
    raise ValueError("Unsupported embedding format: {}".format(path))


def _flatten_embeddings(embeddings):
    if embeddings.ndim > 2:
        return embeddings.reshape(embeddings.shape[0], -1)
    return embeddings


def _compute_bic_aic(embeddings, k_min, k_max):
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    k_values = list(range(k_min, k_max + 1))
    bic_scores = []
    aic_scores = []

    for k in k_values:
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42,
                max_iter=200,
            )
            gmm.fit(embeddings_scaled)
            bic_scores.append(float(gmm.bic(embeddings_scaled)))
            aic_scores.append(float(gmm.aic(embeddings_scaled)))
        except Exception:
            bic_scores.append(float("inf"))
            aic_scores.append(float("inf"))

    return k_values, bic_scores, aic_scores


def _layer_file_map(model_key):
    if model_key == "dimenet10conf":
        return {
            "emb": "emb.pt",
            "output_block_0": "output0.pt",
            "output_block_1": "output1.pt",
            "output_block_2": "output2.pt",
            "output_block_3": "output3.pt",
            "output_block_4": "output4.pt",
            "u_mlp_input": "u_mlp_input.pt",
            "u_mlp_l1": "u_mlp_l1.pt",
            "u_mlp_l2": "u_mlp_l2.pt",
        }
    return {
        "emb": "emb.pt",
        "output_block_0": "output0.pt",
        "output_block_1": "output1.pt",
        "output_block_2": "output2.pt",
        "output_block_3": "output3.pt",
        "output_block_4": "output4.pt",
        "u_mlp_input": "mlp_input.pt",
        "u_mlp_l1": "mlp_l1.pt",
        "u_mlp_l2": "mlp_l2.pt",
    }


def _default_embeddings_dir(embeddings_root, model_key):
    if model_key == "dimenet10conf":
        return Path(embeddings_root) / "dimenet10conf_embeddings"
    return Path(embeddings_root) / "dimenet_noaug_embeddings"


def generate_for_model(model_key, embeddings_dir, output_dir, k_min, k_max):
    embeddings_dir = Path(embeddings_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_map = _layer_file_map(model_key)

    for layer_name, filename in layer_map.items():
        path = embeddings_dir / filename
        if not path.exists():
            print("Skipping {} {} (missing {})".format(model_key, layer_name, path))
            continue

        embeddings = _load_embeddings(path)
        embeddings = _flatten_embeddings(embeddings)
        k_values, bic_scores, aic_scores = _compute_bic_aic(embeddings, k_min, k_max)

        payload = {
            "model": model_key,
            "layer": layer_name,
            "n_samples": int(embeddings.shape[0]),
            "n_features": int(embeddings.shape[1]),
            "k_values": k_values,
            "bic": bic_scores,
            "aic": aic_scores,
        }

        out_path = output_dir / "{}_{}.json".format(model_key, layer_name)
        with open(str(out_path), "w") as f:
            json.dump(payload, f, indent=2)
        print("Wrote {}".format(out_path))


def main():
    default_embeddings_root = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces"
    default_output_dir = "/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/frontend/public/data/bic_aic"

    parser = argparse.ArgumentParser(description="Generate BIC/AIC curves")
    parser.add_argument("--embeddings-root", type=str, default=default_embeddings_root)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument(
        "--models",
        type=str,
        default="dimenet10conf,dimenet_noaug",
        help="Comma-separated model keys",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_key in models:
        embeddings_dir = _default_embeddings_dir(args.embeddings_root, model_key)
        generate_for_model(model_key, embeddings_dir, args.output_dir, args.k_min, args.k_max)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
