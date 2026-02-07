#!/usr/bin/env python3
"""
Generate FMI and CSPA values for each model+layer combination.

This script:
1. Loads embeddings for dimenet10conf and dimenet_noaug models
2. Performs GMM clustering for multiple k values
3. Computes FMI (Fowlkes-Mallows Index) between model pairs
4. Computes CSPA (Cluster-based Similarity Partitioning Algorithm) metrics
5. Updates or creates JSON files with these metrics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fowlkes_mallows_score


def _load_embeddings(path):
    """Load embeddings from various formats"""
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
    """Flatten embeddings if needed"""
    if embeddings.ndim > 2:
        return embeddings.reshape(embeddings.shape[0], -1)
    return embeddings


def _get_gmm_labels(embeddings, k, random_state=42):
    """Get GMM cluster labels for a specific k"""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    try:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            max_iter=200,
        )
        gmm.fit(embeddings_scaled)
        labels = gmm.predict(embeddings_scaled)
        return labels
    except Exception as e:
        print("  Error computing GMM for k={}: {}".format(k, e))
        return None


def compute_fmi_for_k(labels1, labels2, k):
    """Compute FMI between two label arrays for a specific k"""
    if labels1 is None or labels2 is None:
        return None
    
    # Ensure same length
    min_len = min(len(labels1), len(labels2))
    labels1 = labels1[:min_len]
    labels2 = labels2[:min_len]
    
    try:
        fmi = fowlkes_mallows_score(labels1, labels2)
        return float(fmi)
    except Exception as e:
        print("  Error computing FMI for k={}: {}".format(k, e))
        return None


def compute_cspa_for_k(labels1, labels2, k):
    """Compute CSPA co-association matrix for a specific k"""
    if labels1 is None or labels2 is None:
        return None
    
    # Ensure same length
    min_len = min(len(labels1), len(labels2))
    labels1 = labels1[:min_len]
    labels2 = labels2[:min_len]
    
    try:
        # Build co-association: same cluster in both models = 1, otherwise = 0
        sim1 = (labels1[:, None] == labels1[None, :]).astype(float)
        sim2 = (labels2[:, None] == labels2[None, :]).astype(float)
        
        # Average: points are similar if they co-occur in both clusterings
        sim_combined = (sim1 + sim2) / 2.0
        
        # Return Frobenius norm as scalar CSPA metric
        cspa_norm = float(np.linalg.norm(sim_combined, ord='fro'))
        return cspa_norm
    except Exception as e:
        print("  Error computing CSPA for k={}: {}".format(k, e))
        return None


def _layer_file_map(model_key):
    """Map layer names to embedding file names"""
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
    """Get default embeddings directory for a model"""
    if model_key == "dimenet10conf":
        return Path(embeddings_root) / "dimenet10conf_embeddings"
    return Path(embeddings_root) / "dimenet_noaug_embeddings"


def compute_fmi_cspa_for_layer(
    embeddings_10conf, 
    embeddings_noaug, 
    layer_name, 
    k_values
):
    """
    Compute FMI and CSPA values for a layer across k values
    
    Returns:
        Tuple of (fmi_values, cspa_values, avg_fmi, avg_cspa)
    """
    fmi_values = []
    cspa_values = []
    
    for k in k_values:
        # Get labels for both models
        labels_10 = _get_gmm_labels(embeddings_10conf, k)
        labels_no = _get_gmm_labels(embeddings_noaug, k)
        
        # Compute FMI
        fmi = compute_fmi_for_k(labels_10, labels_no, k)
        if fmi is not None:
            fmi_values.append(fmi)
        else:
            fmi_values.append(None)
        
        # Compute CSPA
        cspa = compute_cspa_for_k(labels_10, labels_no, k)
        if cspa is not None:
            cspa_values.append(cspa)
        else:
            cspa_values.append(None)
    
    # Compute averages
    valid_fmi = [f for f in fmi_values if f is not None]
    valid_cspa = [c for c in cspa_values if c is not None]
    
    avg_fmi = float(np.mean(valid_fmi)) if valid_fmi else None
    avg_cspa = float(np.mean(valid_cspa)) if valid_cspa else None
    
    return fmi_values, cspa_values, avg_fmi, avg_cspa


def update_json_files(
    embeddings_root,
    output_dir,
    k_min=2,
    k_max=20
):
    """Update JSON files with FMI and CSPA values"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    k_values = list(range(k_min, k_max + 1))
    layer_map_10conf = _layer_file_map("dimenet10conf")
    layer_map_noaug = _layer_file_map("dimenet_noaug")
    
    embeddings_dir_10conf = _default_embeddings_dir(embeddings_root, "dimenet10conf")
    embeddings_dir_noaug = _default_embeddings_dir(embeddings_root, "dimenet_noaug")
    
    # Load all embeddings first
    print("Loading embeddings...")
    embeddings_dict_10conf = {}
    embeddings_dict_noaug = {}
    
    for layer_name, filename in layer_map_10conf.items():
        path = embeddings_dir_10conf / filename
        if path.exists():
            print("  Loading {}...".format(path.name))
            emb = _load_embeddings(path)
            embeddings_dict_10conf[layer_name] = _flatten_embeddings(emb)
    
    for layer_name, filename in layer_map_noaug.items():
        path = embeddings_dir_noaug / filename
        if path.exists():
            print("  Loading {}...".format(path.name))
            emb = _load_embeddings(path)
            embeddings_dict_noaug[layer_name] = _flatten_embeddings(emb)
    
    # Process each layer
    for layer_name in sorted(set(embeddings_dict_10conf.keys()) & set(embeddings_dict_noaug.keys())):
        print("\nProcessing layer: {}".format(layer_name))
        
        emb_10 = embeddings_dict_10conf[layer_name]
        emb_no = embeddings_dict_noaug[layer_name]
        
        print("  dimenet10conf shape: {}".format(emb_10.shape))
        print("  dimenet_noaug shape: {}".format(emb_no.shape))
        
        # Compute FMI and CSPA
        fmi_values, cspa_values, avg_fmi, avg_cspa = compute_fmi_cspa_for_layer(
            emb_10, emb_no, layer_name, k_values
        )
        
        fmi_strs = ['{:.4f}'.format(v) if v is not None else 'None' for v in fmi_values]
        cspa_strs = ['{:.4f}'.format(v) if v is not None else 'None' for v in cspa_values]
        print("  FMI values: {}".format(fmi_strs))
        print("  CSPA values: {}".format(cspa_strs))
        if avg_fmi:
            print("  Average FMI: {:.4f}".format(avg_fmi))
        else:
            print("  Average FMI: None")
        if avg_cspa:
            print("  Average CSPA: {:.4f}".format(avg_cspa))
        else:
            print("  Average CSPA: None")
        
        # Update dimenet10conf JSON
        json_path_10 = output_dir / "dimenet10conf_{}.json".format(layer_name)
        if json_path_10.exists():
            with open(json_path_10, 'r') as f:
                data = json.load(f)
            data['fmi_vs_noaug'] = fmi_values
            data['cspa_vs_noaug'] = cspa_values
            data['avg_fmi_vs_noaug'] = avg_fmi
            data['avg_cspa_vs_noaug'] = avg_cspa
            with open(json_path_10, 'w') as f:
                json.dump(data, f, indent=2)
            print("  Updated {}".format(json_path_10.name))
        
        # Update dimenet_noaug JSON
        json_path_no = output_dir / "dimenet_noaug_{}.json".format(layer_name)
        if json_path_no.exists():
            with open(json_path_no, 'r') as f:
                data = json.load(f)
            data['fmi_vs_10conf'] = fmi_values  # Same values, just swapped reference
            data['cspa_vs_10conf'] = cspa_values
            data['avg_fmi_vs_10conf'] = avg_fmi
            data['avg_cspa_vs_10conf'] = avg_cspa
            with open(json_path_no, 'w') as f:
                json.dump(data, f, indent=2)
            print("  Updated {}".format(json_path_no.name))


def main():
    default_embeddings_root = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces"
    default_output_dir = "/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/frontend/public/data/bic_aic"
    
    parser = argparse.ArgumentParser(description="Generate FMI/CSPA metrics")
    parser.add_argument("--embeddings-root", type=str, default=default_embeddings_root)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=20)
    args = parser.parse_args()
    
    print("="*70)
    print("Generating FMI and CSPA metrics")
    print("="*70)
    
    update_json_files(
        args.embeddings_root,
        args.output_dir,
        args.k_min,
        args.k_max
    )
    
    print("\n" + "="*70)
    print("Completed!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
