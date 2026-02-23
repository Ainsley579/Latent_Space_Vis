#!/usr/bin/env python3
"""
Unified embedding loader for Latent_Space_Vis backend.

Accepts:
1. Single .pt file: PyTorch-saved object. Supports:
   - Nested dict[epoch_key][layer_key] = tensor (or list of tensors)
   - Nested dict[layer_key][epoch_key] = tensor
   - Flat dict[layer_key] = tensor (treated as single epoch "default")
   - Dict with keys "epochs", "layers", "embeddings" (array shape like n_epochs x n_layers x n_samples x dim)
2. Directory: multiple .pt or .npy files, one per layer (layer name from filename stem).
   Treated as single epoch; model name can be from directory name or argument.

Returns normalized structure:
  {
    "cells": {(epoch_key, layer_key): np.ndarray of shape (n_samples, n_features)},
    "epoch_keys": [ordered epoch identifiers],
    "layer_keys": [ordered layer identifiers],
    "n_samples": int,
    "real_ids": None or list (optional external IDs; if None, use 0..n_samples-1)
  }
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _flatten_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure (n_samples, n_features)."""
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr


def _find_tensors_nested(
    obj: Any,
    path: List[str],
    out: List[Tuple[List[str], np.ndarray]],
    max_depth: int = 10,
) -> None:
    """Recursively find all tensors/ndarrays in nested dict/list. path = list of keys/indices."""
    if max_depth <= 0:
        return
    if hasattr(obj, "shape") and hasattr(obj, "ndim"):
        arr = _to_numpy(obj)
        if arr.size > 0:
            out.append((list(path), _flatten_2d(arr)))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _find_tensors_nested(v, path + [str(k)], out, max_depth - 1)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _find_tensors_nested(v, path + [str(i)], out, max_depth - 1)
        return


def _infer_epoch_layer_keys(
    paths_with_arrays: List[Tuple[List[str], np.ndarray]]
) -> Tuple[List[str], List[str], Dict[Tuple[str, str], np.ndarray]]:
    """
    Infer epoch and layer keys from paths like ["epoch_0", "layer_0"] or ["emb"], ["0","emb"].
    Returns (epoch_keys, layer_keys, cells dict).
    """
    cells = {}
    epoch_set = set()
    layer_set = set()

    for path, arr in paths_with_arrays:
        if len(path) >= 2:
            # Assume first level = epoch, second = layer (or vice versa; we try both)
            a, b = path[0], path[1]
            epoch_set.add(a)
            layer_set.add(b)
            cells[(a, b)] = arr
        elif len(path) == 1:
            # Single key: treat as layer, epoch = "default"
            layer_set.add(path[0])
            epoch_set.add("default")
            cells[("default", path[0])] = arr
        else:
            continue

    # If we only have one epoch, use "default" or the only one
    epoch_keys = sorted(epoch_set)
    layer_keys = sorted(layer_set)
    return epoch_keys, layer_keys, cells


def load_single_pt(pt_path: Path) -> Dict[str, Any]:
    """
    Load a single .pt file and normalize to cells + epoch_keys + layer_keys.

    Handles:
    - dict[epoch][layer] = tensor
    - dict[layer] = tensor  -> epoch = "default"
    - dict["epochs"] = list, dict["layers"] = list, dict["embeddings"] = array
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to load .pt files")
    try:
        raw = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(str(pt_path), map_location="cpu")
    path_arrays: List[Tuple[List[str], np.ndarray]] = []
    _find_tensors_nested(raw, [], path_arrays)

    if not path_arrays:
        raise ValueError("No tensors/arrays found in .pt file")

    # Esol/chemprop/chemeleon: epoch -> split (train/val/test) -> "embeddings" -> layer_name -> tensor
    # Use a single split so n_samples is consistent (prefer "test", then "val", then "train")
    if isinstance(raw, dict):
        for split_name in ("test", "val", "train"):
            cells_esol = {}
            epoch_set = set()
            layer_set = set()
            real_ids = None
            smiles_list = None
            for epoch_key, epoch_data in raw.items():
                if not isinstance(epoch_data, dict) or split_name not in epoch_data:
                    continue
                split_data = epoch_data[split_name]
                if not isinstance(split_data, dict) or "embeddings" not in split_data:
                    continue
                emb_dict = split_data["embeddings"]
                if not isinstance(emb_dict, dict):
                    continue
                for layer_key, tensor in emb_dict.items():
                    if hasattr(tensor, "shape"):
                        arr = _flatten_2d(_to_numpy(tensor))
                        cells_esol[(str(epoch_key), str(layer_key))] = arr
                        epoch_set.add(str(epoch_key))
                        layer_set.add(str(layer_key))
                if real_ids is None and "id" in split_data:
                    real_ids = list(split_data["id"])
                # SMILES from same split (aligned with real_ids) for substructure MCS
                smiles_list = split_data.get("smiles") if isinstance(split_data.get("smiles"), (list, tuple)) else None
            if cells_esol:
                n_s = next(iter(cells_esol.values())).shape[0]
                id_to_smiles = None
                if real_ids is not None and smiles_list is not None and len(smiles_list) == n_s:
                    id_to_smiles = dict(zip(real_ids, [str(s) for s in smiles_list]))
                return {
                    "cells": cells_esol,
                    "epoch_keys": sorted(epoch_set),
                    "layer_keys": sorted(layer_set),
                    "n_samples": n_s,
                    "real_ids": real_ids,
                    "id_to_smiles": id_to_smiles,
                }
            # else try next split
        # Fall through to other handling
    if isinstance(raw, dict):
        if "embeddings" in raw and hasattr(raw["embeddings"], "shape"):
            emb = _to_numpy(raw["embeddings"])
            emb = _flatten_2d(emb)
            epochs = raw.get("epochs", raw.get("epoch", ["default"]))
            layers = raw.get("layers", raw.get("layer", ["layer0"]))
            if not isinstance(epochs, (list, tuple)):
                epochs = [str(epochs)]
            if not isinstance(layers, (list, tuple)):
                layers = [str(layers)]
            # Shape might be (n_epochs, n_layers, n_samples, dim) or (n_samples, dim) single
            if emb.ndim == 2:
                # Single block: one epoch, one layer or many samples
                if emb.shape[0] > emb.shape[1]:
                    # (n_samples, dim)
                    cells = {("default", layers[0] if layers else "layer0"): emb}
                    return {
                        "cells": cells,
                        "epoch_keys": ["default"],
                        "layer_keys": layers if layers else ["layer0"],
                        "n_samples": emb.shape[0],
                        "real_ids": raw.get("real_ids"),
                    }
                else:
                    cells = {}
                    for ei, e in enumerate(epochs):
                        for li, l in enumerate(layers):
                            # Assume 4D: (epoch, layer, sample, dim)
                            if emb.ndim == 4 and ei < emb.shape[0] and li < emb.shape[1]:
                                cells[(str(e), str(l))] = _flatten_2d(emb[ei, li])
                            elif emb.ndim == 3 and li < emb.shape[0]:
                                cells[(str(e), str(l))] = _flatten_2d(emb[li])
                    if not cells:
                        cells[("default", "layer0")] = emb
                    return {
                        "cells": cells,
                        "epoch_keys": list(epochs),
                        "layer_keys": list(layers),
                        "n_samples": next(iter(cells.values())).shape[0],
                        "real_ids": raw.get("real_ids"),
                    }
        # Try nested dict: first level = epoch, second = layer (or layer, epoch)
        epoch_keys, layer_keys, cells = _infer_epoch_layer_keys(path_arrays)
        n_samples = next(iter(cells.values())).shape[0] if cells else 0
        return {
            "cells": cells,
            "epoch_keys": epoch_keys,
            "layer_keys": layer_keys,
            "n_samples": n_samples,
            "real_ids": raw.get("real_ids") if isinstance(raw, dict) else None,
        }

    # Generic: use inferred structure from all found tensors
    epoch_keys, layer_keys, cells = _infer_epoch_layer_keys(path_arrays)
    n_samples = next(iter(cells.values())).shape[0] if cells else 0
    return {
        "cells": cells,
        "epoch_keys": epoch_keys,
        "layer_keys": layer_keys,
        "n_samples": n_samples,
        "real_ids": None,
    }


def load_from_directory(
    dir_path: Path,
    layer_file_map: Optional[Dict[str, str]] = None,
    model_name: str = "default",
) -> Dict[str, Any]:
    """
    Load embeddings from a directory of .pt or .npy files.

    layer_file_map: optional dict layer_name -> filename (e.g. {"emb": "emb.pt"}).
    If None, all .pt and .npy in directory are loaded; stem = layer name.
    model_name becomes the single epoch key.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {dir_path}")

    if layer_file_map:
        cells = {}
        for layer_name, filename in layer_file_map.items():
            path = dir_path / filename
            if not path.exists():
                continue
            if path.suffix == ".pt" and _TORCH_AVAILABLE:
                try:
                    t = torch.load(str(path), map_location="cpu", weights_only=False)
                except TypeError:
                    t = torch.load(str(path), map_location="cpu")
                arr = _flatten_2d(_to_numpy(t))
            else:
                arr = _flatten_2d(np.load(str(path)))
            cells[(model_name, layer_name)] = arr
        layer_keys = sorted(set(k[1] for k in cells))
    else:
        cells = {}
        for f in sorted(dir_path.iterdir()):
            if f.suffix not in (".pt", ".npy"):
                continue
            layer_name = f.stem
            if f.suffix == ".pt" and _TORCH_AVAILABLE:
                try:
                    t = torch.load(str(f), map_location="cpu", weights_only=False)
                except TypeError:
                    t = torch.load(str(f), map_location="cpu")
                arr = _flatten_2d(_to_numpy(t))
            else:
                arr = _flatten_2d(np.load(str(f)))
            cells[(model_name, layer_name)] = arr
        layer_keys = sorted(set(k[1] for k in cells))

    if not cells:
        raise ValueError(f"No embedding files found in {dir_path}")

    n_samples = next(iter(cells.values())).shape[0]
    return {
        "cells": cells,
        "epoch_keys": [model_name],
        "layer_keys": layer_keys,
        "n_samples": n_samples,
        "real_ids": None,
    }


def load_embedding_input(
    path: Union[str, Path],
    model_name: Optional[str] = None,
    layer_file_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Main entry: load from a single .pt file or a directory.

    path: path to .pt file or directory.
    model_name: used when path is a directory (epoch key).
    layer_file_map: used when path is a directory (layer_name -> filename).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.is_file():
        if path.suffix.lower() == ".pt":
            return load_single_pt(path)
        raise ValueError("Single file must be .pt")
    else:
        return load_from_directory(
            path,
            layer_file_map=layer_file_map,
            model_name=model_name or path.name or "default",
        )
