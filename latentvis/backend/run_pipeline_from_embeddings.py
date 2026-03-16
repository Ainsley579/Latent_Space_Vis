#!/usr/bin/env python3
"""
Full pipeline: load embedding input (.pt file or directory), run PGMeans clustering,
BIC/AIC, dimensionality reduction (PCA, MDS, UMAP, t-SNE), Sankey (epoch + layer axes),
scatter JSONs, cluster representatives, and optional substructures (RDKit).

Usage:
  python run_pipeline_from_embeddings.py --input /path/to/chemprop-embeddings.pt --output-dir ./out
  python run_pipeline_from_embeddings.py --input /path/to/embedding_dir --output-dir ./out [--model-name my_model]
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fowlkes_mallows_score

# Backend package root
_BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND))

from embedding_loader import load_embedding_input
from dimensionality_reduction import reduce_embedding
from clustering import fit_clustering

# Optional RDKit for substructures
try:
    from rdkit import Chem
    _RDKIT_AVAILABLE = True
    from mcs_common import common_substructure_smarts_smiles
except Exception:
    _RDKIT_AVAILABLE = False
    common_substructure_smarts_smiles = None


def _json_default(obj):
    """Handle numpy types for JSON serialization (Python 3.6 compat)."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Object of type {} is not JSON serializable".format(type(obj).__name__))


def _epoch_sort_key(value):
    s = str(value)
    return (0, int(s), s) if s.isdigit() else (1, s)


def _model_sort_key(value):
    s = str(value).strip()
    lower = s.lower()
    if "noaug" in lower:
        return (0, s)
    if "10conf" in lower:
        return (1, s)
    if "100conf" in lower:
        return (2, s)
    return _epoch_sort_key(s)


def _flatten_embeddings(emb):
    if emb.ndim > 2:
        return emb.reshape(emb.shape[0], -1)
    return emb


def _cluster_representatives(emb: np.ndarray, labels: np.ndarray, k: int = 5) -> dict:
    """Per-cluster: top-k closest/farthest and full list of indices sorted by distance to center (closest first)."""
    from scipy.spatial.distance import cdist
    uniq = np.unique(labels)
    reps = {}
    for c in uniq:
        mask = labels == c
        pts = emb[mask]
        idx_global = np.where(mask)[0]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0, keepdims=True)
        d = cdist(pts, center).ravel()
        order = np.argsort(d)
        # Full sorted order (all members, closest first) for frontend to slice arbitrarily
        sorted_indices = [int(idx_global[order[i]]) for i in range(len(order))]
        n = min(k, len(order))
        closest = [int(idx_global[order[i]]) for i in range(n)]
        closest_set = set(closest)
        farthest = []
        for i in order[::-1]:
            idx = int(idx_global[i])
            if idx in closest_set:
                continue
            farthest.append(idx)
            if len(farthest) >= n:
                break
        reps[int(c)] = {
            "top_k_closest": closest,
            "top_k_farthest": farthest,
            "k": n,
            "sorted_indices": sorted_indices,
        }
    return reps


def _edges_from_transitions(labels_from: np.ndarray, labels_to: np.ndarray, prefix_from: str, prefix_to: str) -> list:
    """Build list of edges with source_label, target_label, value, and node ids."""
    n_from = int(labels_from.max()) + 1
    n_to = int(labels_to.max()) + 1
    label_to_id = {}
    for i in range(n_from):
        label_to_id[prefix_from + "_C" + str(i)] = len(label_to_id)
    for j in range(n_to):
        label_to_id[prefix_to + "_C" + str(j)] = len(label_to_id)
    counts = defaultdict(int)
    for a, b in zip(labels_from.tolist(), labels_to.tolist()):
        counts[(int(a), int(b))] += 1
    edges = []
    for (c0, c1), val in counts.items():
        if val <= 0:
            continue
        s_label = prefix_from + "_C" + str(c0)
        t_label = prefix_to + "_C" + str(c1)
        sid = label_to_id[s_label]
        tid = label_to_id[t_label]
        src_tot = sum(v for (a, _), v in counts.items() if a == c0)
        pct = 100.0 * val / src_tot if src_tot else 0
        edges.append({
            "source_node_id": sid,
            "target_node_id": tid,
            "source_label": s_label,
            "target_label": t_label,
            "value": int(val),
            "pct_of_source": round(pct, 4),
        })
    return edges


def _sankey_sample_rows(
    ca_df: pd.DataFrame,
    epoch_keys: list,
    layer_keys: list,
    real_id_col: str = "real_id",
) -> list:
    """Build list of {real_id, from_layer, to_layer, from_cluster, to_cluster, from_model?, to_model?} for samples."""
    rows = []
    layer_cols = [f"{ly}_cluster" for ly in layer_keys if f"{ly}_cluster" in ca_df.columns]
    epoch_col = "epoch" if "epoch" in ca_df.columns else None
    for i in range(len(ca_df)):
        rid = ca_df.iloc[i][real_id_col]
        for j in range(len(layer_cols) - 1):
            l0, l1 = layer_cols[j], layer_cols[j + 1]
            if l0 not in ca_df.columns or l1 not in ca_df.columns:
                continue
            lay0 = l0.replace("_cluster", "")
            lay1 = l1.replace("_cluster", "")
            try:
                c0 = int(ca_df.iloc[i][l0])
                c1 = int(ca_df.iloc[i][l1])
            except (ValueError, TypeError):
                continue
            r = {
                "real_id": int(rid) if hasattr(rid, "item") else rid,
                "from_layer": lay0,
                "to_layer": lay1,
                "from_cluster": int(c0),
                "to_cluster": int(c1),
            }
            if epoch_col and epoch_col in ca_df.columns:
                r["from_model"] = r["to_model"] = str(ca_df.iloc[i][epoch_col])
            rows.append(r)
    return rows


def _preferred_layer_order(layer_keys: list) -> list:
    """
    Normalize layer traversal order for sankey generation:
    fingerprint -> gnn_layer_1..N -> ffn/fnn_layer_1..N -> remaining (original order).
    """
    ordered_input = [str(x).strip() for x in layer_keys if str(x).strip()]
    layer_set = set(ordered_input)

    def _sort_numeric(names: list, pattern: str) -> list:
        out = []
        for name in names:
            m = re.match(pattern, name)
            if m:
                out.append((int(m.group(1)), name))
        out.sort(key=lambda t: t[0])
        return [name for _, name in out]

    ordered = []
    if "fingerprint" in layer_set:
        ordered.append("fingerprint")

    ordered.extend(_sort_numeric(ordered_input, r"^gnn_layer_(\d+)$"))

    ffn_like = [n for n in ordered_input if re.match(r"^(ffn|fnn)_layer_(\d+)$", n)]
    ffn_like_sorted = sorted(
        ffn_like,
        key=lambda n: int(re.match(r"^(?:ffn|fnn)_layer_(\d+)$", n).group(1))
    )
    for name in ffn_like_sorted:
        if name not in ordered:
            ordered.append(name)

    for name in ordered_input:
        if name not in ordered:
            ordered.append(name)
    return ordered


def run_pipeline(
    input_path,
    output_dir,
    model_name=None,
    layer_file_map=None,
    input_path2=None,
    model_name2=None,
    layer_file_map2=None,
    input_path3=None,
    model_name3=None,
    layer_file_map3=None,
    k_min: int = 2,
    k_max: int = 15,
    default_reduction: str = "mds",
    smiles_csv=None,
    smiles_id_column: str = "real_id",
    smiles_smiles_column: str = "smiles",
    random_state: int = 42,
    clustering_method: str = "pgmeans",
    max_epochs: Optional[int] = None,
    skip_tsne: bool = False,
    n_clusters_override: Optional[int] = None,
    linkage_override: Optional[str] = None,
    knn_k_override: Optional[int] = None,
    reuse_scatter_coords: bool = False,
    substructure_max_members: Optional[int] = None,
    scatter_subdir: str = "scatterplot",
    data_split: Optional[str] = None,
    write_fmi_cspa: bool = True,
    interactive_output_suffix: str = "",
    verbose_dr: bool = True,
    resume: bool = True,
) -> bool:
    """
    Load embedding input (optionally multiple directories for multiple epochs/models),
    run full analysis, write all frontend JSONs.

    default_reduction: method used for the default scatter file (no suffix); one of pca, mds, umap, tsne.
    If input_path2/input_path3 are set, their cells are merged into the same analysis run.
    interactive_output_suffix: when set (e.g. "_current"), write topk, sankey, substructures,
    clustering_metrics to topk_current, sankey_current, etc., so original artifacts are not overwritten.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    print("Loading embedding input...")
    data = load_embedding_input(
        input_path,
        model_name=model_name,
        layer_file_map=layer_file_map,
        data_split=data_split,
    )
    cells = dict(data["cells"])
    epoch_keys = list(data["epoch_keys"])
    layer_keys = list(data["layer_keys"])
    layer_keys = _preferred_layer_order(layer_keys)
    n_samples = data["n_samples"]
    real_ids = data.get("real_ids")
    id_to_smiles_from_input = data.get("id_to_smiles") or {}

    extra_inputs = [
        (input_path2, model_name2, layer_file_map2, "second"),
        (input_path3, model_name3, layer_file_map3, "third"),
    ]
    for extra_input_path, extra_model_name, extra_layer_map, ordinal in extra_inputs:
        if not extra_input_path or not extra_model_name:
            continue
        data_extra = load_embedding_input(
            extra_input_path,
            model_name=extra_model_name,
            layer_file_map=extra_layer_map,
            data_split=data_split,
        )
        if data_extra["n_samples"] != n_samples:
            print(
                f"Error: {ordinal} input n_samples={data_extra['n_samples']} != first n_samples={n_samples}"
            )
            return False
        cells.update(data_extra["cells"])
        epoch_keys.extend(data_extra["epoch_keys"])
        for lk in data_extra["layer_keys"]:
            if lk not in layer_keys:
                layer_keys.append(lk)
        print(f"Merged {ordinal} epoch: {extra_model_name}")
    if real_ids is None:
        real_ids = list(range(n_samples))

    epoch_keys_sorted = sorted(dict.fromkeys(epoch_keys), key=_model_sort_key)
    epoch_keys = epoch_keys_sorted

    if max_epochs is not None and max_epochs > 0:
        keep_epochs = set(epoch_keys_sorted[: max_epochs])
        cells = {(ek, lk): v for (ek, lk), v in cells.items() if ek in keep_epochs}
        epoch_keys = [e for e in epoch_keys_sorted if e in keep_epochs]
        print(f"Limited to {len(epoch_keys)} epochs (--max-epochs {max_epochs})")
    print(f"Epoch keys: {epoch_keys}, Layer keys: {layer_keys}, n_samples: {n_samples}")

    # Clustering methods to run: default pgmeans; if clustering_method=="all" then pgmeans, hdbscan, agglomerative, knn
    if clustering_method == "all":
        run_methods = ["pgmeans", "hdbscan", "agglomerative", "knn"]
    else:
        run_methods = [clustering_method]

    # Per-cell results: (epoch, layer) -> { method -> {labels, k, coords, bic, aic, reps} }
    results = {}
    cluster_assignments = []  # list of dicts: real_id, epoch, layer_0_cluster, ... (use default method for CA)
    run_signature = {
        "clustering_method": clustering_method,
        "run_methods": run_methods,
        "k_min": int(k_min),
        "k_max": int(k_max),
        "skip_tsne": bool(skip_tsne),
        "n_clusters_override": n_clusters_override,
        "linkage_override": linkage_override,
        "data_split": data_split,
        "max_epochs": max_epochs,
        "n_samples": int(n_samples),
        "random_state": int(random_state),
    }
    signature_id = hashlib.sha1(
        json.dumps(run_signature, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()[:12]
    resume_dir = out / ".pipeline_resume" / signature_id
    if resume:
        resume_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(epoch_key: str, layer_key: str) -> Path:
        return resume_dir / "{}__{}.json".format(str(epoch_key), str(layer_key))

    def _load_checkpoint(epoch_key: str, layer_key: str, emb: np.ndarray, dr_list: list):
        if not resume:
            return None
        p = _checkpoint_path(epoch_key, layer_key)
        if not p.is_file():
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        coords_raw = payload.get("coords") or {}
        methods_raw = payload.get("methods") or {}
        if not isinstance(coords_raw, dict) or not isinstance(methods_raw, dict):
            return None
        out_cell = {"coords": {}}
        for dr_name in dr_list:
            arr = coords_raw.get(dr_name)
            if arr is None:
                return None
            arr_np = np.asarray(arr, dtype=np.float64)
            if arr_np.shape != (n_samples, 2):
                return None
            out_cell["coords"][dr_name] = arr_np
        for cl_method in run_methods:
            item = methods_raw.get(cl_method)
            if not isinstance(item, dict):
                return None
            labels = np.asarray(item.get("labels"), dtype=np.int64)
            if labels.shape[0] != n_samples:
                return None
            reps = _cluster_representatives(emb, labels, k=5)
            out_cell[cl_method] = {
                "labels": labels,
                "k": int(item.get("k", -1)),
                "k_values": item.get("k_values", []),
                "bic": item.get("bic", []),
                "aic": item.get("aic", []),
                "reps": reps,
            }
        return out_cell

    def _write_checkpoint(epoch_key: str, layer_key: str, cell_result: dict):
        if not resume:
            return
        p = _checkpoint_path(epoch_key, layer_key)
        payload = {"coords": {}, "methods": {}}
        for dr_name, coord in (cell_result.get("coords") or {}).items():
            payload["coords"][dr_name] = np.asarray(coord).tolist()
        for cl_method in run_methods:
            item = cell_result.get(cl_method) or {}
            payload["methods"][cl_method] = {
                "labels": np.asarray(item.get("labels", []), dtype=np.int64).tolist(),
                "k": int(item.get("k", -1)),
                "k_values": item.get("k_values", []),
                "bic": item.get("bic", []),
                "aic": item.get("aic", []),
            }
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _scatter_source_dirs(root: Path) -> list:
        """Candidate dirs where scatterplot JSONs may already exist (for coordinate reuse)."""
        dirs = []
        # Baseline scatter directories are always named "scatterplot" under each clustering method.
        d0 = root / "scatterplot"
        if d0.is_dir():
            dirs.append(d0)
        clustering_dir = root / "clustering"
        if clustering_dir.is_dir():
            for sub in clustering_dir.iterdir():
                d = sub / "scatterplot"
                if d.is_dir():
                    dirs.append(d)
        return dirs

    def _load_coords_from_scatter_json(scatter_dir: Path, model_layer: str, dr_name: str, real_ids_list: list):
        """
        Read existing scatter JSON and return Nx2 numpy array in real_ids order.
        File format: list of {real_id, x, y, cluster_id?}.
        """
        p = scatter_dir / "{}_{}.json".format(model_layer, dr_name)
        if not p.is_file():
            return None
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(rows, list) or not rows:
            return None
        xy_by_id = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            rid = r.get("real_id")
            if rid is None:
                continue
            try:
                x = float(r.get("x"))
                y = float(r.get("y"))
            except (TypeError, ValueError):
                continue
            xy_by_id[str(rid)] = (x, y)
        if not xy_by_id:
            return None
        arr = np.zeros((len(real_ids_list), 2), dtype=np.float32)
        for i, rid in enumerate(real_ids_list):
            xy = xy_by_id.get(str(rid))
            if xy is None:
                return None
            arr[i, 0] = xy[0]
            arr[i, 1] = xy[1]
        return arr

    for (ek, lk), emb in cells.items():
        emb = _flatten_embeddings(emb)
        if emb.shape[0] != n_samples:
            print(f"  Skip ({ek}, {lk}): shape {emb.shape} != n_samples")
            continue
        print(f"  Processing ({ek}, {lk}) ...", flush=True)
        dr_list = ["pca", "mds", "umap"] if skip_tsne else ["pca", "mds", "umap", "tsne"]
        resumed_cell = _load_checkpoint(ek, lk, emb, dr_list)
        if resumed_cell is not None:
            if skip_tsne and "tsne" not in resumed_cell["coords"]:
                resumed_cell["coords"]["tsne"] = resumed_cell["coords"]["pca"].copy()
            results[(ek, lk)] = resumed_cell
            print("    resumed from checkpoint", flush=True)
            default_labels = results[(ek, lk)][run_methods[0]]["labels"]
            for i in range(n_samples):
                rid = real_ids[i] if i < len(real_ids) else i
                row = {"real_id": rid, "epoch": ek, f"{lk}_cluster": int(default_labels[i])}
                cluster_assignments.append(row)
            continue
        coords = {}
        model_layer = "{}_{}".format(ek, lk)
        if reuse_scatter_coords:
            # Reuse precomputed scatter coordinates if present, to avoid expensive DR recomputation.
            loaded_all = True
            for dr_method in dr_list:
                if dr_method == "tsne" and skip_tsne:
                    continue
                found = None
                for sd in _scatter_source_dirs(out):
                    found = _load_coords_from_scatter_json(sd, model_layer, dr_method, real_ids)
                    if found is not None:
                        break
                if found is None:
                    loaded_all = False
                    break
                coords[dr_method] = found
            if loaded_all:
                print("    scatter coords reused (no DR recompute)", flush=True)
        if not coords:
            for dr_method in dr_list:
                try:
                    coords[dr_method] = reduce_embedding(emb, method=dr_method, random_state=random_state)
                    print(f"    {dr_method} ok", flush=True)
                except Exception as e:
                    print(f"    {dr_method} failed: {e}", flush=True)
                    coords[dr_method] = reduce_embedding(emb, method="pca", random_state=random_state)
                    print(f"    {dr_method} -> pca fallback", flush=True)
        if skip_tsne:
            coords["tsne"] = coords["pca"].copy()
            print(f"    tsne skipped (using pca)", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        results[(ek, lk)] = {"coords": coords}
        for cl_method in run_methods:
            if cl_method == "agglomerative" and (n_clusters_override is not None or linkage_override is not None):
                from clustering import fit_agglomerative
                cl_out = fit_agglomerative(
                    emb,
                    n_clusters=n_clusters_override,
                    k_min=k_min,
                    k_max=k_max,
                    random_state=random_state,
                    linkage=linkage_override or "ward",
                )
            elif cl_method == "knn":
                knn_k = knn_k_override if knn_k_override is not None else max(2, int(k_max))
                cl_out = fit_clustering(
                    emb,
                    method=cl_method,
                    k_min=k_min,
                    k_max=k_max,
                    random_state=random_state,
                    n_neighbors=knn_k,
                )
            else:
                cl_out = fit_clustering(
                    emb, method=cl_method, k_min=k_min, k_max=k_max, random_state=random_state
                )
            labels = cl_out["labels"]
            reps = _cluster_representatives(emb, labels, k=5)
            results[(ek, lk)][cl_method] = {
                "labels": labels,
                "k": cl_out["k"],
                "k_values": cl_out["k_values"],
                "bic": cl_out["bic"],
                "aic": cl_out["aic"],
                "reps": reps,
            }
        # Default method for cluster_assignments (first in run_methods)
        default_labels = results[(ek, lk)][run_methods[0]]["labels"]
        for i in range(n_samples):
            rid = real_ids[i] if i < len(real_ids) else i
            row = {"real_id": rid, "epoch": ek, f"{lk}_cluster": int(default_labels[i])}
            cluster_assignments.append(row)
        _write_checkpoint(ek, lk, results[(ek, lk)])

    # Build cluster_assignments DataFrame: one row per (real_id, epoch) with columns layer0_cluster, layer1_cluster, ...
    rows_by_key = defaultdict(dict)
    for row in cluster_assignments:
        key = (row["real_id"], row["epoch"])
        for k, v in row.items():
            if k not in ("real_id", "epoch"):
                rows_by_key[key][k] = v
        rows_by_key[key]["real_id"] = row["real_id"]
        rows_by_key[key]["epoch"] = row["epoch"]
    ca_df = pd.DataFrame([rows_by_key[k] for k in sorted(rows_by_key)])

    def res(ek, lk, method):
        return results.get((ek, lk), {}).get(method, {})

    real_id_to_idx = {real_ids[i]: i for i in range(len(real_ids))}

    for lk in layer_keys:
        col = f"{lk}_cluster"
        if col not in ca_df.columns and results:
            vals = []
            for _, row in ca_df.iterrows():
                ek, rid = row["epoch"], row["real_id"]
                lab = res(ek, lk, run_methods[0]).get("labels")
                if lab is None:
                    vals.append(-1)
                else:
                    i = real_id_to_idx.get(rid)
                    vals.append(int(lab[i]) if i is not None else -1)
            ca_df[col] = vals

    # Write shared baseline lookup only for baseline runs (no interactive suffix).
    if id_to_smiles_from_input and not interactive_output_suffix:
        smiles_out = out / "real_id_to_smiles.json"
        smiles_out.write_text(
            json.dumps({str(k): v for k, v in id_to_smiles_from_input.items()}, indent=2),
            encoding="utf-8",
        )
        print("Wrote {} ({} entries) for frontend molecule lookup.".format(smiles_out, len(id_to_smiles_from_input)))

    # ---- Write outputs per clustering method (root for first method, clustering/<method>/ for others) ----
    for cl_method in run_methods:
        base = out if cl_method == run_methods[0] else (out / "clustering" / cl_method)
        scatter_dir = base / scatter_subdir
        scatter_dir.mkdir(parents=True, exist_ok=True)
        for (ek, lk), cell in results.items():
            coords = cell.get("coords", {})
            r = res(ek, lk, cl_method)
            labels = r.get("labels")
            if labels is None:
                continue
            model_layer = "{}_{}".format(ek, lk)
            for dr_name, coord in coords.items():
                arr = coord if hasattr(coord, "tolist") else np.asarray(coord)
                points = []
                for i in range(len(arr)):
                    rid = real_ids[i] if i < len(real_ids) else i
                    points.append({
                        "real_id": str(rid),
                        "x": float(arr[i, 0]),
                        "y": float(arr[i, 1]),
                        "cluster_id": int(labels[i]),
                    })
                (scatter_dir / "{}_{}.json".format(model_layer, dr_name)).write_text(json.dumps(points, indent=2))
            default_coord = coords.get(default_reduction, coords.get("pca"))
            if default_coord is not None:
                default_points = []
                for i in range(len(default_coord)):
                    rid = real_ids[i] if i < len(real_ids) else i
                    default_points.append({
                        "real_id": str(rid),
                        "x": float(default_coord[i, 0]),
                        "y": float(default_coord[i, 1]),
                        "cluster_id": int(labels[i]),
                    })
                (scatter_dir / "{}.json".format(model_layer)).write_text(json.dumps(default_points, indent=2))
        bic_dir = base / ("bic_aic" + interactive_output_suffix)
        bic_dir.mkdir(parents=True, exist_ok=True)
        for (ek, lk) in results:
            r = res(ek, lk, cl_method)
            if not r:
                continue
            payload = {
                "model": ek,
                "layer": lk,
                "k_values": r.get("k_values", []),
                "aic": r.get("aic", []),
                "bic": r.get("bic", []),
            }
            (bic_dir / "{}_{}.json".format(ek, lk)).write_text(json.dumps(payload, indent=2))
        topk_list = []
        sorted_members_list = []
        for (ek, lk) in results:
            r = res(ek, lk, cl_method)
            reps = r.get("reps", {})
            for cid, rep in reps.items():
                sorted_real_ids = [real_ids[i] if i < len(real_ids) else i for i in rep["sorted_indices"]]
                topk_list.append({
                    "layer_name": lk,
                    "model_type": ek,
                    "cluster_id": cid,
                    "top_k_closest_real_ids": [real_ids[i] if i < len(real_ids) else i for i in rep["top_k_closest"]],
                    "top_k_farthest_real_ids": [real_ids[i] if i < len(real_ids) else i for i in rep["top_k_farthest"]],
                    "k": rep["k"],
                })
                sorted_members_list.append({
                    "layer_name": lk,
                    "model_type": ek,
                    "cluster_id": cid,
                    "sorted_real_ids": sorted_real_ids,
                    "count": len(sorted_real_ids),
                })
        topk_dir = base / ("topk" + interactive_output_suffix)
        topk_dir.mkdir(parents=True, exist_ok=True)
        (topk_dir / "cluster_representatives.json").write_text(json.dumps(topk_list, indent=2))
        (topk_dir / "cluster_sorted_members.json").write_text(json.dumps(sorted_members_list, indent=2))
        # all_sankey key order: epoch_keys and layer_keys (model order). Frontend reads layer order from this JSON.
        all_sankey = {"epoch": {}, "layer": {}}
        for ek in epoch_keys:
            edges = []
            for j in range(len(layer_keys) - 1):
                l0, l1 = layer_keys[j], layer_keys[j + 1]
                r0, r1 = res(ek, l0, cl_method), res(ek, l1, cl_method)
                lab0, lab1 = r0.get("labels"), r1.get("labels")
                if lab0 is None or lab1 is None:
                    continue
                edges.extend(_edges_from_transitions(lab0, lab1, l0, l1))
            if edges:
                all_sankey["epoch"][ek] = {"type": "spatial_epoch", "epoch": ek, "edges": edges}
        for lk in layer_keys:
            edges = []
            for i in range(len(epoch_keys) - 1):
                e0, e1 = epoch_keys[i], epoch_keys[i + 1]
                r0, r1 = res(e0, lk, cl_method), res(e1, lk, cl_method)
                lab0, lab1 = r0.get("labels"), r1.get("labels")
                if lab0 is None or lab1 is None:
                    continue
                edges.extend(_edges_from_transitions(lab0, lab1, str(e0), str(e1)))
            if edges:
                all_sankey["layer"][lk] = {"type": "spatial_layer", "layer": lk, "edges": edges}
        sankey_dir = base / ("sankey" + interactive_output_suffix)
        sankey_dir.mkdir(parents=True, exist_ok=True)
        (sankey_dir / "all_sankey.json").write_text(json.dumps(all_sankey, indent=2))
        ca_df_method = ca_df.copy()
        for lk in layer_keys:
            col = "{}_cluster".format(lk)
            vals = []
            for _, row in ca_df_method.iterrows():
                ek, rid = row["epoch"], row["real_id"]
                lab = res(ek, lk, cl_method).get("labels")
                if lab is None:
                    vals.append(-1)
                else:
                    i = real_id_to_idx.get(rid)
                    vals.append(int(lab[i]) if i is not None else -1)
            ca_df_method[col] = vals
        sample_rows = _sankey_sample_rows(ca_df_method, epoch_keys, layer_keys)
        (sankey_dir / "sankey_samples.json").write_text(
            json.dumps(sample_rows, indent=2, default=_json_default)
        )
        cm_dir = base / ("clustering_metrics" + interactive_output_suffix)
        cm_dir.mkdir(parents=True, exist_ok=True)
        ca_df_method.to_csv(cm_dir / "cluster_assignments_with_real_id.csv", index=False)
        # Substructure (MCS): use SMILES from CSV if provided, else from .pt (id_to_smiles_from_input)
        id_to_smiles = {}
        if smiles_csv and Path(smiles_csv).exists():
            smiles_df = pd.read_csv(smiles_csv)
            if smiles_id_column in smiles_df.columns and smiles_smiles_column in smiles_df.columns:
                id_to_smiles = dict(zip(smiles_df[smiles_id_column].astype(str), smiles_df[smiles_smiles_column].astype(str)))
                for k, v in zip(smiles_df[smiles_id_column].astype(int), smiles_df[smiles_smiles_column].astype(str)):
                    id_to_smiles[k] = v
        if not id_to_smiles and id_to_smiles_from_input:
            id_to_smiles = id_to_smiles_from_input
        if _RDKIT_AVAILABLE and id_to_smiles:
            def _get_smi(rid):
                v = id_to_smiles.get(rid)
                if v is not None:
                    return v
                v = id_to_smiles.get(str(rid))
                if v is not None:
                    return v
                try:
                    return id_to_smiles.get(int(rid))
                except (TypeError, ValueError):
                    return None
            # Map (model_type, layer_name, cluster_id) -> full sorted real_ids (closest first) for flexible M selection
            cluster_sorted_map = {}
            for sm in sorted_members_list:
                key = (sm.get("model_type"), sm.get("layer_name"), sm.get("cluster_id"))
                if key[0] is None or key[1] is None or key[2] is None:
                    continue
                if isinstance(sm.get("sorted_real_ids"), list):
                    cluster_sorted_map[key] = list(sm["sorted_real_ids"])

            substruct_list = []
            for entry in topk_list:
                key_tuple = (entry.get("model_type"), entry.get("layer_name"), entry.get("cluster_id"))
                base_sorted = cluster_sorted_map.get(key_tuple)
                for group, key in [("knn", "top_k_closest_real_ids"), ("kfn", "top_k_farthest_real_ids")]:
                    # Determine candidate real_ids for MCS:
                    # - If substructure_max_members is set and we have full sorted_real_ids, use first M (nearest)
                    #   for knn and last M (farthest) for kfn.
                    # - Otherwise fall back to the top-k lists in topk_list entry.
                    if substructure_max_members is not None and substructure_max_members > 0 and base_sorted:
                        m = min(substructure_max_members, len(base_sorted))
                        if group == "knn":
                            rids = base_sorted[:m]
                        else:  # kfn
                            rids = list(reversed(base_sorted))[:m]
                    else:
                        rids = entry[key]
                    mols = []
                    missing = []
                    invalid = []
                    seen_canonical = set()
                    for rid in rids:
                        smi = _get_smi(rid)
                        if not smi:
                            missing.append(rid)
                            continue
                        mol = Chem.MolFromSmiles(smi)
                        if mol is None:
                            invalid.append(rid)
                            continue
                        canonical = Chem.MolToSmiles(mol, canonical=True)
                        if canonical in seen_canonical:
                            continue
                        seen_canonical.add(canonical)
                        mols.append(mol)
                    smiles_list = [Chem.MolToSmiles(m, canonical=True) for m in mols]
                    if common_substructure_smarts_smiles:
                        mcs_smarts, mcs_smiles = common_substructure_smarts_smiles(
                            smiles_list, timeout_s=30, ring_matches_ring_only=True, complete_rings_only=True,
                            match_valences=True, atom_compare="elements", bond_compare="order",
                        )
                    else:
                        mcs_smarts, mcs_smiles = None, None
                    # When MCS is null (diverse molecules), frontend can show this instead of blank
                    mcs_note = None
                    if mcs_smarts is None and mcs_smiles is None and len(mols) > 0:
                        mcs_note = "No common substructure found"
                    elif missing:
                        mcs_note = "Missing SMILES for {} real_id(s)".format(len(missing))
                    substruct_list.append({
                        "layer_name": entry["layer_name"], "model_type": entry["model_type"], "cluster_id": entry["cluster_id"],
                        "group": group, "k": entry["k"], "real_ids": rids, "valid_molecules": len(mols),
                        "missing_real_ids": missing, "invalid_smiles_real_ids": invalid,
                        "mcs_smarts": mcs_smarts, "mcs_smiles": mcs_smiles, "mcs_note": mcs_note,
                    })
            sub_dir = base / ("substructures" + interactive_output_suffix)
            sub_dir.mkdir(parents=True, exist_ok=True)
            (sub_dir / "cluster_substructures.json").write_text(json.dumps(substruct_list, indent=2))
        print("Wrote outputs for clustering method: {} -> {}".format(cl_method, base))

    if not _RDKIT_AVAILABLE and (id_to_smiles_from_input or (smiles_csv and Path(smiles_csv).exists())):
        print("RDKit not available; skipping substructures.")
    elif not id_to_smiles_from_input and (not smiles_csv or not Path(smiles_csv).exists()):
        print("No SMILES CSV or .pt smiles provided; skipping substructures.")

    # FMI/CSPA summary (default clustering method only).
    # This can be expensive (O(n^2) per epoch pair); allow callers to disable it.
    if write_fmi_cspa:
        fmi_cspa_layers = []
        for lk in layer_keys:
            fmi_vals, cspa_vals = [], []
            for i in range(len(epoch_keys) - 1):
                e0, e1 = epoch_keys[i], epoch_keys[i + 1]
                lab0 = res(e0, lk, run_methods[0]).get("labels")
                lab1 = res(e1, lk, run_methods[0]).get("labels")
                if lab0 is None or lab1 is None:
                    continue
                try:
                    fmi_vals.append(fowlkes_mallows_score(lab0, lab1))
                    sim0 = (lab0[:, None] == lab0[None, :]).astype(float)
                    sim1 = (lab1[:, None] == lab1[None, :]).astype(float)
                    cspa_vals.append(float(np.linalg.norm((sim0 + sim1) / 2.0, ord="fro")))
                except Exception:
                    pass
            if fmi_vals or cspa_vals:
                fmi_cspa_layers.append({
                    "layer": lk,
                    "avg_fmi": float(np.mean(fmi_vals)) if fmi_vals else None,
                    "avg_cspa": float(np.mean(cspa_vals)) if cspa_vals else None,
                    "count": len(fmi_vals),
                })
        if fmi_cspa_layers:
            fmi_dir = out / ("bic_aic" + interactive_output_suffix)
            fmi_dir.mkdir(parents=True, exist_ok=True)
            (fmi_dir / "avg_fmi_cspa_by_layer.json").write_text(
                json.dumps({"layers": fmi_cspa_layers, "models": epoch_keys}, indent=2)
            )
            print("Wrote avg_fmi_cspa_by_layer.json (FMI/CSPA between consecutive epochs).")

    return True


def main():
    ap = argparse.ArgumentParser(description="Run full pipeline from embedding .pt or directory")
    ap.add_argument("--input", "-i", required=True, help="Path to .pt file or directory of .pt/.npy")
    ap.add_argument("--output-dir", "-o", default="./pipeline_output", help="Output directory for all JSONs")
    ap.add_argument("--model-name", "-m", default=None, help="Model/epoch name when input is directory")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=15)
    ap.add_argument("--default-reduction", choices=["pca", "mds", "umap", "tsne"], default="mds")
    ap.add_argument("--smiles-csv", default=None, help="CSV with real_id and SMILES for substructure MCS")
    ap.add_argument("--smiles-id-column", default="real_id")
    ap.add_argument("--smiles-smiles-column", default="smiles")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--layer-file-map", default=None, help="Path to JSON file: layer_name -> filename (e.g. {\"emb\": \"emb.pt\"})")
    ap.add_argument("--input2", default=None, help="Second embedding dir (for second epoch, e.g. dimenet10conf)")
    ap.add_argument("--model-name2", default=None, help="Model name for second input")
    ap.add_argument("--layer-file-map2", default=None, help="JSON path for second input layer map")
    ap.add_argument("--input3", default=None, help="Third embedding dir (for a third epoch/model)")
    ap.add_argument("--model-name3", default=None, help="Model name for third input")
    ap.add_argument("--layer-file-map3", default=None, help="JSON path for third input layer map")
    ap.add_argument("--clustering-method", default="pgmeans", choices=["pgmeans", "hdbscan", "agglomerative", "knn", "all"],
                    help="Clustering method: pgmeans (default), hdbscan, agglomerative, knn, or all")
    ap.add_argument(
        "--agg-k",
        type=int,
        default=None,
        help="Fixed number of clusters for agglomerative clustering.",
    )
    ap.add_argument(
        "--agg-linkage",
        choices=["ward", "complete", "average"],
        default="ward",
        help="Linkage for agglomerative clustering.",
    )
    ap.add_argument(
        "--knn-k",
        type=int,
        default=None,
        help="K for KNN clustering (number of neighbors). Used when --clustering-method knn (or all).",
    )
    ap.add_argument("--max-epochs", type=int, default=None, help="Limit to first N epochs (e.g. 10 for quick run)")
    ap.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE (use PCA for tsne coords); much faster")
    ap.add_argument(
        "--substructure-max-members",
        type=int,
        default=None,
        help="Max number of nearest/farthest members per cluster to use for substructure MCS (default: all available).",
    )
    ap.add_argument(
        "--output-to-frontend",
        action="store_true",
        help="Write directly to frontend/public/data instead of copying later",
    )
    ap.add_argument(
        "--data-split",
        default=None,
        choices=["train", "val", "test"],
        help="When input is an ESOL-style .pt (chemprop/chemeleon), restrict to a single split; "
             "default keeps existing preference order test→val→train.",
    )
    ap.add_argument(
        "--no-fmi-cspa",
        action="store_true",
        help="Skip expensive FMI/CSPA summary computation (avg_fmi_cspa_by_layer.json).",
    )
    ap.add_argument(
        "--quiet-dr",
        action="store_true",
        help="Reduce per-cell dimensionality reduction logging (e.g. for interactive runs).",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable per-cell checkpoint resume support.",
    )
    ap.add_argument(
        "--reuse-scatter-coords",
        action="store_true",
        help="Reuse existing scatterplot JSON coordinates from the output dir when available.",
    )
    args = ap.parse_args()

    output_dir = args.output_dir
    if args.output_to_frontend:
        backend_root = Path(__file__).resolve().parent
        output_dir = str(backend_root.parent / "frontend" / "public" / "data")
        print("Output to frontend: {}".format(output_dir))

    layer_file_map = None
    if args.layer_file_map:
        with open(args.layer_file_map) as f:
            layer_file_map = json.load(f)
    layer_file_map2 = None
    if args.layer_file_map2:
        with open(args.layer_file_map2) as f:
            layer_file_map2 = json.load(f)
    layer_file_map3 = None
    if args.layer_file_map3:
        with open(args.layer_file_map3) as f:
            layer_file_map3 = json.load(f)

    success = run_pipeline(
        input_path=args.input,
        output_dir=output_dir,
        model_name=args.model_name,
        k_min=args.k_min,
        k_max=args.k_max,
        default_reduction=args.default_reduction,
        smiles_csv=args.smiles_csv,
        smiles_id_column=args.smiles_id_column,
        smiles_smiles_column=args.smiles_smiles_column,
        random_state=args.random_state,
        layer_file_map=layer_file_map,
        input_path2=args.input2,
        model_name2=args.model_name2,
        layer_file_map2=layer_file_map2,
        input_path3=args.input3,
        model_name3=args.model_name3,
        layer_file_map3=layer_file_map3,
        clustering_method=args.clustering_method,
        n_clusters_override=args.agg_k,
        linkage_override=args.agg_linkage,
        knn_k_override=args.knn_k,
        max_epochs=args.max_epochs,
        skip_tsne=args.skip_tsne,
        substructure_max_members=args.substructure_max_members,
        data_split=args.data_split,
        write_fmi_cspa=not args.no_fmi_cspa,
        verbose_dr=not getattr(args, "quiet_dr", False),
        resume=not args.no_resume,
        reuse_scatter_coords=bool(args.reuse_scatter_coords),
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
