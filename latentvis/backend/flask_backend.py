#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

from flask import Flask, jsonify, request, send_file


# Backend dir on path for embedding_loader / clustering
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

app = Flask(__name__)
GENERATE_RANGE_JOBS: Dict[str, Dict[str, object]] = {}
GENERATE_RANGE_JOBS_LOCK = threading.Lock()
MAX_GENERATE_RANGE_JOBS = 64
SCATTER_CELL_CACHE: Dict[str, Dict[str, object]] = {}
SCATTER_CELL_CACHE_LOCK = threading.Lock()
MAX_SCATTER_CELL_CACHE = 256


def _resolve_project_root() -> Path:
    env_root = os.environ.get("LATENTVIS_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()

    resolved = Path(__file__).resolve()
    # Local dev layout:
    #   <repo>/latentvis/backend/flask_backend.py
    # Container layout may be:
    #   /app/flask_backend.py
    if len(resolved.parents) >= 3:
        return resolved.parents[2]
    return _BACKEND_DIR


def _resolve_data_root(project_root: Path) -> Path:
    env_data_root = os.environ.get("LATENTVIS_DATA_ROOT", "").strip()
    if env_data_root:
        return Path(env_data_root).resolve()

    candidates = [
        _BACKEND_DIR / "data",
        project_root / "latentvis" / "frontend" / "public" / "data",
        _BACKEND_DIR.parent / "frontend" / "public" / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


PROJECT_ROOT = _resolve_project_root()
DATA_ROOT = _resolve_data_root(PROJECT_ROOT)
DEFAULT_CASES = ("chemprop", "chemeleon")
DEFAULT_CLUSTERING = "pgmeans"

# Embedding .pt paths (override with env LATENTVIS_CHEMPROP_PT / LATENTVIS_CHEMELEON_PT)
_DEFAULT_PT_ROOT = Path("/global/cfs/projectdirs/m636/nhadler/projects/esol/data")
_LOCAL_PT_ROOT = _BACKEND_DIR / "data" / "embeddings"


def _default_embedding_path(case: str) -> Path:
    local = _LOCAL_PT_ROOT / f"{case}-embeddings.pt"
    if local.is_file():
        return local
    return _DEFAULT_PT_ROOT / f"{case}-embeddings.pt"


EMBEDDING_PATHS = {
    "chemprop": Path(os.environ.get("LATENTVIS_CHEMPROP_PT", str(_default_embedding_path("chemprop")))),
    "chemeleon": Path(os.environ.get("LATENTVIS_CHEMELEON_PT", str(_default_embedding_path("chemeleon")))),
}


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


def _normalize_smiles(s: str) -> str:
    return (s or "").strip()


def _normalize_split(value: Optional[str]) -> Optional[str]:
    """Normalize split value to train/val/test or None."""
    if not value:
        return None
    v = str(value).strip().lower()
    if v not in ("train", "val", "test"):
        return None
    return v


def _expand_split_selector(value: Optional[str]) -> List[str]:
    """
    Expand split selector to concrete split list.
    Supports: train/test/val, train+test, test+train, all.
    For this app, "all" means train+test.
    """
    if value is None:
        return ["test"]
    raw = str(value).strip().lower()
    if not raw:
        return ["test"]
    if raw == "all":
        return ["train", "test"]
    if "+" in raw:
        out: List[str] = []
        for token in [p.strip() for p in raw.split("+") if p.strip()]:
            sp = _normalize_split(token)
            if sp and sp not in out:
                out.append(sp)
        return out
    sp = _normalize_split(raw)
    return [sp] if sp else []


def _resolve_train_test_splits(value: Optional[str], default: str = "test") -> List[str]:
    """
    Resolve split selector for APIs backed by train/test artifacts.
    Accepts all/train/test/train+test/test+train.
    """
    raw = "" if value is None else str(value).strip().lower()
    if not raw:
        return [default]
    if raw == "all":
        return ["train", "test"]
    if raw in ("train+test", "test+train"):
        return ["train", "test"] if raw.startswith("train") else ["test", "train"]
    sp = _normalize_split(raw)
    if sp in ("train", "test"):
        return [sp]
    return [default]


def _case_dir(case: str, clustering: Optional[str] = None, data_split: Optional[str] = None) -> Path:
    """
    Resolve case directory for a given dataset, clustering method, and data split.

    Paths are regularized: both test and train live under a split segment.
    - DATA_ROOT/<case>/test/...  (default when split is None or "test")
    - DATA_ROOT/<case>/train/...
    """
    method = (clustering or DEFAULT_CLUSTERING).strip().lower()
    split = _normalize_split(data_split) or "test"
    method_dir = "pgmeans" if method in ("pgmeans", "gmm") else method

    # Preferred split-aware layout (new):
    #   /data/<case>/<split>/clustering/<method>/
    split_aware_clustering = DATA_ROOT / case / split / "clustering" / method_dir
    if split_aware_clustering.exists():
        return split_aware_clustering

    # Alternate split-aware layout:
    #   /data/<case>/<split>/<method>/
    split_aware_flat = DATA_ROOT / case / split / method_dir
    if split_aware_flat.exists():
        return split_aware_flat

    # "pgmeans" old layout fallback:
    #   /data/<case>/<split>/
    if method in ("pgmeans", "gmm"):
        return DATA_ROOT / case / split

    # Non-pgmeans methods are split-aware in current frontend data layout:
    #   /data/<case>/<split>/clustering/<method>/...
    # Keep compatibility with older unsplit layout:
    #   /data/<case>/clustering/<method>/...
    split_aware = DATA_ROOT / case / split / "clustering" / method_dir
    legacy = DATA_ROOT / case / "clustering" / method_dir
    if split_aware.exists() or not legacy.exists():
        return split_aware
    return legacy


def _ordered_models(models: Iterable[str]) -> List[str]:
    unique = []
    seen = set()
    for m in models:
        mm = str(m).strip()
        if not mm or mm in seen:
            continue
        seen.add(mm)
        unique.append(mm)

    def rank(name: str) -> Tuple[int, object]:
        if name.isdigit():
            return (0, int(name))
        return (1, name)

    return sorted(unique, key=rank)


def _ordered_layers(layers: Iterable[str]) -> List[str]:
    preferred = [
        "ffn_layer_1",
        "ffn_layer_2",
        "fingerprint",
        "gnn_layer_1",
        "gnn_layer_2",
        "gnn_layer_3",
        "gnn_layer_4",
        "gnn_layer_5",
        "gnn_layer_6",
        "emb",
        "output_block_0",
        "output_block_1",
        "output_block_2",
        "output_block_3",
        "output_block_4",
        "u_mlp_input",
        "u_mlp_l1",
        "u_mlp_l2",
    ]
    layer_set = {str(x).strip() for x in layers if str(x).strip()}
    out = [l for l in preferred if l in layer_set]
    for l in sorted(layer_set):
        if l not in out:
            out.append(l)
    return out


def _read_json(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_relpath(raw: str) -> Optional[Path]:
    s = str(raw or "").strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return None
    if any(part in ("", ".", "..") for part in p.parts):
        return None
    return p


def _scatter_parse_cell_from_filename(
    dataset: str,
    clustering: Optional[str],
    split_norm: str,
    file_name: str,
) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse <model>_<layer>_<projection>_(xy|cluster).json into
    (model, layer, projection, kind). Returns None if not matched.
    """
    name = str(file_name or "").strip()
    if not name.endswith(".json"):
        return None
    stem = name[:-5]
    kind = ""
    if stem.endswith("_xy"):
        kind = "xy"
        stem = stem[:-3]
    elif stem.endswith("_cluster"):
        kind = "cluster"
        stem = stem[:-8]
    else:
        return None

    projection = ""
    for proj in ("pca", "mds", "umap", "tsne"):
        token = f"_{proj}"
        if stem.endswith(token):
            projection = proj
            stem = stem[: -len(token)]
            break
    if not projection:
        return None

    compat = _materialize_compat_sankey_files(
        dataset,
        clustering=clustering,
        write_files=False,
        data_split=split_norm,
    )
    models = [str(x) for x in (compat.get("models") or [])]
    layers = [str(x) for x in (compat.get("layers") or [])]
    for m in models:
        for l in layers:
            if f"{m}_{l}" == stem:
                return m, l, projection, kind
    return None


def _scatter_parse_request_filename(file_name: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse requested scatter filename:
      <cell_core>_<projection>_(xy|cluster).json
    Returns (cell_core, projection, kind) or None.
    """
    name = str(file_name or "").strip()
    if not name.endswith(".json"):
        return None
    stem = name[:-5]
    if stem.endswith("_xy"):
        kind = "xy"
        stem = stem[:-3]
    elif stem.endswith("_cluster"):
        kind = "cluster"
        stem = stem[:-8]
    else:
        return None
    for proj in ("pca", "mds", "umap", "tsne"):
        token = f"_{proj}"
        if stem.endswith(token):
            return stem[: -len(token)], proj, kind
    return None


def _scatter_split_combined_file(combined_path: Path, kind: str) -> Optional[List[Dict[str, object]]]:
    """
    Split combined scatter rows (<...>_<projection>.json) into:
      - xy rows: [{real_id, x, y}]
      - cluster rows: [{real_id, cluster_id}]
    """
    data = _read_json(combined_path, None)
    if not isinstance(data, list):
        return None
    out: List[Dict[str, object]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        rid = row.get("real_id")
        if rid in (None, ""):
            continue
        rid_s = str(rid)
        if kind == "xy":
            try:
                x = float(row.get("x", 0) or 0)
            except Exception:
                x = 0.0
            try:
                y = float(row.get("y", 0) or 0)
            except Exception:
                y = 0.0
            out.append({"real_id": rid_s, "x": x, "y": y})
        else:
            try:
                cid = int(float(row.get("cluster_id", 0) or 0))
            except Exception:
                cid = 0
            out.append({"real_id": rid_s, "cluster_id": cid})
    return out


def _scatter_compute_xy_rows(
    dataset: str,
    epoch: str,
    layer: str,
    projection: str,
    split_norm: str,
) -> List[Dict[str, object]]:
    import numpy as np
    from dimensionality_reduction import reduce_embedding

    X, real_ids = _load_embedding_cell(dataset, epoch, layer, data_split=split_norm)
    X_arr = np.asarray(X)
    if X_arr.ndim > 2:
        X_arr = X_arr.reshape(X_arr.shape[0], -1)
    coords = reduce_embedding(X_arr, method=str(projection), random_state=42)
    out: List[Dict[str, object]] = []
    for i in range(coords.shape[0]):
        rid = str(real_ids[i]) if i < len(real_ids) else str(i)
        out.append(
            {
                "real_id": rid,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
            }
        )
    return out


def _scatter_compute_cluster_rows(
    dataset: str,
    clustering: str,
    epoch: str,
    layer: str,
    split_norm: str,
) -> List[Dict[str, object]]:
    rows, err = _load_cluster_rows_for_cell(dataset, clustering, epoch, layer, [split_norm])
    if err == "cluster_assignments_not_found":
        # Fallback from sankey samples if assignment CSV is absent.
        key = f"{epoch}__{layer}"
        samples = _read_json(_case_dir(dataset, clustering, split_norm) / "sankey" / "sankey_samples.json", [])
        derived = _build_scatter_cluster_by_cell_from_samples(samples).get(key, [])
        return derived if isinstance(derived, list) else []
    return [
        {
            "real_id": str(r.get("real_id", "")),
            "cluster_id": int(r.get("cluster_id", 0)),
        }
        for r in rows
        if r.get("real_id") not in (None, "")
    ]


def _scatter_legacy_rows(base: Path, model: str, layer: str, projection: str) -> Optional[List[Dict[str, object]]]:
    """
    Legacy scatter file fallback:
      <model>_<layer>_<projection>.json with rows containing x/y/cluster_id/real_id.
    """
    legacy = base / f"{model}_{layer}_{projection}.json"
    if not legacy.is_file():
        return None
    data = _read_json(legacy, None)
    if not isinstance(data, list):
        return None
    out: List[Dict[str, object]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        rid = row.get("real_id")
        if rid in (None, ""):
            continue
        try:
            x = float(row.get("x", 0) or 0)
        except Exception:
            x = 0.0
        try:
            y = float(row.get("y", 0) or 0)
        except Exception:
            y = 0.0
        try:
            cid = int(float(row.get("cluster_id", 0) or 0))
        except Exception:
            cid = 0
        out.append(
            {
                "real_id": str(rid),
                "x": x,
                "y": y,
                "cluster_id": cid,
            }
        )
    return out


def _materialize_compat_sankey_files(
    case: str,
    clustering: Optional[str] = None,
    write_files: bool = True,
    data_split: Optional[str] = None,
) -> Dict[str, object]:
    case_dir = _case_dir(case, clustering, data_split=data_split)
    sankey_dir = case_dir / "sankey"
    all_path = sankey_dir / "all_sankey.json"
    samples_path = sankey_dir / "sankey_samples.json"

    all_sankey = _read_json(all_path, {"epoch": {}, "layer": {}})
    samples = _read_json(samples_path, [])
    if not isinstance(samples, list):
        samples = []

    compat = _build_compat_payload_from_sankey(all_sankey, samples)
    models = compat["models"]
    layers = compat["layers"]
    model_rows = compat["model_samples"]
    by_layer = compat["vertical_samples_by_layer"]
    all_vertical = compat["vertical_samples"]

    if write_files:
        for m in models:
            _write_json(sankey_dir / f"{m}_sankey_samples.json", model_rows.get(m, []))

    if write_files:
        _write_json(sankey_dir / "vertical_model_samples.json", all_vertical)
        for l in layers:
            _write_json(sankey_dir / f"vertical_model_samples_{l}.json", by_layer.get(l, []))

    return {
        "models": len(models),
        "layers": len(layers),
        "sankey_samples": len(samples),
        "vertical_samples": len(all_vertical),
        "all_sankey": all_sankey,
        "model_samples": model_rows,
        "vertical_samples_by_layer": by_layer,
    }


def _build_compat_payload_from_sankey(
    all_sankey: Dict[str, object],
    samples: List[Dict[str, object]],
) -> Dict[str, object]:
    models = _ordered_models(
        list((all_sankey.get("epoch") or {}).keys())
        + [str(r.get("from_model", "")).strip() for r in samples]
        + [str(r.get("to_model", "")).strip() for r in samples]
    )
    layers = _ordered_layers(
        list((all_sankey.get("layer") or {}).keys())
        + [str(r.get("from_layer", "")).strip() for r in samples]
        + [str(r.get("to_layer", "")).strip() for r in samples]
    )

    model_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in models}
    for row in samples:
        fm = str(row.get("from_model", "")).strip()
        tm = str(row.get("to_model", "")).strip()
        if fm and tm and fm == tm:
            model_rows.setdefault(fm, []).append(row)
        elif fm:
            model_rows.setdefault(fm, []).append(row)

    assignments: Dict[Tuple[str, str], Dict[str, int]] = {}
    for row in samples:
        rid = str(row.get("real_id", "")).strip()
        model = str(row.get("from_model", "")).strip() or str(row.get("to_model", "")).strip()
        fl = str(row.get("from_layer", "")).strip()
        tl = str(row.get("to_layer", "")).strip()
        fc = row.get("from_cluster")
        tc = row.get("to_cluster")
        if not rid or not model:
            continue
        key = (rid, model)
        if key not in assignments:
            assignments[key] = {}
        try:
            if fl != "":
                assignments[key][fl] = int(fc)
            if tl != "":
                assignments[key][tl] = int(tc)
        except Exception:
            pass

    by_layer: Dict[str, List[Dict[str, object]]] = {l: [] for l in layers}
    all_vertical: List[Dict[str, object]] = []
    for i in range(len(models) - 1):
        src = models[i]
        tgt = models[i + 1]
        real_ids = sorted({rid for (rid, _m) in assignments.keys()})
        for rid in real_ids:
            src_map = assignments.get((rid, src))
            tgt_map = assignments.get((rid, tgt))
            if not src_map or not tgt_map:
                continue
            rid_value: object = int(rid) if rid.isdigit() else rid
            for layer in layers:
                if layer not in src_map or layer not in tgt_map:
                    continue
                item = {
                    "real_id": rid_value,
                    "from_layer": layer,
                    "to_layer": layer,
                    "from_model": src,
                    "to_model": tgt,
                    "from_cluster": int(src_map[layer]),
                    "to_cluster": int(tgt_map[layer]),
                }
                all_vertical.append(item)
                by_layer.setdefault(layer, []).append(item)

    return {
        "models": models,
        "layers": layers,
        "model_samples": model_rows,
        "vertical_samples": all_vertical,
        "vertical_samples_by_layer": by_layer,
    }


def _build_scatter_cluster_by_cell_from_samples(
    samples: List[Dict[str, object]],
) -> Dict[str, List[Dict[str, object]]]:
    # key: "<model>__<layer>" -> [{"real_id": ..., "cluster_id": ...}, ...]
    by_cell: Dict[str, Dict[str, int]] = {}
    for row in samples:
        rid = str(row.get("real_id", "")).strip()
        if not rid:
            continue
        fm = str(row.get("from_model", "")).strip()
        tm = str(row.get("to_model", "")).strip()
        fl = str(row.get("from_layer", "")).strip()
        tl = str(row.get("to_layer", "")).strip()
        try:
            fc = int(row.get("from_cluster"))
            tc = int(row.get("to_cluster"))
        except Exception:
            continue

        if fm and fl:
            key = f"{fm}__{fl}"
            by_cell.setdefault(key, {})[rid] = fc
        if tm and tl:
            key = f"{tm}__{tl}"
            by_cell.setdefault(key, {})[rid] = tc

    out: Dict[str, List[Dict[str, object]]] = {}
    for cell_key, rid_map in by_cell.items():
        rows = [{"real_id": rid, "cluster_id": int(cid)} for rid, cid in rid_map.items()]
        out[cell_key] = rows
    return out


def _load_generated_case_payload(base: Path) -> Dict[str, object]:
    sankey_dir = base / "sankey"
    topk_dir = base / "topk"
    sub_dir = base / "substructures"

    all_sankey = _read_json(sankey_dir / "all_sankey.json", {"epoch": {}, "layer": {}})
    samples = _read_json(sankey_dir / "sankey_samples.json", [])
    if not isinstance(samples, list):
        samples = []

    compat = _build_compat_payload_from_sankey(all_sankey, samples)
    scatter_cluster_by_cell = _build_scatter_cluster_by_cell_from_samples(samples)

    topk_reps = _read_json(topk_dir / "cluster_representatives.json", [])
    cluster_sorted = _read_json(topk_dir / "cluster_sorted_members.json", [])
    substructures = _read_json(sub_dir / "cluster_substructures.json", [])
    if not isinstance(topk_reps, list):
        topk_reps = []
    if not isinstance(cluster_sorted, list):
        cluster_sorted = []
    if not isinstance(substructures, list):
        substructures = []

    return {
        "all_sankey": all_sankey,
        "model_samples": compat["model_samples"],
        "vertical_samples_by_layer": compat["vertical_samples_by_layer"],
        "topk_representatives": topk_reps,
        "cluster_sorted_members": cluster_sorted,
        "substructures": substructures,
        "scatter_cluster_by_cell": scatter_cluster_by_cell,
    }


def _search_smiles_substructure_in_case(case: str, smiles_query: str, data_split: Optional[str] = None):
    """
    Substructure search: return list of real_ids whose molecule contains the query
    (query is a substructure; all IDs that contain it are returned).
    Uses RDKit HasSubstructMatch. Returns (matches, error_message).
    """
    split_selector = (str(data_split).strip().lower() if data_split is not None else "all") or "all"
    if split_selector not in ("all", "train", "test", "val"):
        split_selector = "all"

    meta_path = DATA_ROOT / case / "id_smiles_y_true_y_pred.json"
    rows: List[dict] = []
    if meta_path.is_file():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                rows = [r for r in raw if isinstance(r, dict)]
        except Exception as e:
            print("[flask-backend] failed to load %s: %s" % (meta_path, e), flush=True)
            return [], None
    else:
        # Backward compatibility for older layouts.
        legacy_split = "test" if split_selector == "all" else split_selector
        legacy_path = DATA_ROOT / case / legacy_split / "real_id_to_smiles.json"
        if not legacy_path.is_file():
            return [], None
        try:
            raw = json.loads(legacy_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                rows = [r for r in raw if isinstance(r, dict)]
            elif isinstance(raw, dict):
                rows = [{"real_id": k, "smiles": v, "split": legacy_split} for k, v in raw.items()]
        except Exception as e:
            print("[flask-backend] failed to load %s: %s" % (legacy_path, e), flush=True)
            return [], None
    q = _normalize_smiles(smiles_query)
    if not q:
        return [], "empty_smiles"
    try:
        from rdkit import Chem
        query_mol = Chem.MolFromSmiles(q)
        if query_mol is None:
            return [], "invalid_query_smiles"
    except Exception as e:
        print("[flask-backend] rdkit/query mol failed: %s" % e, flush=True)
        return [], "invalid_query_smiles"
    matches = []
    seen_ids = set()
    for row in rows:
        row_split = str(row.get("split") or "").strip().lower()
        if split_selector == "all":
            if row_split and row_split not in ("train", "test"):
                continue
        elif row_split != split_selector:
            continue
        real_id = row.get("real_id", row.get("id"))
        if real_id is None:
            continue
        rid = str(real_id)
        if rid in seen_ids:
            continue
        smi = _normalize_smiles(str(row.get("smiles", "")))
        if not smi:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and mol.HasSubstructMatch(query_mol):
                seen_ids.add(rid)
                matches.append(rid)
        except Exception:
            continue
    return matches, None


@app.route("/search-smiles", methods=["POST", "OPTIONS"])
def search_smiles():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    smiles = payload.get("smiles")
    dataset = payload.get("dataset")  # optional: limit search to a single case
    data_split = payload.get("data_split")

    print(
        f"[flask-backend] search-smiles smiles={smiles}, dataset={dataset}, data_split={data_split}",
        flush=True,
    )

    if not isinstance(smiles, str) or not smiles.strip():
        return jsonify({"ok": False, "error": "invalid_smiles"}), 400

    cases = [dataset] if isinstance(dataset, str) else list(DEFAULT_CASES)
    results = {}
    total = 0
    for case in cases:
        ids, err = _search_smiles_substructure_in_case(case, smiles, data_split=data_split)
        if err:
            return jsonify({"ok": False, "error": err}), 400
        if ids:
            results[case] = ids
            total += len(ids)
        print(
            "[flask-backend] search-smiles case=%s substructure_matched=%s ids_preview=%s"
            % (case, len(ids), ids[:20]),
            flush=True,
        )

    print("[flask-backend] search-smiles total_ids=%s" % total, flush=True)
    return jsonify({"ok": True, "smiles": smiles, "matches": results, "total_ids": total})


def _cluster_id_from_label(label: str) -> int:
    """Parse cluster ID from label like 'ffn_layer_1_C0' or 'fingerprint_C2'. Returns -1 if no _C suffix."""
    if not label or "_C" not in label:
        return -1
    try:
        return int(label.split("_C")[-1])
    except (ValueError, TypeError):
        return -1


def _filter_sankey_for_range(case: str, min_value: int, max_value: int, clustering: Optional[str] = None) -> Tuple[int, int]:
    """Filter all_sankey.json and all *_sankey_samples.json / vertical_model_samples_*.json
    in-place for a given [min,max] **cluster ID** range (clustering range, not epoch).

    - all_sankey: keep edges where both source_label and target_label have cluster ID in [min, max].
    - Sample files: keep rows where from_cluster and to_cluster are in [min, max].

    Returns (kept_edges, kept_sample_rows) for logging.
    """
    case_dir = _case_dir(case, clustering)
    sankey_dir = case_dir / "sankey"
    all_sankey_path = sankey_dir / "all_sankey.json"
    samples_path = sankey_dir / "sankey_samples.json"

    def cluster_in_range(cid: int) -> bool:
        return min_value <= cid <= max_value

    kept_edges = 0
    kept_rows = 0

    if all_sankey_path.is_file():
        try:
            data = json.loads(all_sankey_path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover - defensive
            print(f"[flask-backend] failed to load {all_sankey_path}: {e}", flush=True)
        else:
            epoch_dict = data.get("epoch") or {}
            for ek, payload in epoch_dict.items():
                edges = payload.get("edges") or []
                filtered_edges = [
                    e
                    for e in edges
                    if cluster_in_range(_cluster_id_from_label(e.get("source_label", "")))
                    and cluster_in_range(_cluster_id_from_label(e.get("target_label", "")))
                ]
                kept_edges += len(filtered_edges)
                payload["edges"] = filtered_edges
            data["epoch"] = epoch_dict
            all_sankey_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def filter_sample_rows(rows: list) -> list:
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                fc = int(row.get("from_cluster"))
                tc = int(row.get("to_cluster"))
            except (ValueError, TypeError):
                continue
            if cluster_in_range(fc) and cluster_in_range(tc):
                out.append(row)
        return out

    if samples_path.is_file():
        try:
            rows = json.loads(samples_path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover - defensive
            print(f"[flask-backend] failed to load {samples_path}: {e}", flush=True)
        else:
            filtered = filter_sample_rows(rows)
            kept_rows = len(filtered)
            samples_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    for f in sankey_dir.iterdir():
        if not f.is_file() or f.suffix != ".json":
            continue
        name = f.name
        if not (
            name.endswith("_sankey_samples.json")
            or name.startswith("vertical_model_samples")
        ):
            continue
        try:
            rows = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[flask-backend] skip filter {f.name}: {e}", flush=True)
            continue
        if not isinstance(rows, list):
            continue
        filtered = filter_sample_rows(rows)
        f.write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    return kept_edges, kept_rows


def _set_generate_range_job(job_id: str, **updates: object) -> None:
    with GENERATE_RANGE_JOBS_LOCK:
        job = GENERATE_RANGE_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def _trim_generate_range_jobs() -> None:
    with GENERATE_RANGE_JOBS_LOCK:
        if len(GENERATE_RANGE_JOBS) <= MAX_GENERATE_RANGE_JOBS:
            return
        completed = sorted(
            (
                (jid, job)
                for jid, job in GENERATE_RANGE_JOBS.items()
                if str(job.get("status")) in ("succeeded", "failed")
            ),
            key=lambda item: float(item[1].get("updated_at", 0.0)),
        )
        for jid, _ in completed:
            if len(GENERATE_RANGE_JOBS) <= MAX_GENERATE_RANGE_JOBS:
                break
            GENERATE_RANGE_JOBS.pop(jid, None)


def _generate_range_impl(payload: Dict[str, object], job_id: Optional[str] = None) -> Tuple[Dict[str, object], int]:
    """Re-run clustering for ALL cells with user-supplied params."""
    include_payload = bool(payload.get("include_payload", False))
    # Always run generate-range in memory-only mode for frontend runtime payloads.
    # Ignore client-provided persist flag to avoid mutating on-disk JSON artifacts.
    persist = False
    min_value = payload.get("min")
    max_value = payload.get("max")
    k_fixed = payload.get("k")
    linkage = (payload.get("linkage") or "ward").strip().lower()
    if linkage not in ("ward", "complete", "average"):
        linkage = "ward"
    dataset = payload.get("dataset")
    clustering = (payload.get("clustering") or "gmm").lower().strip()
    data_split = payload.get("data_split") or payload.get("preferred_split")

    if clustering == "pgmeans":
        clustering = "gmm"

    # Optional substructure member cap for MCS (first M nearest / farthest per cluster)
    sub_m = payload.get("substructure_max_members") or payload.get("sub_m")

    # Agglomerative: always honor linkage; optionally honor fixed k.
    # KNN: honor fixed k as n_neighbors when provided.
    n_clusters_override = None
    linkage_override = None
    knn_k_override = None
    if clustering == "agglomerative":
        linkage_override = linkage
    if clustering == "knn":
        if k_fixed is not None:
            try:
                knn_k_override = max(1, int(k_fixed))
            except (ValueError, TypeError):
                return {"ok": False, "error": "invalid_k"}, 400
            print(
                f"[flask-backend] generate-range all cells knn k={knn_k_override} dataset={dataset} split={data_split}",
                flush=True,
            )
        else:
            # Keep UI compatibility: if only min/max are provided, use max as K.
            try:
                knn_k_override = max(1, int(max_value))
            except (ValueError, TypeError):
                return {"ok": False, "error": "invalid_k"}, 400
            print(
                f"[flask-backend] generate-range all cells knn k(from max)={knn_k_override} "
                f"dataset={dataset} split={data_split}",
                flush=True,
            )
        k_min = k_max = max(2, min(knn_k_override, 15))
    elif clustering == "agglomerative" and k_fixed is not None:
        try:
            n_clusters_override = int(k_fixed)
            n_clusters_override = max(2, n_clusters_override)
        except (ValueError, TypeError):
            return {"ok": False, "error": "invalid_k"}, 400
        k_min = k_max = n_clusters_override
        print(
            f"[flask-backend] generate-range all cells agg k={n_clusters_override} linkage={linkage_override} "
            f"dataset={dataset} split={data_split}",
            flush=True,
        )
    else:
        try:
            k_min = int(min_value)
            k_max = int(max_value)
        except (ValueError, TypeError):
            return {"ok": False, "error": "invalid_range"}, 400
        if k_min > k_max:
            k_min, k_max = k_max, k_min
        if k_min < 2:
            k_min = 2
        if k_max < k_min:
            k_max = k_min
        print(
            f"[flask-backend] generate-range k_min={k_min}, k_max={k_max}, dataset={dataset}, "
            f"clustering={clustering}, split={data_split}",
            flush=True,
        )

    if not dataset or dataset not in DEFAULT_CASES:
        return {"ok": False, "error": "invalid_dataset"}, 400
    if clustering not in ("gmm", "agglomerative", "hdbscan", "knn"):
        return {"ok": False, "error": "invalid_clustering"}, 400

    input_path = EMBEDDING_PATHS.get(dataset)
    if not input_path or not input_path.is_file():
        return {"ok": False, "error": "embedding_not_found", "detail": str(input_path)}, 404

    split_norm = _normalize_split(data_split) or "test"
    temp_root: Optional[Path] = None
    if persist:
        case_dir = DATA_ROOT / dataset / split_norm
        case_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_root = Path(tempfile.mkdtemp(prefix=f"latentvis_generate_{dataset}_{clustering}_"))
        case_dir = temp_root / dataset / split_norm
        case_dir.mkdir(parents=True, exist_ok=True)

    try:
        from run_pipeline_from_embeddings import run_pipeline
        if job_id:
            _set_generate_range_job(job_id, message="running_pipeline")

        ok = run_pipeline(
            input_path=str(input_path),
            output_dir=str(case_dir),
            k_min=k_min,
            k_max=k_max,
            clustering_method=clustering,
            skip_tsne=True,
            n_clusters_override=n_clusters_override,
            linkage_override=linkage_override,
            knn_k_override=knn_k_override,
            reuse_scatter_coords=True,
            substructure_max_members=sub_m,
            scatter_subdir="scatterplot",
            data_split=split_norm,
            write_fmi_cspa=True,
            interactive_output_suffix="",
            resume=False,
            verbose_dr=False,
        )
        if not ok:
            if temp_root is not None:
                shutil.rmtree(temp_root, ignore_errors=True)
            return {"ok": False, "error": "pipeline_failed"}, 500
    except Exception as e:
        print(f"[flask-backend] generate-range pipeline failed: {e}", flush=True)
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)
        return {"ok": False, "error": "pipeline_failed", "detail": str(e)}, 500

    # Do not materialize per-model/per-layer Sankey compatibility files on disk.
    # Frontend fetches those via /data-file and backend computes them on demand in memory.

    # Recompute transition stats for updated cluster assignments after generate-range.
    if persist:
        transition_script = _BACKEND_DIR / "scripts" / "compute_layer_transition_stats.py"
        if transition_script.is_file():
            try:
                subprocess.run(
                    [sys.executable, str(transition_script)],
                    cwd=str(_BACKEND_DIR.parent),
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
            except subprocess.SubprocessError as e:
                print(f"[flask-backend] compute_layer_transition_stats failed (non-fatal): {e}", flush=True)
                stderr = getattr(e, "stderr", None)
                if stderr:
                    try:
                        print("[flask-backend] compute_layer_transition_stats stderr:\n%s" % stderr.decode("utf-8", errors="ignore"), flush=True)
                    except Exception:
                        print("[flask-backend] compute_layer_transition_stats stderr(raw): %r" % (stderr,), flush=True)

    updated = _collect_updated_results_summary(base, output_suffix="")
    summary = {dataset: {"status": "done", "updated": updated}}

    out = {"ok": True, "min": k_min, "max": k_max, "dataset": dataset, "clustering": clustering, "summary": summary}
    if include_payload:
        out["payload"] = {dataset: _load_generated_case_payload(base)}
        out["run_id"] = str(int(time.time() * 1000))
    if n_clusters_override is not None:
        out["k"] = n_clusters_override
        out["linkage"] = linkage_override
    out["persist"] = persist
    if not persist:
        out["note"] = "Generated in temporary backend workspace; no files were written to public/data."
    if temp_root is not None:
        shutil.rmtree(temp_root, ignore_errors=True)
    return out, 200


def _generate_range_worker(job_id: str, payload: Dict[str, object]) -> None:
    _set_generate_range_job(job_id, status="running", message="starting")
    try:
        result, status_code = _generate_range_impl(payload, job_id=job_id)
        if status_code >= 400:
            _set_generate_range_job(
                job_id,
                status="failed",
                message="failed",
                error={"status_code": status_code, "detail": result},
            )
            return
        _set_generate_range_job(job_id, status="succeeded", message="done", result=result)
    except Exception as e:
        print(f"[flask-backend] generate-range async worker failed: {e}", flush=True)
        _set_generate_range_job(
            job_id,
            status="failed",
            message="failed",
            error={"status_code": 500, "detail": str(e)},
        )
    finally:
        _trim_generate_range_jobs()


@app.route("/generate-range", methods=["POST", "OPTIONS"])
def generate_range():
    """Queue a generate-range job and return job_id for polling."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    async_mode = bool(payload.get("async", True))
    if not async_mode:
        out, status_code = _generate_range_impl(payload)
        return jsonify(out), status_code

    job_id = uuid.uuid4().hex
    now = time.time()
    with GENERATE_RANGE_JOBS_LOCK:
        GENERATE_RANGE_JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "message": "queued",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }
    worker = threading.Thread(target=_generate_range_worker, args=(job_id, payload), daemon=True)
    worker.start()
    return jsonify({"ok": True, "job_id": job_id, "status": "queued"}), 202


@app.route("/generate-range/jobs/<job_id>", methods=["GET", "OPTIONS"])
def generate_range_job_status(job_id: str):
    if request.method == "OPTIONS":
        return ("", 204)
    with GENERATE_RANGE_JOBS_LOCK:
        job = GENERATE_RANGE_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job_not_found", "job_id": job_id}), 404

    status = str(job.get("status", "unknown"))
    out: Dict[str, object] = {
        "ok": status != "failed",
        "job_id": job_id,
        "status": status,
        "message": str(job.get("message", "")),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }
    if status == "succeeded":
        out["result"] = job.get("result")
    elif status == "failed":
        out["error"] = job.get("error")
    return jsonify(out)


def _collect_updated_results_summary(base: Path, output_suffix: str = "") -> Dict[str, object]:
    """Scan pipeline output dir and return counts for scatter, topk, substructures, sankey, cluster_assignments."""
    summary = {"cells": 0, "scatter_files": 0, "topk_representatives": 0, "topk_sorted_members": 0, "substructure_entries": 0, "sankey_ok": False, "cluster_assignments_ok": False}
    if not base.is_dir():
        return summary
    scatter_dir = base / ("scatterplot" + output_suffix)
    if scatter_dir.is_dir():
        pca_files = list(scatter_dir.glob("*_pca.json"))
        summary["cells"] = len(pca_files)
        summary["scatter_files"] = len(list(scatter_dir.glob("*.json")))
    topk_dir = base / ("topk" + output_suffix)
    if (topk_dir / "cluster_representatives.json").is_file():
        try:
            data = json.loads((topk_dir / "cluster_representatives.json").read_text(encoding="utf-8"))
            summary["topk_representatives"] = len(data) if isinstance(data, list) else 0
        except Exception:
            pass
    if (topk_dir / "cluster_sorted_members.json").is_file():
        try:
            data = json.loads((topk_dir / "cluster_sorted_members.json").read_text(encoding="utf-8"))
            summary["topk_sorted_members"] = len(data) if isinstance(data, list) else 0
        except Exception:
            pass
    sub_dir = base / ("substructures" + output_suffix)
    if (sub_dir / "cluster_substructures.json").is_file():
        try:
            data = json.loads((sub_dir / "cluster_substructures.json").read_text(encoding="utf-8"))
            summary["substructure_entries"] = len(data) if isinstance(data, list) else 0
        except Exception:
            pass
    summary["sankey_ok"] = (base / ("sankey" + output_suffix) / "all_sankey.json").is_file()
    summary["cluster_assignments_ok"] = (base / ("clustering_metrics" + output_suffix) / "cluster_assignments_with_real_id.csv").is_file()
    return summary


@app.route("/materialize-sankey-files", methods=["POST", "OPTIONS"])
def materialize_sankey_files():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    dataset = payload.get("dataset")
    clustering = payload.get("clustering", DEFAULT_CLUSTERING)
    data_split = payload.get("data_split") or payload.get("preferred_split")
    include_payload = bool(payload.get("include_payload", False))
    write_files = bool(payload.get("write_files", True))
    cases = [dataset] if isinstance(dataset, str) else list(DEFAULT_CASES)

    summary: Dict[str, Dict[str, int]] = {}
    payload_out: Dict[str, Dict[str, object]] = {}
    for case in cases:
        compat = _materialize_compat_sankey_files(
            case,
            clustering=clustering,
            write_files=write_files,
            data_split=data_split if isinstance(dataset, str) else None,
        )
        summary[case] = {
            "models": int(compat["models"]),
            "layers": int(compat["layers"]),
            "sankey_samples": int(compat["sankey_samples"]),
            "vertical_samples": int(compat["vertical_samples"]),
        }
        if include_payload:
            payload_out[case] = {
                "all_sankey": compat["all_sankey"],
                "model_samples": compat["model_samples"],
                "vertical_samples_by_layer": compat["vertical_samples_by_layer"],
            }
        print(
            f"[flask-backend] materialize-sankey-files case={case}, clustering={clustering}, "
            f"models={compat['models']}, layers={compat['layers']}, vertical_samples={compat['vertical_samples']}",
            flush=True,
        )

    out: Dict[str, object] = {"ok": True, "summary": summary, "clustering": clustering}
    if include_payload:
        out["payload"] = payload_out
    return jsonify(out)


@app.route("/sankey/all", methods=["GET"])
def sankey_all():
    dataset = request.args.get("dataset", "")
    clustering = request.args.get("clustering", DEFAULT_CLUSTERING)
    data_split = request.args.get("data_split")
    if dataset not in DEFAULT_CASES:
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400
    path = _case_dir(dataset, clustering, data_split) / "sankey" / "all_sankey.json"
    payload = _read_json(path, None)
    if payload is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify(payload)


@app.route("/sankey/model", methods=["GET"])
def sankey_model():
    dataset = request.args.get("dataset", "")
    clustering = request.args.get("clustering", DEFAULT_CLUSTERING)
    data_split = request.args.get("data_split")
    model = request.args.get("model", "")
    if dataset not in DEFAULT_CASES or not model:
        return jsonify({"ok": False, "error": "invalid_args"}), 400
    path = _case_dir(dataset, clustering, data_split) / "sankey" / f"{model}_sankey_samples.json"
    payload = _read_json(path, None)
    if payload is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify(payload)


@app.route("/sankey/vertical", methods=["GET"])
def sankey_vertical():
    dataset = request.args.get("dataset", "")
    clustering = request.args.get("clustering", DEFAULT_CLUSTERING)
    data_split = request.args.get("data_split")
    layer = request.args.get("layer", "")
    if dataset not in DEFAULT_CASES or not layer:
        return jsonify({"ok": False, "error": "invalid_args"}), 400
    path = _case_dir(dataset, clustering, data_split) / "sankey" / f"vertical_model_samples_{layer}.json"
    payload = _read_json(path, None)
    if payload is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify(payload)


@app.route("/data-file", methods=["GET", "OPTIONS"])
def data_file():
    """
    Serve JSON/binary files from backend DATA_ROOT with path controls.
    Query params:
      - dataset: required (chemprop/chemeleon)
      - split: optional (train/test/val/all; all -> train for base path)
      - clustering: optional (pgmeans/agglomerative/hdbscan)
      - section: optional (sankey/scatterplot/topk/substructures/clustering_metrics/bic_aic/...)
      - file: required relative file path under the resolved base
      - dataset_root: optional (1/true), serve from DATA_ROOT/<dataset> directly
    """
    if request.method == "OPTIONS":
        return ("", 204)

    dataset = str(request.args.get("dataset", "")).strip().lower()
    if dataset not in DEFAULT_CASES:
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400

    file_raw = request.args.get("file", "")
    file_rel = _safe_relpath(file_raw)
    if file_rel is None:
        return jsonify({"ok": False, "error": "invalid_file"}), 400

    split_raw = str(request.args.get("split", "")).strip().lower()
    split_for_base = "train" if split_raw == "all" else split_raw
    clustering = request.args.get("clustering")
    section_raw = request.args.get("section")
    section_rel = _safe_relpath(section_raw) if section_raw else None
    dataset_root = str(request.args.get("dataset_root", "")).strip().lower() in ("1", "true", "yes")

    if dataset_root:
        base = DATA_ROOT / dataset
    else:
        base = _case_dir(dataset, clustering, data_split=split_for_base)
        if section_rel is not None:
            base = base / section_rel

    target = (base / file_rel).resolve()
    try:
        target.relative_to(base.resolve())
    except Exception:
        return jsonify({"ok": False, "error": "invalid_path"}), 400

    if not target.exists() and (section_rel == Path("sankey") if section_rel else False):
        # Compute compatibility samples from all_sankey + sankey_samples on demand.
        name = target.name
        compat = _materialize_compat_sankey_files(
            dataset,
            clustering=clustering,
            write_files=False,
            data_split=split_for_base,
        )
        if name.endswith("_sankey_samples.json"):
            model = name[: -len("_sankey_samples.json")]
            rows = (compat.get("model_samples") or {}).get(model, [])
            return jsonify(rows if isinstance(rows, list) else [])
        if name.startswith("vertical_model_samples_") and name.endswith(".json"):
            layer = name[len("vertical_model_samples_") : -len(".json")]
            rows = (compat.get("vertical_samples_by_layer") or {}).get(layer, [])
            return jsonify(rows if isinstance(rows, list) else [])
        if name == "vertical_model_samples.json":
            rows = compat.get("vertical_samples") or []
            return jsonify(rows if isinstance(rows, list) else [])

    if not target.exists() and (section_rel == Path("scatterplot") if section_rel else False):
        try:
            name = target.name
            req = _scatter_parse_request_filename(name)
            if req is not None:
                cell_core, projection, kind = req
                method = str(clustering or DEFAULT_CLUSTERING).strip().lower()
                method_dir = "pgmeans" if method in ("pgmeans", "gmm") else method
                scatter_base_candidates: List[Path] = []
                # Preferred new layout
                scatter_base_candidates.append(DATA_ROOT / dataset / split_for_base / "clustering" / method_dir / "scatterplot")
                # User's current layout: /<dataset>/<split>/<method>/scatterplot
                scatter_base_candidates.append(DATA_ROOT / dataset / split_for_base / method_dir / "scatterplot")
                # Legacy pgmeans layout
                scatter_base_candidates.append(DATA_ROOT / dataset / split_for_base / "scatterplot")
                # Current resolved base (already includes /scatterplot for most callers)
                scatter_base_candidates.append(base)
                seen_bases = set()
                uniq_bases: List[Path] = []
                for b in scatter_base_candidates:
                    bb = b.resolve()
                    if str(bb) in seen_bases:
                        continue
                    seen_bases.add(str(bb))
                    uniq_bases.append(bb)

                for sb in uniq_bases:
                    combined = sb / f"{cell_core}_{projection}.json"
                    if not combined.is_file():
                        continue
                    rows = _scatter_split_combined_file(combined, kind)
                    if rows is not None:
                        return jsonify(rows)

            parsed = _scatter_parse_cell_from_filename(dataset, clustering, split_for_base, name)
            if parsed is not None:
                model, layer, projection, kind = parsed
                cache_key = f"{dataset}|{str(clustering or DEFAULT_CLUSTERING).lower()}|{split_for_base}|{model}|{layer}|{projection}"
                try:
                    with SCATTER_CELL_CACHE_LOCK:
                        cached = SCATTER_CELL_CACHE.get(cache_key)
                    if cached is None:
                        xy_rows = _scatter_compute_xy_rows(dataset, model, layer, projection, split_for_base)
                        cluster_rows = _scatter_compute_cluster_rows(
                            dataset,
                            str(clustering or DEFAULT_CLUSTERING).lower(),
                            model,
                            layer,
                            split_for_base,
                        )
                        cached = {"xy": xy_rows, "cluster": cluster_rows}
                        with SCATTER_CELL_CACHE_LOCK:
                            if len(SCATTER_CELL_CACHE) >= MAX_SCATTER_CELL_CACHE and SCATTER_CELL_CACHE:
                                # Drop an arbitrary old entry to bound memory.
                                SCATTER_CELL_CACHE.pop(next(iter(SCATTER_CELL_CACHE)))
                            SCATTER_CELL_CACHE[cache_key] = cached
                    rows = cached.get(kind) if isinstance(cached, dict) else []
                    return jsonify(rows if isinstance(rows, list) else [])
                except Exception as e:
                    # Fallback to legacy combined scatter file if available.
                    legacy_rows = _scatter_legacy_rows(base, model, layer, projection)
                    if isinstance(legacy_rows, list):
                        if kind == "xy":
                            return jsonify(
                                [
                                    {"real_id": r["real_id"], "x": r.get("x", 0), "y": r.get("y", 0)}
                                    for r in legacy_rows
                                ]
                            )
                        return jsonify(
                            [
                                {"real_id": r["real_id"], "cluster_id": int(float(r.get("cluster_id", 0) or 0))}
                                for r in legacy_rows
                            ]
                        )
                    return jsonify(
                        {
                            "ok": False,
                            "error": "scatter_compute_failed",
                            "detail": str(e),
                            "dataset": dataset,
                            "clustering": str(clustering or DEFAULT_CLUSTERING).lower(),
                            "split": split_for_base,
                            "model": model,
                            "layer": layer,
                            "projection": projection,
                            "kind": kind,
                        }
                    ), 500
        except Exception as e:
            return jsonify(
                {
                    "ok": False,
                    "error": "scatter_route_failed",
                    "detail": str(e),
                    "dataset": dataset,
                    "clustering": str(clustering or DEFAULT_CLUSTERING).lower(),
                    "split": split_for_base,
                    "file": str(file_rel),
                }
            ), 500

    if not target.exists():
        return jsonify({"ok": False, "error": "not_found", "path": str(target)}), 404

    if target.suffix.lower() == ".json":
        payload = _read_json(target, None)
        if payload is None:
            return jsonify({"ok": False, "error": "invalid_json", "path": str(target)}), 500
        return jsonify(payload)

    return send_file(str(target))


@app.route("/real-id-split-map", methods=["GET", "OPTIONS"])
def real_id_split_map():
    if request.method == "OPTIONS":
        return ("", 204)

    dataset = str(request.args.get("dataset", "")).strip().lower()
    datasets = [dataset] if dataset in DEFAULT_CASES else list(DEFAULT_CASES)

    out = {"generated_from": "real_id_to_smiles.json", "datasets": {}}
    for ds in datasets:
        train_map = _read_json(DATA_ROOT / ds / "train" / "real_id_to_smiles.json", {})
        test_map = _read_json(DATA_ROOT / ds / "test" / "real_id_to_smiles.json", {})
        train_ids = set(train_map.keys()) if isinstance(train_map, dict) else set()
        test_ids = set(test_map.keys()) if isinstance(test_map, dict) else set()
        all_ids = sorted(train_ids | test_ids)
        split_by_real_id: Dict[str, str] = {}
        overlap = 0
        for rid in all_ids:
            in_train = rid in train_ids
            in_test = rid in test_ids
            if in_train and in_test:
                split_by_real_id[rid] = "both"
                overlap += 1
            elif in_train:
                split_by_real_id[rid] = "train"
            else:
                split_by_real_id[rid] = "test"
        out["datasets"][ds] = {
            "counts": {
                "train": len(train_ids),
                "test": len(test_ids),
                "unique_total": len(all_ids),
                "overlap_both": overlap,
            },
            "split_by_real_id": split_by_real_id,
        }
    return jsonify(out)


@app.route("/cluster-containment", methods=["POST", "OPTIONS"])
def cluster_containment():
    """
    Compute containment/overlap matrix between user-provided clustering and backend clustering for one cell.
    Request body: dataset, epoch, layer, clustering (our method), optional data_split, use_current,
    and user_clustering: list of { "real_id": <id>, "user_cluster_id": <0-based id> } (or spreadsheet rows).
    Returns: user_cluster_ids, our_cluster_ids, matrix (rows=user, cols=our, value=pct of user cluster in our cluster),
    and entries list for each (user_cluster_id, our_cluster_id, count, percentage_of_user_cluster).
    """
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(silent=True) or {}
    dataset = payload.get("dataset")
    epoch = payload.get("epoch")
    layer = payload.get("layer")
    clustering = (payload.get("clustering") or DEFAULT_CLUSTERING).strip().lower()
    if clustering == "pgmeans":
        clustering = "gmm"
    data_split = payload.get("data_split")
    _ = payload.get("use_current")  # deprecated, ignored
    user_clustering = payload.get("user_clustering")
    if not isinstance(user_clustering, list):
        user_clustering = []

    if not dataset or dataset not in DEFAULT_CASES:
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400
    if epoch is None or layer is None:
        return jsonify({"ok": False, "error": "missing_epoch_or_layer"}), 400
    if clustering not in ("gmm", "agglomerative", "hdbscan", "knn"):
        return jsonify({"ok": False, "error": "invalid_clustering"}), 400

    import csv as csv_module
    splits = _resolve_train_test_splits(data_split, default="test")
    our_rid_to_cluster: Dict[str, int] = {}
    layer_col = "{}_cluster".format(str(layer).strip())
    saw_csv = False
    checked_paths: List[str] = []
    for split in splits:
        case_dir = _case_dir(dataset, clustering, split)
        csv_path = case_dir / "clustering_metrics" / "cluster_assignments_with_real_id.csv"
        checked_paths.append(str(csv_path))
        if not csv_path.is_file():
            continue
        saw_csv = True
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                if str(row.get("epoch")) != str(epoch):
                    continue
                rid = row.get("real_id")
                if rid is None:
                    continue
                rid_s = str(rid)
                key = f"{split}::{rid_s}" if len(splits) > 1 else rid_s
                try:
                    val = row.get(layer_col)
                    cid = int(float(val)) if val not in (None, "") else -1
                except (ValueError, TypeError):
                    cid = -1
                our_rid_to_cluster[key] = cid
                our_rid_to_cluster.setdefault(rid_s, cid)
    if not saw_csv:
        return jsonify({"ok": False, "error": "cluster_assignments_not_found", "checked_paths": checked_paths}), 404

    user_rid_to_cluster: Dict[str, int] = {}
    for item in user_clustering:
        if not isinstance(item, dict):
            continue
        rid = item.get("real_id")
        cid = item.get("user_cluster_id")
        if rid is None or cid is None:
            continue
        try:
            user_rid_to_cluster[str(rid)] = int(cid)
        except (ValueError, TypeError):
            continue

    if not user_rid_to_cluster:
        return jsonify({"ok": False, "error": "empty_user_clustering"}), 400

    user_cids = sorted(set(user_rid_to_cluster.values()))
    our_cids = sorted(set(our_rid_to_cluster.values()) - {-1})
    if not our_cids:
        return jsonify({"ok": False, "error": "no_our_clusters"}), 400

    counts: Dict[Tuple[int, int], int] = {}
    user_sizes: Dict[int, int] = {}
    for rid, u_cid in user_rid_to_cluster.items():
        user_sizes[u_cid] = user_sizes.get(u_cid, 0) + 1
        o_cid = -1
        if "::" in rid:
            o_cid = our_rid_to_cluster.get(rid, -1)
        else:
            for sp in splits:
                c = our_rid_to_cluster.get(f"{sp}::{rid}", None)
                if c is not None:
                    o_cid = c
                    break
            if o_cid < 0:
                o_cid = our_rid_to_cluster.get(rid, -1)
        if o_cid < 0:
            continue
        counts[(u_cid, o_cid)] = counts.get((u_cid, o_cid), 0) + 1

    matrix = []
    entries = []
    for u_cid in user_cids:
        row = []
        size_u = user_sizes.get(u_cid, 0) or 1
        for o_cid in our_cids:
            c = counts.get((u_cid, o_cid), 0)
            pct = round(100.0 * c / size_u, 4)
            row.append(pct)
            entries.append({
                "user_cluster_id": u_cid,
                "our_cluster_id": o_cid,
                "count": c,
                "percentage_of_user_cluster": pct,
            })
        matrix.append(row)

    return jsonify({
        "ok": True,
        "dataset": dataset,
        "epoch": str(epoch),
        "layer": layer,
        "clustering": clustering,
        "resolved_splits": splits,
        "user_cluster_ids": user_cids,
        "our_cluster_ids": our_cids,
        "matrix": matrix,
        "entries": entries,
    })


@app.route("/cell-cluster-metrics", methods=["POST", "OPTIONS"])
def cell_cluster_metrics():
    """
    Compute external clustering metrics for every (epoch, layer) cell using user labels as ground truth.
    Request body:
      - dataset, clustering, optional data_split
      - user_clustering: list of { "real_id": <id>, "user_cluster_id": <int> }
    Response:
      - metrics_by_cell: { "<epoch>|<layer>": { metric_name: value_or_null, sample_count: n } }
      - metric_names
    """
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    dataset = payload.get("dataset")
    clustering = (payload.get("clustering") or DEFAULT_CLUSTERING).strip().lower()
    if clustering == "pgmeans":
        clustering = "gmm"
    data_split = payload.get("data_split")
    _ = payload.get("use_current")  # deprecated, ignored
    user_clustering = payload.get("user_clustering")
    if not isinstance(user_clustering, list):
        user_clustering = []

    if not dataset or dataset not in DEFAULT_CASES:
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400
    if clustering not in ("gmm", "agglomerative", "hdbscan", "knn"):
        return jsonify({"ok": False, "error": "invalid_clustering"}), 400

    user_rid_to_cluster: Dict[str, int] = {}
    for item in user_clustering:
        if not isinstance(item, dict):
            continue
        rid = item.get("real_id")
        cid = item.get("user_cluster_id")
        if rid is None or cid is None:
            continue
        try:
            user_rid_to_cluster[str(rid)] = int(cid)
        except (ValueError, TypeError):
            continue
    if not user_rid_to_cluster:
        return jsonify({"ok": False, "error": "empty_user_clustering"}), 400

    try:
        from sklearn.metrics import (
            adjusted_mutual_info_score,
            adjusted_rand_score,
            fowlkes_mallows_score,
            homogeneity_completeness_v_measure,
            normalized_mutual_info_score,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": "sklearn_unavailable", "detail": str(e)}), 500

    import csv as csv_module

    metric_names = [
        "ari",
        "ami",
        "nmi",
        "homogeneity",
        "completeness",
        "v_measure",
        "fowlkes_mallows",
    ]

    splits = _resolve_train_test_splits(data_split, default="test")
    # cell_key -> (y_true_list, y_pred_list)
    cell_truth: Dict[str, List[int]] = {}
    cell_pred: Dict[str, List[int]] = {}
    layer_cols: List[str] = []
    saw_csv = False
    checked_paths: List[str] = []
    for split in splits:
        case_dir = _case_dir(dataset, clustering, split)
        csv_path = case_dir / "clustering_metrics" / "cluster_assignments_with_real_id.csv"
        checked_paths.append(str(csv_path))
        if not csv_path.is_file():
            continue
        saw_csv = True
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            fields = list(reader.fieldnames or [])
            layer_cols = [c for c in fields if c.endswith("_cluster")]
            for row in reader:
                rid = row.get("real_id")
                epoch = row.get("epoch")
                if rid is None or epoch is None:
                    continue
                rid_s = str(rid)
                u = None
                for candidate in _split_tagged_id_candidates(rid_s, split=split):
                    u = user_rid_to_cluster.get(candidate)
                    if u is not None:
                        break
                if u is None:
                    continue
                epoch_key = str(epoch)
                for col in layer_cols:
                    raw = row.get(col)
                    if raw in (None, ""):
                        continue
                    try:
                        pred = int(float(raw))
                    except (ValueError, TypeError):
                        continue
                    if pred < 0:
                        continue
                    layer = col[: -len("_cluster")]
                    cell_key = f"{epoch_key}|{layer}"
                    if cell_key not in cell_truth:
                        cell_truth[cell_key] = []
                        cell_pred[cell_key] = []
                    cell_truth[cell_key].append(u)
                    cell_pred[cell_key].append(pred)
    if not saw_csv:
        return jsonify({"ok": False, "error": "cluster_assignments_not_found", "checked_paths": checked_paths}), 404

    metrics_by_cell: Dict[str, Dict[str, Optional[float]]] = {}
    for cell_key, y_true in cell_truth.items():
        y_pred = cell_pred.get(cell_key, [])
        n = min(len(y_true), len(y_pred))
        result: Dict[str, Optional[float]] = {m: None for m in metric_names}
        result["sample_count"] = float(n)
        if n < 2:
            metrics_by_cell[cell_key] = result
            continue

        labels_true = y_true[:n]
        labels_pred = y_pred[:n]
        unique_true = len(set(labels_true))
        unique_pred = len(set(labels_pred))
        if unique_true < 1 or unique_pred < 1:
            metrics_by_cell[cell_key] = result
            continue

        try:
            result["ari"] = float(adjusted_rand_score(labels_true, labels_pred))
            result["ami"] = float(adjusted_mutual_info_score(labels_true, labels_pred))
            result["nmi"] = float(normalized_mutual_info_score(labels_true, labels_pred))
            h, c, v = homogeneity_completeness_v_measure(labels_true, labels_pred)
            result["homogeneity"] = float(h)
            result["completeness"] = float(c)
            result["v_measure"] = float(v)
            result["fowlkes_mallows"] = float(fowlkes_mallows_score(labels_true, labels_pred))
        except Exception:
            # Keep nulls for unstable/degenerate cells.
            pass
        metrics_by_cell[cell_key] = result

    return jsonify({
        "ok": True,
        "dataset": dataset,
        "clustering": clustering,
        "resolved_splits": splits,
        "metric_names": metric_names,
        "layer_columns": layer_cols,
        "metrics_by_cell": metrics_by_cell,
    })


@app.route("/cluster-tanimoto", methods=["POST", "OPTIONS"])
def cluster_tanimoto():
    """
    Compute per-cluster average pairwise Tanimoto for one (epoch, layer) cell.
    Request body:
      - dataset: chemprop|chemeleon
      - epoch: model/epoch key
      - layer: layer key
      - clustering: pgmeans|gmm|agglomerative|hdbscan|knn (optional, default pgmeans)
      - data_split: train|test|val|train+test|all (optional, default test)
      - radius: Morgan radius (default 2)
      - n_bits: fingerprint bits (default 2048)
    """
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    dataset = payload.get("dataset")
    epoch = payload.get("epoch")
    layer = payload.get("layer")
    clustering = str(payload.get("clustering") or DEFAULT_CLUSTERING).strip().lower()
    if clustering == "pgmeans":
        clustering = "gmm"
    split_selector = payload.get("data_split")
    splits = _expand_split_selector(split_selector)

    try:
        radius = int(payload.get("radius", 2))
        n_bits = int(payload.get("n_bits", 2048))
    except Exception:
        return jsonify({"ok": False, "error": "invalid_fingerprint_params"}), 400

    if not dataset or dataset not in DEFAULT_CASES:
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400
    if epoch is None or layer is None:
        return jsonify({"ok": False, "error": "missing_epoch_or_layer"}), 400
    if clustering not in ("gmm", "agglomerative", "hdbscan", "knn"):
        return jsonify({"ok": False, "error": "invalid_clustering"}), 400
    if not splits:
        return jsonify({"ok": False, "error": "invalid_data_split"}), 400

    try:
        from rdkit import Chem  # noqa: F401
    except Exception as e:
        return jsonify({"ok": False, "error": "rdkit_unavailable", "detail": str(e)}), 500

    cluster_rows, err = _load_cluster_rows_for_cell(
        dataset=dataset,
        clustering=clustering,
        epoch=str(epoch),
        layer=str(layer),
        splits=splits,
    )
    if err == "cluster_assignments_not_found":
        return jsonify({"ok": False, "error": err}), 404
    if not cluster_rows:
        return jsonify({"ok": False, "error": "cell_not_found_or_empty"}), 404

    smiles_by_split: Dict[str, Dict[str, str]] = {}
    for split in splits:
        smiles_by_split[split] = _load_id_to_smiles(dataset, data_split=split)

    cluster_to_smiles: Dict[int, List[str]] = {}
    missing_smiles = 0
    for row in cluster_rows:
        rid = str(row.get("real_id"))
        cid = int(row.get("cluster_id"))
        split = str(row.get("split"))
        smi = smiles_by_split.get(split, {}).get(rid, "")
        if not smi:
            missing_smiles += 1
            continue
        cluster_to_smiles.setdefault(cid, []).append(smi)

    clusters = []
    for cid in sorted(cluster_to_smiles.keys()):
        smi_list = cluster_to_smiles[cid]
        avg_sim, n_valid, n_pairs = _average_tanimoto(smi_list, radius=radius, n_bits=n_bits)
        clusters.append({
            "cluster_id": int(cid),
            "tanimoto": round(float(avg_sim), 6),
            "n_molecules": len(smi_list),
            "n_valid_molecules": int(n_valid),
            "n_pairs": int(n_pairs),
        })

    return jsonify({
        "ok": True,
        "dataset": dataset,
        "epoch": str(epoch),
        "layer": str(layer),
        "clustering": clustering,
        "data_split": str(split_selector) if split_selector is not None else "test",
        "resolved_splits": splits,
        "radius": radius,
        "n_bits": n_bits,
        "n_rows": len(cluster_rows),
        "n_missing_smiles": int(missing_smiles),
        "clusters": clusters,
    })


def _load_embedding_cell(dataset, epoch, layer, data_split: Optional[str] = None):
    """Load embedding matrix and real_ids for (dataset, epoch, layer). Returns (X, real_ids)."""
    pt_path = EMBEDDING_PATHS.get(dataset)
    if not pt_path or not pt_path.is_file():
        raise FileNotFoundError("Embedding .pt not found for dataset=%s: %s" % (dataset, pt_path))
    from embedding_loader import load_embedding_input
    data = load_embedding_input(pt_path, data_split=data_split)
    cells = data.get("cells") or {}
    key = (str(epoch), str(layer))
    X = cells.get(key)
    if X is None:
        raise KeyError("No embedding for (epoch=%s, layer=%s) in %s" % (epoch, layer, dataset))
    real_ids = data.get("real_ids")
    if real_ids is None:
        real_ids = list(range(len(X)))
    return X, list(real_ids)


def _cluster_representatives(X, labels, k=5):
    """Per-cluster: top-k closest/farthest indices (global). Returns dict cluster_id -> {top_k_closest, top_k_farthest, k, sorted_indices}."""
    import numpy as np
    from scipy.spatial.distance import cdist
    X = np.asarray(X)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    uniq = np.unique(labels)
    reps = {}
    for c in uniq:
        mask = labels == c
        pts = X[mask]
        idx_global = np.where(mask)[0]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0, keepdims=True)
        d = cdist(pts, center).ravel()
        order = np.argsort(d)
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
        reps[int(c)] = {"top_k_closest": closest, "top_k_farthest": farthest, "k": n, "sorted_indices": sorted_indices}
    return reps


def _substructures_for_cell(X, real_ids, labels, epoch, layer, id_to_smiles, k_rep=5, max_members=None):
    """Compute MCS substructure entries for one (epoch, layer) cell. Returns list of dicts matching cluster_substructures.json shape.

    k_rep controls how many nearest/farthest points per cluster are considered in total.
    max_members, if set, limits how many of those are used for MCS (first max_members from nearest/farthest)."""
    if not id_to_smiles:
        return []
    try:
        from rdkit import Chem
    except Exception:
        return []
    try:
        from mcs_common import common_substructure_smarts_smiles
    except Exception:
        common_substructure_smarts_smiles = None
    # Increase representative window if max_members is larger than k_rep so we have enough candidates.
    k_use = max(k_rep, int(max_members) if max_members is not None else k_rep)
    reps = _cluster_representatives(X, labels, k=k_use)
    real_ids = list(real_ids)
    n = len(real_ids)

    def get_smi(rid):
        v = id_to_smiles.get(rid) or id_to_smiles.get(str(rid))
        if v is not None:
            return v
        try:
            return id_to_smiles.get(int(rid))
        except (TypeError, ValueError):
            return None

    substruct_list = []
    for cluster_id, rep in reps.items():
        for group, key in [("knn", "top_k_closest"), ("kfn", "top_k_farthest")]:
            idx_list = list(rep[key])
            if max_members is not None and max_members > 0:
                idx_list = idx_list[: int(max_members)]
            rids = [real_ids[i] if i < n else i for i in idx_list]
            mols = []
            missing = []
            invalid = []
            seen_canonical = set()
            for rid in rids:
                smi = get_smi(rid)
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
            mcs_note = None
            if mcs_smarts is None and mcs_smiles is None and len(mols) > 0:
                mcs_note = "No common substructure found"
            elif missing:
                mcs_note = "Missing SMILES for {} real_id(s)".format(len(missing))
            substruct_list.append({
                "layer_name": str(layer),
                "model_type": str(epoch),
                "cluster_id": int(cluster_id),
                "group": group,
                "k": rep["k"],
                "real_ids": rids,
                "valid_molecules": len(mols),
                "missing_real_ids": missing,
                "invalid_smiles_real_ids": invalid,
                "mcs_smarts": mcs_smarts,
                "mcs_smiles": mcs_smiles,
                "mcs_note": mcs_note,
            })
    return substruct_list


def _parse_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _load_id_to_smiles(dataset: str, data_split: Optional[str] = None) -> Dict[str, str]:
    splits = _resolve_train_test_splits(data_split, default="test")
    out: Dict[str, str] = {}

    # Prefer consolidated metadata file when available.
    meta_path = DATA_ROOT / dataset / "id_smiles_y_true_y_pred.json"
    if meta_path.is_file():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            raw = []
        if isinstance(raw, list):
            allow = set(splits)
            for row in raw:
                if not isinstance(row, dict):
                    continue
                rid = row.get("real_id", row.get("id"))
                smi = row.get("smiles")
                sp = str(row.get("split", "")).strip().lower()
                if rid is None or smi is None:
                    continue
                if sp and sp not in allow:
                    continue
                rid_s = str(rid)
                smi_s = str(smi)
                if rid_s not in out:
                    out[rid_s] = smi_s
                if sp in ("train", "test"):
                    out[f"{sp}::{rid_s}"] = smi_s
            if out:
                return out

    # Fallback to per-split real_id_to_smiles.json
    for split in splits:
        mapping_path = DATA_ROOT / dataset / split / "real_id_to_smiles.json"
        if not mapping_path.is_file():
            continue
        try:
            raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, list):
            for row in raw:
                if not isinstance(row, dict):
                    continue
                rid = row.get("real_id")
                smi = row.get("smiles")
                if rid is None or smi is None:
                    continue
                rid_s = str(rid)
                smi_s = str(smi)
                if rid_s not in out:
                    out[rid_s] = smi_s
                out[f"{split}::{rid_s}"] = smi_s
            continue
        if isinstance(raw, dict):
            for k, v in raw.items():
                rid_s = str(k)
                smi_s = str(v)
                if rid_s not in out:
                    out[rid_s] = smi_s
                out[f"{split}::{rid_s}"] = smi_s
    return out


def _load_cluster_rows_for_cell(
    dataset: str,
    clustering: str,
    epoch: str,
    layer: str,
    splits: List[str],
) -> Tuple[List[Dict[str, object]], Optional[str]]:
    """
    Load selected-split cluster assignment rows for one cell.
    Row format: {"real_id": str, "cluster_id": int, "split": str}.
    """
    import csv as csv_module

    layer_col = "{}_cluster".format(str(layer).strip())
    out: List[Dict[str, object]] = []
    saw_csv = False
    for split in splits:
        case_dir = _case_dir(dataset, clustering, split)
        csv_path = case_dir / "clustering_metrics" / "cluster_assignments_with_real_id.csv"
        if not csv_path.is_file():
            continue
        saw_csv = True
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                if str(row.get("epoch")) != str(epoch):
                    continue
                rid = row.get("real_id")
                if rid in (None, ""):
                    continue
                try:
                    cid = int(float(row.get(layer_col)))
                except (ValueError, TypeError):
                    continue
                if cid < 0:
                    continue
                out.append({"real_id": str(rid), "cluster_id": int(cid), "split": split})
    if not saw_csv:
        return [], "cluster_assignments_not_found"
    return out, None


def _split_tagged_id_candidates(rid: str, split: Optional[str] = None) -> List[str]:
    rid_s = str(rid)
    out: List[str] = []
    if split in ("train", "test", "val"):
        out.append(f"{split}::{rid_s}")
    out.append(rid_s)
    raw = rid_s.strip()
    if raw != rid_s:
        out.append(raw)
    return out


def _average_tanimoto(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> Tuple[float, int, int]:
    """
    Mean pairwise Tanimoto for one cluster.
    Returns: (mean_similarity, n_valid_molecules, n_pairs).
    """
    from rdkit import Chem
    from rdkit.Chem import DataStructs, rdMolDescriptors

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(_normalize_smiles(str(smi)))
        if mol is None:
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
        fps.append(fp)

    n = len(fps)
    if n < 2:
        return 0.0, n, 0

    total = 0.0
    pairs = 0
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        total += float(sum(sims))
        pairs += len(sims)
    return (total / pairs) if pairs > 0 else 0.0, n, pairs


def _mcs_for_real_ids(real_ids: List[object], id_to_smiles: Dict[str, str]) -> Dict[str, object]:
    try:
        from rdkit import Chem
    except Exception:
        return {
            "valid_molecules": 0,
            "missing_real_ids": [str(x) for x in real_ids],
            "invalid_smiles_real_ids": [],
            "mcs_smarts": None,
            "mcs_smiles": None,
            "mcs_note": "RDKit unavailable",
        }
    try:
        from mcs_common import common_substructure_smarts_smiles
    except Exception:
        common_substructure_smarts_smiles = None

    missing: List[str] = []
    invalid: List[str] = []
    mols = []
    seen_canonical = set()
    for rid in real_ids:
        rid_str = str(rid)
        smi = id_to_smiles.get(rid_str)
        if not smi:
            missing.append(rid_str)
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(rid_str)
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        mols.append(mol)

    smiles_list = [Chem.MolToSmiles(m, canonical=True) for m in mols]
    if common_substructure_smarts_smiles and smiles_list:
        mcs_smarts, mcs_smiles = common_substructure_smarts_smiles(
            smiles_list,
            timeout_s=30,
            ring_matches_ring_only=True,
            complete_rings_only=True,
            match_valences=True,
            atom_compare="elements",
            bond_compare="order",
        )
    else:
        mcs_smarts, mcs_smiles = None, None

    mcs_note = None
    if not smiles_list:
        mcs_note = "No valid molecules for MCS"
    elif mcs_smarts is None and mcs_smiles is None:
        mcs_note = "No common substructure found"
    elif missing:
        mcs_note = "Missing SMILES for {} real_id(s)".format(len(missing))
    return {
        "valid_molecules": len(mols),
        "missing_real_ids": missing,
        "invalid_smiles_real_ids": invalid,
        "mcs_smarts": mcs_smarts,
        "mcs_smiles": mcs_smiles,
        "mcs_note": mcs_note,
    }


@app.route("/substructures/cluster-topm", methods=["GET", "POST", "OPTIONS"])
@app.route("/substructures/cell-topm", methods=["GET", "POST", "OPTIONS"])
def substructures_cluster_topm():
    """
    Return top-m substructures from sorted cluster members for a single cell (epoch, layer).
    - If cluster_id is provided: return that single cluster.
    - If cluster_id is omitted or "all": return all clusters in the cell.
    Supports GET query or POST JSON with: dataset, epoch, layer, m,
    optional: cluster_id, group (knn|kfn|both), clustering, data_split.
    """
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    args = request.args

    def pick(name: str, default=None):
        if request.method == "GET":
            return args.get(name, default)
        return payload.get(name, args.get(name, default))

    dataset = pick("dataset")
    epoch = pick("epoch")
    layer = pick("layer")
    cluster_id_raw = pick("cluster_id")
    m_raw = pick("m", 32)
    group = str(pick("group", "both")).strip().lower()
    clustering = str(pick("clustering", DEFAULT_CLUSTERING)).strip().lower()
    data_split = pick("data_split") or pick("preferred_split")
    _ = pick("use_current", None)  # deprecated, ignored

    if dataset not in DEFAULT_CASES:
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400
    if epoch is None or layer is None:
        return jsonify({"ok": False, "error": "missing_args"}), 400
    if group not in ("knn", "kfn", "both"):
        return jsonify({"ok": False, "error": "invalid_group"}), 400
    cluster_all = False
    cluster_id = None
    if cluster_id_raw is None or (isinstance(cluster_id_raw, str) and cluster_id_raw.strip().lower() == "all"):
        cluster_all = True
    else:
        try:
            cluster_id = int(cluster_id_raw)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "invalid_cluster_id"}), 400
    m_all = False
    if isinstance(m_raw, str) and m_raw.strip().lower() == "all":
        m_all = True
        m = None
    else:
        try:
            m = max(1, min(500, int(m_raw)))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "invalid_m"}), 400

    splits = _resolve_train_test_splits(data_split, default="test")
    targets = []
    checked_paths: List[str] = []
    for split in splits:
        case_dir = _case_dir(dataset, clustering, split)
        sorted_path = case_dir / "topk" / "cluster_sorted_members.json"
        checked_paths.append(str(sorted_path))
        if not sorted_path.is_file():
            continue
        rows = _read_json(sorted_path, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("model_type")) != str(epoch):
                continue
            if str(row.get("layer_name")) != str(layer):
                continue
            try:
                row_cluster = int(row.get("cluster_id"))
            except (TypeError, ValueError):
                continue
            if cluster_all or row_cluster == cluster_id:
                item = dict(row)
                item["_split"] = split
                targets.append(item)

    if not targets:
        if cluster_all:
            return jsonify({"ok": False, "error": "cell_not_found", "checked_paths": checked_paths}), 404
        return jsonify({"ok": False, "error": "cluster_not_found", "checked_paths": checked_paths}), 404

    id_to_smiles = _load_id_to_smiles(dataset, data_split=data_split)
    out_rows = []
    cluster_summary = []
    by_cluster: Dict[int, List[str]] = {}
    by_cluster_split: Dict[int, List[str]] = {}
    multi_split = len(splits) > 1
    for target in targets:
        row_cluster = int(target.get("cluster_id"))
        split_name = str(target.get("_split", "test"))
        sorted_real_ids = target.get("sorted_real_ids") or []
        if not isinstance(sorted_real_ids, list):
            continue
        tagged = [f"{split_name}::{str(x)}" if multi_split else str(x) for x in sorted_real_ids]
        by_cluster.setdefault(row_cluster, [])
        by_cluster_split.setdefault(row_cluster, [])
        by_cluster[row_cluster].extend(tagged)
        by_cluster_split[row_cluster].extend([split_name] * len(tagged))

    for row_cluster in sorted(by_cluster.keys()):
        sorted_ids = by_cluster.get(row_cluster, [])
        split_flags = by_cluster_split.get(row_cluster, [])
        # Preserve first-seen order and keep split alignment.
        dedup_ids: List[str] = []
        dedup_split: List[str] = []
        seen = set()
        for rid, sp in zip(sorted_ids, split_flags):
            if rid in seen:
                continue
            seen.add(rid)
            dedup_ids.append(rid)
            dedup_split.append(sp)

        m_use = len(dedup_ids) if m_all else min(m, len(dedup_ids))
        knn_ids = dedup_ids[:m_use]
        knn_splits = dedup_split[:m_use]
        kfn_ids = list(reversed(dedup_ids[-m_use:]))
        kfn_splits = list(reversed(dedup_split[-m_use:]))
        cluster_summary.append({
            "cluster_id": row_cluster,
            "total_members": len(dedup_ids),
            "m": int(m_use),
        })
        for grp in ("knn", "kfn"):
            if group != "both" and group != grp:
                continue
            ids = knn_ids if grp == "knn" else kfn_ids
            id_splits = knn_splits if grp == "knn" else kfn_splits
            mcs = _mcs_for_real_ids(ids, id_to_smiles)
            out_rows.append({
                "layer_name": str(layer),
                "model_type": str(epoch),
                "cluster_id": row_cluster,
                "group": grp,
                "k": int(m_use),
                "real_ids": ids,
                "real_id_splits": id_splits,
                "valid_molecules": mcs["valid_molecules"],
                "missing_real_ids": mcs["missing_real_ids"],
                "invalid_smiles_real_ids": mcs["invalid_smiles_real_ids"],
                "mcs_smarts": mcs["mcs_smarts"],
                "mcs_smiles": mcs["mcs_smiles"],
                "mcs_note": mcs["mcs_note"],
            })

    cluster_summary = sorted(cluster_summary, key=lambda x: x["cluster_id"])
    out_rows = sorted(out_rows, key=lambda x: (int(x["cluster_id"]), 0 if x["group"] == "knn" else 1))

    return jsonify({
        "ok": True,
        "dataset": dataset,
        "clustering": clustering,
        "data_split": str(data_split) if data_split is not None else "test",
        "resolved_splits": splits,
        "use_current": False,
        "epoch": str(epoch),
        "layer": str(layer),
        "cluster_id": "all" if cluster_all else int(cluster_id),
        "cluster_count": len(cluster_summary),
        "m_requested": "all" if m_all else int(m),
        "clusters": cluster_summary,
        "rows": out_rows,
    })


@app.route("/recluster-agglomerative", methods=["POST", "OPTIONS"])
def recluster_agglomerative():
    """Re-run Agglomerative clustering for a single cell (epoch, layer). Frontend sends dataset, epoch, layer, k, linkage.
    Returns new labels/assignments and substructures for this cell so the UI can update scatter and substructure panel."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    dataset = payload.get("dataset")
    epoch = payload.get("epoch")
    layer = payload.get("layer")
    k = payload.get("k")
    sub_m = payload.get("substructure_max_members") or payload.get("sub_m")
    linkage = payload.get("linkage", "ward")
    data_split = payload.get("data_split") or payload.get("preferred_split")

    print("[flask-backend] recluster-agglomerative dataset=%s epoch=%s layer=%s k=%s linkage=%s" % (dataset, epoch, layer, k, linkage), flush=True)

    if not dataset or dataset not in DEFAULT_CASES:
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400
    if epoch is None or layer is None:
        return jsonify({"ok": False, "error": "missing_epoch_or_layer"}), 400

    link = (linkage or "ward").lower()
    if link not in ("ward", "complete", "average"):
        link = "ward"

    try:
        X, real_ids = _load_embedding_cell(dataset, epoch, layer, data_split=data_split)
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": "embedding_not_found", "detail": str(e)}), 404
    except KeyError as e:
        return jsonify({"ok": False, "error": "cell_not_found", "detail": str(e)}), 404

    try:
        from clustering import fit_agglomerative
        out = fit_agglomerative(X, n_clusters=k, k_min=2, k_max=15, random_state=42, linkage=link)
    except Exception as e:
        print("[flask-backend] recluster-agglomerative failed: %s" % e, flush=True)
        return jsonify({"ok": False, "error": "clustering_failed", "detail": str(e)}), 500

    labels = out["labels"]
    k_out = int(out["k"])
    assignments = [{"real_id": str(rid), "cluster_id": int(labels[i])} for i, rid in enumerate(real_ids)]

    # Flatten X for representatives/substructures (may be 3D from .pt)
    import numpy as np
    X_flat = np.asarray(X)
    if X_flat.ndim > 2:
        X_flat = X_flat.reshape(X_flat.shape[0], -1)

    id_to_smiles = {}
    case_dir = DATA_ROOT / dataset
    smiles_path = case_dir / "real_id_to_smiles.json"
    if smiles_path.is_file():
        try:
            raw = json.loads(smiles_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                id_to_smiles = {str(r["real_id"]): r["smiles"] for r in raw if isinstance(r, dict) and "real_id" in r and "smiles" in r}
            else:
                id_to_smiles = {str(k): v for k, v in raw.items()}
        except Exception as e:
            print("[flask-backend] recluster load smiles failed: %s" % e, flush=True)

    substructures = _substructures_for_cell(
        X_flat,
        real_ids,
        labels,
        epoch,
        layer,
        id_to_smiles,
        k_rep=5,
        max_members=sub_m,
    )
    if substructures:
        print("[flask-backend] recluster-agglomerative substructures for cell: %s entries" % len(substructures), flush=True)

    return jsonify({
        "ok": True,
        "dataset": dataset,
        "epoch": str(epoch),
        "layer": str(layer),
        "k": k_out,
        "linkage": link,
        "real_ids": [str(r) for r in real_ids],
        "labels": [int(x) for x in labels],
        "assignments": assignments,
        "substructures": substructures,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
    app.run(host="0.0.0.0", port=port, debug=debug)
