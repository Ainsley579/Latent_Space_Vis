#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

from flask import Flask, jsonify, request


# Backend dir on path for embedding_loader / clustering
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

app = Flask(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "latentvis" / "frontend" / "public" / "data"
DEFAULT_CASES = ("chemprop", "chemeleon")
DEFAULT_CLUSTERING = "pgmeans"

# Embedding .pt paths (override with env LATENTVIS_CHEMPROP_PT / LATENTVIS_CHEMELEON_PT)
_DEFAULT_PT_ROOT = Path("/global/cfs/projectdirs/m636/nhadler/projects/esol/data")
EMBEDDING_PATHS = {
    "chemprop": Path(os.environ.get("LATENTVIS_CHEMPROP_PT", str(_DEFAULT_PT_ROOT / "chemprop-embeddings.pt"))),
    "chemeleon": Path(os.environ.get("LATENTVIS_CHEMELEON_PT", str(_DEFAULT_PT_ROOT / "chemeleon-embeddings.pt"))),
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


def _case_dir(case: str, clustering: Optional[str] = None, data_split: Optional[str] = None) -> Path:
    """
    Resolve case directory for a given dataset, clustering method, and data split.

    Paths are regularized: both test and train live under a split segment.
    - DATA_ROOT/<case>/test/...  (default when split is None or "test")
    - DATA_ROOT/<case>/train/...
    """
    method = (clustering or DEFAULT_CLUSTERING).strip().lower()
    split = _normalize_split(data_split) or "test"
    root = DATA_ROOT / case / split
    if method == "pgmeans":
        return root
    return root / "clustering" / method


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

    # model -> rows
    model_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in models}
    for row in samples:
        fm = str(row.get("from_model", "")).strip()
        tm = str(row.get("to_model", "")).strip()
        if fm and tm and fm == tm:
            model_rows.setdefault(fm, []).append(row)
        elif fm:
            model_rows.setdefault(fm, []).append(row)

    if write_files:
        # Write per-model files.
        for m in models:
            _write_json(sankey_dir / f"{m}_sankey_samples.json", model_rows.get(m, []))

    # Build per-real-id cluster assignments per model/layer from sample rows.
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


def _search_smiles_substructure_in_case(case: str, smiles_query: str):
    """
    Substructure search: return list of real_ids whose molecule contains the query
    (query is a substructure; all IDs that contain it are returned).
    Uses RDKit HasSubstructMatch. Returns (matches, error_message).
    """
    case_dir = DATA_ROOT / case
    mapping_path = case_dir / "real_id_to_smiles.json"
    if not mapping_path.is_file():
        return [], None
    try:
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            mapping = {str(r["real_id"]): r["smiles"] for r in raw if isinstance(r, dict) and "real_id" in r and "smiles" in r}
        else:
            mapping = {str(k): v for k, v in raw.items()}
    except Exception as e:
        print("[flask-backend] failed to load %s: %s" % (mapping_path, e), flush=True)
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
    for real_id, smi in mapping.items():
        smi = _normalize_smiles(str(smi))
        if not smi:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and mol.HasSubstructMatch(query_mol):
                matches.append(str(real_id))
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

    print(f"[flask-backend] search-smiles smiles={smiles}, dataset={dataset}", flush=True)

    if not isinstance(smiles, str) or not smiles.strip():
        return jsonify({"ok": False, "error": "invalid_smiles"}), 400

    cases = [dataset] if isinstance(dataset, str) else list(DEFAULT_CASES)
    results = {}
    total = 0
    for case in cases:
        ids, err = _search_smiles_substructure_in_case(case, smiles)
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


@app.route("/generate-range", methods=["POST", "OPTIONS"])
def generate_range():
    """Re-run clustering for ALL cells with user-supplied params. Updates scatter, topk, cluster_sorted_members,
    substructures, sankey. Accepts: (min, max) for k_min/k_max; or for agglomerative (k, linkage) to apply fixed k and linkage to all cells."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
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

    # Agglomerative with fixed k: apply k and linkage to all cells
    n_clusters_override = None
    linkage_override = None
    if clustering == "agglomerative" and k_fixed is not None:
        try:
            n_clusters_override = int(k_fixed)
            n_clusters_override = max(2, n_clusters_override)
        except (ValueError, TypeError):
            return jsonify({"ok": False, "error": "invalid_k"}), 400
        linkage_override = linkage
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
            return jsonify({"ok": False, "error": "invalid_range"}), 400
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
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400
    if clustering not in ("gmm", "agglomerative", "hdbscan"):
        return jsonify({"ok": False, "error": "invalid_clustering"}), 400

    input_path = EMBEDDING_PATHS.get(dataset)
    if not input_path or not input_path.is_file():
        return jsonify({"ok": False, "error": "embedding_not_found", "detail": str(input_path)}), 404

    split_norm = _normalize_split(data_split) or "test"
    case_dir = DATA_ROOT / dataset / split_norm
    case_dir.mkdir(parents=True, exist_ok=True)

    try:
        from run_pipeline_from_embeddings import run_pipeline

        ok = run_pipeline(
            input_path=str(input_path),
            output_dir=str(case_dir),
            k_min=k_min,
            k_max=k_max,
            clustering_method=clustering,
            skip_tsne=True,
            n_clusters_override=n_clusters_override,
            linkage_override=linkage_override,
            reuse_scatter_coords=True,
            substructure_max_members=sub_m,
            scatter_subdir="scatterplot_current",
            data_split=split_norm,
            write_fmi_cspa=False,
            interactive_output_suffix="_current",
            verbose_dr=False,
        )
        if not ok:
            return jsonify({"ok": False, "error": "pipeline_failed"}), 500
    except Exception as e:
        print(f"[flask-backend] generate-range pipeline failed: {e}", flush=True)
        return jsonify({"ok": False, "error": "pipeline_failed", "detail": str(e)}), 500

    # Base dir where pipeline wrote outputs for this method (root for gmm, clustering/<method> for others)
    base = case_dir if clustering == "gmm" else (case_dir / "clustering" / clustering)
    materialize_script = _BACKEND_DIR / "scripts" / "materialize_sankey_compat_files.py"
    if materialize_script.is_file():
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(materialize_script),
                    "--dataset-root",
                    str(base),
                    "--vertical-mode",
                    "adjacent",
                    "--clustering-metrics-subdir",
                    "clustering_metrics_current",
                    "--sankey-subdir",
                    "sankey_current",
                ],
                cwd=str(_BACKEND_DIR.parent),
                check=True,
                capture_output=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutError) as e:
            print(f"[flask-backend] materialize_sankey_compat failed (non-fatal): {e}", flush=True)

    # Optional: recompute layer-to-layer and epoch-to-epoch transition stats for updated cluster assignments.
    # This is disabled by default for generate-range to keep response time low; run the script manually
    # or from a dedicated job if transition stats are needed.
    if bool(payload.get("compute_transition_stats")):
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
            except (subprocess.CalledProcessError, subprocess.TimeoutError) as e:
                print(f"[flask-backend] compute_layer_transition_stats failed (non-fatal): {e}", flush=True)

    # Collect updated-results summary from interactive output dirs (scatterplot_current, topk_current, etc.)
    updated = _collect_updated_results_summary(base, output_suffix="_current")
    summary = {dataset: {"status": "done", "updated": updated}}

    out = {"ok": True, "min": k_min, "max": k_max, "dataset": dataset, "clustering": clustering, "summary": summary}
    if n_clusters_override is not None:
        out["k"] = n_clusters_override
        out["linkage"] = linkage_override
    return jsonify(out)


def _collect_updated_results_summary(base: Path, output_suffix: str = "") -> Dict[str, object]:
    """Scan pipeline output dir and return counts for scatter, topk, substructures, sankey, cluster_assignments.
    When output_suffix is '_current', scans scatterplot_current, topk_current, etc. (interactive run)."""
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
    use_current = bool(payload.get("use_current"))
    user_clustering = payload.get("user_clustering")
    if not isinstance(user_clustering, list):
        user_clustering = []

    if not dataset or dataset not in DEFAULT_CASES:
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400
    if epoch is None or layer is None:
        return jsonify({"ok": False, "error": "missing_epoch_or_layer"}), 400
    if clustering not in ("gmm", "agglomerative", "hdbscan"):
        return jsonify({"ok": False, "error": "invalid_clustering"}), 400

    case_dir = _case_dir(dataset, clustering, data_split)
    cm_subdir = "clustering_metrics_current" if use_current else "clustering_metrics"
    csv_path = case_dir / cm_subdir / "cluster_assignments_with_real_id.csv"
    if not csv_path.is_file():
        return jsonify({"ok": False, "error": "cluster_assignments_not_found", "path": str(csv_path)}), 404

    import csv as csv_module
    our_rid_to_cluster: Dict[str, int] = {}
    layer_col = "{}_cluster".format(str(layer).strip())
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            if str(row.get("epoch")) != str(epoch):
                continue
            rid = row.get("real_id")
            if rid is None:
                continue
            try:
                val = row.get(layer_col)
                our_rid_to_cluster[str(rid)] = int(float(val)) if val not in (None, "") else -1
            except (ValueError, TypeError):
                our_rid_to_cluster[str(rid)] = -1

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
        "user_cluster_ids": user_cids,
        "our_cluster_ids": our_cids,
        "matrix": matrix,
        "entries": entries,
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
    app.run(host="127.0.0.1", port=5001, debug=True)
