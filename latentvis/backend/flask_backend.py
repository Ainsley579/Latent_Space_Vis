#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, jsonify, request


# Backend dir on path for embedding_loader / clustering
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

app = Flask(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "latentvis" / "frontend" / "public" / "data"
DEFAULT_CASES = ("chemprop", "chemeleon")

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
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
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


def _filter_sankey_for_range(case: str, min_value: int, max_value: int) -> Tuple[int, int]:
    """Filter all_sankey.json and all *_sankey_samples.json / vertical_model_samples_*.json
    in-place for a given [min,max] **cluster ID** range (clustering range, not epoch).

    - all_sankey: keep edges where both source_label and target_label have cluster ID in [min, max].
    - Sample files: keep rows where from_cluster and to_cluster are in [min, max].

    Returns (kept_edges, kept_sample_rows) for logging.
    """
    case_dir = DATA_ROOT / case
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
    """Re-run clustering pipeline with user-supplied k_min/k_max (min/max), then materialize sankey compat files."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    min_value = payload.get("min")
    max_value = payload.get("max")
    dataset = payload.get("dataset")
    clustering = (payload.get("clustering") or "gmm").lower().strip()

    # Frontend may send "pgmeans" for default clustering -> backend uses "gmm"
    if clustering == "pgmeans":
        clustering = "gmm"

    print(
        f"[flask-backend] generate-range k_min={min_value}, k_max={max_value}, dataset={dataset}, clustering={clustering}",
        flush=True,
    )

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

    if not dataset or dataset not in DEFAULT_CASES:
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400
    if clustering not in ("gmm", "agglomerative", "hdbscan"):
        return jsonify({"ok": False, "error": "invalid_clustering"}), 400

    input_path = EMBEDDING_PATHS.get(dataset)
    if not input_path or not input_path.is_file():
        return jsonify({"ok": False, "error": "embedding_not_found", "detail": str(input_path)}), 404

    case_dir = DATA_ROOT / dataset
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
                [sys.executable, str(materialize_script), "--dataset-root", str(base), "--vertical-mode", "adjacent"],
                cwd=str(_BACKEND_DIR.parent),
                check=True,
                capture_output=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutError) as e:
            print(f"[flask-backend] materialize_sankey_compat failed (non-fatal): {e}", flush=True)

    return jsonify({
        "ok": True,
        "min": k_min,
        "max": k_max,
        "dataset": dataset,
        "clustering": clustering,
        "summary": {dataset: {"status": "done"}},
    })


def _load_embedding_cell(dataset, epoch, layer):
    """Load embedding matrix and real_ids for (dataset, epoch, layer). Returns (X, real_ids)."""
    pt_path = EMBEDDING_PATHS.get(dataset)
    if not pt_path or not pt_path.is_file():
        raise FileNotFoundError("Embedding .pt not found for dataset=%s: %s" % (dataset, pt_path))
    from embedding_loader import load_embedding_input
    data = load_embedding_input(pt_path)
    cells = data.get("cells") or {}
    key = (str(epoch), str(layer))
    X = cells.get(key)
    if X is None:
        raise KeyError("No embedding for (epoch=%s, layer=%s) in %s" % (epoch, layer, dataset))
    real_ids = data.get("real_ids")
    if real_ids is None:
        real_ids = list(range(len(X)))
    return X, list(real_ids)


@app.route("/recluster-agglomerative", methods=["POST", "OPTIONS"])
def recluster_agglomerative():
    """Re-run Agglomerative clustering with user-chosen k and linkage (ward/complete/average)."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    dataset = payload.get("dataset")
    epoch = payload.get("epoch")
    layer = payload.get("layer")
    k = payload.get("k")
    linkage = payload.get("linkage", "ward")

    print("[flask-backend] recluster-agglomerative dataset=%s epoch=%s layer=%s k=%s linkage=%s" % (dataset, epoch, layer, k, linkage), flush=True)

    if not dataset or dataset not in DEFAULT_CASES:
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400
    if epoch is None or layer is None:
        return jsonify({"ok": False, "error": "missing_epoch_or_layer"}), 400

    link = (linkage or "ward").lower()
    if link not in ("ward", "complete", "average"):
        link = "ward"

    try:
        X, real_ids = _load_embedding_cell(dataset, epoch, layer)
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
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
