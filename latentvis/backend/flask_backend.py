#!/usr/bin/env python3

import json
import os
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
    # Simple normalization: strip whitespace; do not change case.
    return (s or "").strip()


def _search_smiles_in_case(case: str, smiles_query: str) -> List[str]:
    """Return list of real_ids in a given case whose SMILES matches the query."""
    case_dir = DATA_ROOT / case
    mapping_path = case_dir / "real_id_to_smiles.json"
    if not mapping_path.is_file():
        return []
    try:
        mapping: Dict[str, str] = json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover - defensive
        print(f"[flask-backend] failed to load {mapping_path}: {e}", flush=True)
        return []
    q = _normalize_smiles(smiles_query)
    if not q:
        return []
    matches: List[str] = []
    for real_id, smi in mapping.items():
        if _normalize_smiles(str(smi)) == q:
            matches.append(str(real_id))
    return matches


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
    results: Dict[str, List[str]] = {}
    total = 0
    for case in cases:
        ids = _search_smiles_in_case(case, smiles)
        if ids:
            results[case] = ids
            total += len(ids)

    return jsonify({"ok": True, "smiles": smiles, "matches": results, "total_ids": total})


def _filter_sankey_for_range(case: str, min_value: int, max_value: int) -> Tuple[int, int]:
    """Filter all_sankey.json and sankey_samples.json in-place for a given [min,max] model range.

    Returns (kept_epochs, kept_rows) for logging.
    """
    case_dir = DATA_ROOT / case
    sankey_dir = case_dir / "sankey"
    all_sankey_path = sankey_dir / "all_sankey.json"
    samples_path = sankey_dir / "sankey_samples.json"

    kept_epochs = 0
    kept_rows = 0

    if all_sankey_path.is_file():
        try:
            data = json.loads(all_sankey_path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover - defensive
            print(f"[flask-backend] failed to load {all_sankey_path}: {e}", flush=True)
        else:
            epoch_dict = data.get("epoch") or {}
            new_epoch = {}
            for ek, payload in epoch_dict.items():
                try:
                    ek_int = int(ek)
                except (ValueError, TypeError):
                    # Keep non-numeric keys unchanged
                    new_epoch[ek] = payload
                    continue
                if min_value <= ek_int <= max_value:
                    new_epoch[ek] = payload
            kept_epochs = len(new_epoch)
            data["epoch"] = new_epoch
            all_sankey_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if samples_path.is_file():
        try:
            rows = json.loads(samples_path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover - defensive
            print(f"[flask-backend] failed to load {samples_path}: {e}", flush=True)
        else:
            def in_range(model_str: str) -> bool:
                try:
                    v = int(model_str)
                except (ValueError, TypeError):
                    return True  # keep non-numeric
                return (min_value <= v <= max_value)

            filtered = [
                row
                for row in rows
                if in_range(row.get("from_model")) and in_range(row.get("to_model"))
            ]
            kept_rows = len(filtered)
            samples_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    return kept_epochs, kept_rows


@app.route("/generate-range", methods=["POST", "OPTIONS"])
def generate_range():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    min_value = payload.get("min")
    max_value = payload.get("max")
    dataset = payload.get("dataset")  # optional: limit to a single case

    print(f"[flask-backend] generate-range min={min_value}, max={max_value}, dataset={dataset}", flush=True)

    try:
        min_int = int(min_value)
        max_int = int(max_value)
    except (ValueError, TypeError):
        return jsonify({"ok": False, "error": "invalid_range"}), 400
    if min_int > max_int:
        min_int, max_int = max_int, min_int

    cases = [dataset] if isinstance(dataset, str) else list(DEFAULT_CASES)
    summary: Dict[str, Dict[str, int]] = {}
    for case in cases:
        epochs, rows = _filter_sankey_for_range(case, min_int, max_int)
        summary[case] = {"kept_epochs": epochs, "kept_sample_rows": rows}
        print(
            f"[flask-backend] generate-range case={case} -> kept_epochs={epochs}, kept_sample_rows={rows}",
            flush=True,
        )

    return jsonify({"ok": True, "min": min_int, "max": max_int, "summary": summary})


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
