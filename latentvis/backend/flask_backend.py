#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

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
    # Simple normalization: strip whitespace; do not change case.
    return (s or "").strip()


def _case_dir(case: str, clustering: str | None = None) -> Path:
    method = (clustering or DEFAULT_CLUSTERING).strip().lower()
    if method == "pgmeans":
        return DATA_ROOT / case
    return DATA_ROOT / case / "clustering" / method


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
    clustering: str | None = None,
    write_files: bool = True,
) -> Dict[str, object]:
    case_dir = _case_dir(case, clustering)
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
        preview = ids[:20]
        print(
            f"[flask-backend] search-smiles case={case} matched={len(ids)} ids={preview}",
            flush=True,
        )

    print(f"[flask-backend] search-smiles total_ids={total}", flush=True)

    return jsonify({"ok": True, "smiles": smiles, "matches": results, "total_ids": total})


def _filter_sankey_for_range(case: str, min_value: int, max_value: int, clustering: str | None = None) -> Tuple[int, int]:
    """Filter all_sankey.json and sankey_samples.json in-place for a given [min,max] model range.

    Returns (kept_epochs, kept_rows) for logging.
    """
    case_dir = _case_dir(case, clustering)
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

    print(
        f"[flask-backend] generate-range min={min_value}, max={max_value}, dataset={dataset}",
        flush=True,
    )

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
        summary[case] = {
            "kept_epochs": epochs,
            "kept_sample_rows": rows,
        }
        print(
            f"[flask-backend] generate-range case={case} -> kept_epochs={epochs}, kept_sample_rows={rows}",
            flush=True,
        )
    return jsonify({"ok": True, "min": min_int, "max": max_int, "summary": summary})


@app.route("/materialize-sankey-files", methods=["POST", "OPTIONS"])
def materialize_sankey_files():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    dataset = payload.get("dataset")
    clustering = payload.get("clustering", DEFAULT_CLUSTERING)
    include_payload = bool(payload.get("include_payload", False))
    write_files = bool(payload.get("write_files", True))
    cases = [dataset] if isinstance(dataset, str) else list(DEFAULT_CASES)

    summary: Dict[str, Dict[str, int]] = {}
    payload_out: Dict[str, Dict[str, object]] = {}
    for case in cases:
        compat = _materialize_compat_sankey_files(case, clustering=clustering, write_files=write_files)
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
    if dataset not in DEFAULT_CASES:
        return jsonify({"ok": False, "error": "invalid_dataset"}), 400
    path = _case_dir(dataset, clustering) / "sankey" / "all_sankey.json"
    payload = _read_json(path, None)
    if payload is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify(payload)


@app.route("/sankey/model", methods=["GET"])
def sankey_model():
    dataset = request.args.get("dataset", "")
    clustering = request.args.get("clustering", DEFAULT_CLUSTERING)
    model = request.args.get("model", "")
    if dataset not in DEFAULT_CASES or not model:
        return jsonify({"ok": False, "error": "invalid_args"}), 400
    path = _case_dir(dataset, clustering) / "sankey" / f"{model}_sankey_samples.json"
    payload = _read_json(path, None)
    if payload is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify(payload)


@app.route("/sankey/vertical", methods=["GET"])
def sankey_vertical():
    dataset = request.args.get("dataset", "")
    clustering = request.args.get("clustering", DEFAULT_CLUSTERING)
    layer = request.args.get("layer", "")
    if dataset not in DEFAULT_CASES or not layer:
        return jsonify({"ok": False, "error": "invalid_args"}), 400
    path = _case_dir(dataset, clustering) / "sankey" / f"vertical_model_samples_{layer}.json"
    payload = _read_json(path, None)
    if payload is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify(payload)


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
