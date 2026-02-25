#!/usr/bin/env python3

import json
import os
import subprocess
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


def _filter_sankey_for_range(case: str, min_value: int, max_value: int, clustering: str | None = None) -> Tuple[int, int]:
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
