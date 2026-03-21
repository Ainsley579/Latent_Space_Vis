#!/usr/bin/env python3
"""
Build precomputed scatter bundle JSONs from existing per-cell scatter files.

Output path:
  latentvis/backend/data/<dataset>/scatterplot_bundles/<clustering>_all.json

Bundle schema:
  {
    "ok": true,
    "dataset": "...",
    "clustering": "...",
    "projections": ["pca", ...],
    "models": [...],
    "layers": [...],
    "cells": {
      "<model>__<layer>": {
        "<projection>": {
          "xy": [{"real_id","x","y","split"}...],
          "cluster": [{"real_id","cluster_id","split"}...]
        }
      }
    }
  }
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]  # latentvis/backend
DATA_ROOT = ROOT / "data"
DEFAULT_CLUSTERINGS = ("pgmeans", "agglomerative", "hdbscan")
VALID_PROJECTIONS = ("pca", "mds", "umap", "tsne")


def read_json(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def case_dir(dataset: str, split: str, clustering: str) -> Path:
    method = clustering.strip().lower()
    method_dir = "pgmeans" if method == "pgmeans" else method
    flat = DATA_ROOT / dataset / split / method_dir
    with_clustering = DATA_ROOT / dataset / split / "clustering" / method_dir
    if with_clustering.exists():
        return with_clustering
    return flat


def ordered_models_layers(dataset: str, clustering: str) -> Tuple[List[str], List[str]]:
    p = case_dir(dataset, "train", clustering) / "sankey" / "all_sankey.json"
    payload = read_json(p, {})
    epoch = payload.get("epoch") if isinstance(payload, dict) else {}
    layer = payload.get("layer") if isinstance(payload, dict) else {}
    models = [str(x) for x in (epoch.keys() if isinstance(epoch, dict) else [])]
    layers = [str(x) for x in (layer.keys() if isinstance(layer, dict) else [])]
    return models, layers


def load_combined_rows(path: Path) -> List[Dict[str, object]]:
    rows = read_json(path, [])
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, object]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("real_id", "")).strip()
        if not rid:
            continue
        try:
            x = float(r.get("x", 0) or 0)
            y = float(r.get("y", 0) or 0)
            cid = int(r.get("cluster_id", 0) or 0)
        except Exception:
            continue
        out.append({"real_id": rid, "x": x, "y": y, "cluster_id": cid})
    return out


def build_bundle(dataset: str, clustering: str, projections: List[str]) -> Dict[str, object]:
    models, layers = ordered_models_layers(dataset, clustering)
    cells: Dict[str, Dict[str, Dict[str, List[Dict[str, object]]]]] = {}

    for m in models:
        for l in layers:
            cell_key = f"{m}__{l}"
            per_proj: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
            for proj in projections:
                xy_rows: List[Dict[str, object]] = []
                cluster_rows: List[Dict[str, object]] = []
                for split in ("train", "test"):
                    scatter_dir = case_dir(dataset, split, clustering) / "scatterplot"
                    file_path = scatter_dir / f"{m}_{l}_{proj}.json"
                    rows = load_combined_rows(file_path)
                    for r in rows:
                        xy_rows.append(
                            {
                                "real_id": r["real_id"],
                                "x": r["x"],
                                "y": r["y"],
                                "split": split,
                            }
                        )
                        cluster_rows.append(
                            {
                                "real_id": r["real_id"],
                                "cluster_id": r["cluster_id"],
                                "split": split,
                            }
                        )
                per_proj[proj] = {"xy": xy_rows, "cluster": cluster_rows}
            cells[cell_key] = per_proj

    return {
        "ok": True,
        "dataset": dataset,
        "clustering": clustering,
        "projections": projections,
        "models": models,
        "layers": layers,
        "cells": cells,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build scatter bundle JSON from existing scatterplot files")
    ap.add_argument("--dataset", required=True, help="dataset name, e.g. chemprop")
    ap.add_argument(
        "--clustering",
        default="pgmeans,agglomerative,hdbscan",
        help="comma-separated clustering methods",
    )
    ap.add_argument(
        "--projections",
        default="pca,mds,umap,tsne",
        help="comma-separated projections",
    )
    args = ap.parse_args()

    dataset = str(args.dataset).strip().lower()
    methods = [x.strip().lower() for x in str(args.clustering).split(",") if x.strip()]
    methods = [m for m in methods if m in DEFAULT_CLUSTERINGS or m == "knn"]
    if not methods:
        raise SystemExit("No valid clustering method provided.")
    projections = [x.strip().lower() for x in str(args.projections).split(",") if x.strip()]
    projections = [p for p in projections if p in VALID_PROJECTIONS]
    if not projections:
        raise SystemExit("No valid projection provided.")

    for method in methods:
        bundle = build_bundle(dataset, method, projections)
        out_path = DATA_ROOT / dataset / "scatterplot_bundles" / f"{method}_all.json"
        write_json(out_path, bundle)
        print(f"Wrote {out_path} (models={len(bundle['models'])}, layers={len(bundle['layers'])}, projections={len(projections)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
