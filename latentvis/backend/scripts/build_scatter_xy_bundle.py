#!/usr/bin/env python3
"""
Build a shared XY-only scatter bundle (cluster-agnostic):
  latentvis/backend/data/<dataset>/scatterplot_bundles/xy_all.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]  # latentvis/backend
DATA_ROOT = ROOT / "data"
METHODS = ("pgmeans", "agglomerative", "hdbscan")
PROJECTIONS = ("pca", "mds", "umap", "tsne")


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
    m = clustering.strip().lower()
    method_dir = "pgmeans" if m == "pgmeans" else m
    with_clustering = DATA_ROOT / dataset / split / "clustering" / method_dir
    flat = DATA_ROOT / dataset / split / method_dir
    if with_clustering.exists():
        return with_clustering
    return flat


def ordered_models_layers(dataset: str) -> Tuple[List[str], List[str]]:
    # Prefer pgmeans train all_sankey as canonical ordering.
    for method in METHODS:
        p = case_dir(dataset, "train", method) / "sankey" / "all_sankey.json"
        payload = read_json(p, {})
        if isinstance(payload, dict):
            epoch = payload.get("epoch") or {}
            layer = payload.get("layer") or {}
            if isinstance(epoch, dict) and isinstance(layer, dict) and epoch and layer:
                return [str(k) for k in epoch.keys()], [str(k) for k in layer.keys()]
    return [], []


def load_xy_rows(dataset: str, split: str, model: str, layer: str, projection: str) -> List[Dict[str, object]]:
    for method in METHODS:
        scatter_dir = case_dir(dataset, split, method) / "scatterplot"
        path = scatter_dir / f"{model}_{layer}_{projection}.json"
        rows = read_json(path, None)
        if not isinstance(rows, list):
            continue
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
            except Exception:
                continue
            out.append({"real_id": rid, "x": x, "y": y, "split": split})
        if out:
            return out
    return []


def build_xy_bundle(dataset: str, projections: List[str]) -> Dict[str, object]:
    models, layers = ordered_models_layers(dataset)
    cells: Dict[str, Dict[str, Dict[str, List[Dict[str, object]]]]] = {}
    for m in models:
        for l in layers:
            cell_key = f"{m}__{l}"
            per_proj: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
            for proj in projections:
                rows_train = load_xy_rows(dataset, "train", m, l, proj)
                rows_test = load_xy_rows(dataset, "test", m, l, proj)
                per_proj[proj] = {"xy": [*rows_train, *rows_test]}
            cells[cell_key] = per_proj
    return {
        "ok": True,
        "dataset": dataset,
        "kind": "xy_all",
        "projections": projections,
        "models": models,
        "layers": layers,
        "cells": cells,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build shared XY-only scatter bundle")
    ap.add_argument("--dataset", required=True, help="dataset name, e.g. chemprop")
    ap.add_argument("--projections", default="pca,mds,umap,tsne")
    args = ap.parse_args()

    dataset = str(args.dataset).strip().lower()
    projections = [p.strip().lower() for p in str(args.projections).split(",") if p.strip()]
    projections = [p for p in projections if p in PROJECTIONS]
    if not projections:
        projections = list(PROJECTIONS)

    bundle = build_xy_bundle(dataset, projections)
    out_path = DATA_ROOT / dataset / "scatterplot_bundles" / "xy_all.json"
    write_json(out_path, bundle)
    print(f"Wrote {out_path} (models={len(bundle['models'])}, layers={len(bundle['layers'])}, projections={len(projections)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

