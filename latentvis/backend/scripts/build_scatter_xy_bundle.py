#!/usr/bin/env python3
"""
Build a shared XY-only scatter bundle (cluster-agnostic):
  latentvis/backend/data/<dataset>/xy_all.json
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


def infer_models_layers_from_dir(scatter_dir: Path, projections: List[str]) -> Tuple[List[str], List[str]]:
    models = set()
    layers = set()
    proj_set = set(projections)
    for p in scatter_dir.glob("*.json"):
        stem = p.stem
        if stem.endswith("_xy"):
            stem = stem[: -len("_xy")]
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        proj = parts[-1].lower()
        if proj not in proj_set:
            continue
        model = parts[0]
        layer = "_".join(parts[1:-1])
        if not model or not layer:
            continue
        models.add(model)
        layers.add(layer)
    model_list = sorted(models, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
    layer_list = sorted(layers)
    return model_list, layer_list


def load_xy_rows_from_dir(
    scatter_dir: Path,
    split: str,
    model: str,
    layer: str,
    projection: str,
) -> List[Dict[str, object]]:
    candidates = [
        scatter_dir / f"{model}_{layer}_{projection}.json",
        scatter_dir / f"{model}_{layer}_{projection}_xy.json",
    ]
    rows = None
    for p in candidates:
        data = read_json(p, None)
        if isinstance(data, list):
            rows = data
            break
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
        except Exception:
            continue
        out.append({"real_id": rid, "x": x, "y": y, "split": split})
    return out


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


def build_xy_bundle_from_dirs(
    dataset: str,
    projections: List[str],
    train_scatter_dir: Path,
    test_scatter_dir: Path | None = None,
) -> Dict[str, object]:
    models, layers = infer_models_layers_from_dir(train_scatter_dir, projections)
    if test_scatter_dir and test_scatter_dir.is_dir():
        m2, l2 = infer_models_layers_from_dir(test_scatter_dir, projections)
        models = sorted(set(models) | set(m2), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
        layers = sorted(set(layers) | set(l2))
    cells: Dict[str, Dict[str, Dict[str, List[Dict[str, object]]]]] = {}
    for m in models:
        for l in layers:
            cell_key = f"{m}__{l}"
            per_proj: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
            for proj in projections:
                rows_train = load_xy_rows_from_dir(train_scatter_dir, "train", m, l, proj)
                rows_test = (
                    load_xy_rows_from_dir(test_scatter_dir, "test", m, l, proj)
                    if test_scatter_dir and test_scatter_dir.is_dir()
                    else []
                )
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
    ap.add_argument(
        "--input-scatter-dir-train",
        default="",
        help="Optional external train scatterplot dir (reads <model>_<layer>_<projection>.json)",
    )
    ap.add_argument(
        "--input-scatter-dir-test",
        default="",
        help="Optional external test scatterplot dir (same naming as train dir)",
    )
    ap.add_argument(
        "--output-path",
        default="",
        help="Optional output path (default: latentvis/backend/data/<dataset>/xy_all.json)",
    )
    args = ap.parse_args()

    dataset = str(args.dataset).strip().lower()
    projections = [p.strip().lower() for p in str(args.projections).split(",") if p.strip()]
    projections = [p for p in projections if p in PROJECTIONS]
    if not projections:
        projections = list(PROJECTIONS)

    train_dir = Path(str(args.input_scatter_dir_train)).expanduser().resolve() if str(args.input_scatter_dir_train).strip() else None
    test_dir = Path(str(args.input_scatter_dir_test)).expanduser().resolve() if str(args.input_scatter_dir_test).strip() else None
    if train_dir is not None:
        if not train_dir.is_dir():
            raise FileNotFoundError(f"--input-scatter-dir-train is not a directory: {train_dir}")
        bundle = build_xy_bundle_from_dirs(dataset, projections, train_dir, test_dir)
    else:
        bundle = build_xy_bundle(dataset, projections)
    out_path = (
        Path(str(args.output_path)).expanduser().resolve()
        if str(args.output_path).strip()
        else (DATA_ROOT / dataset / "xy_all.json")
    )
    write_json(out_path, bundle)
    print(f"Wrote {out_path} (models={len(bundle['models'])}, layers={len(bundle['layers'])}, projections={len(projections)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
