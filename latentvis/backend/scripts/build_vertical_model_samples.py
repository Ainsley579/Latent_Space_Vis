#!/usr/bin/env python3
"""
Build per-sample transitions between two models within the same layer.

Output JSON format:
[
  {
    "real_id": "123",
    "layer": "emb",
    "from_model": "dimenet10conf",
    "to_model": "dimenet_noaug",
    "from_cluster": 1,
    "to_cluster": 2
  },
  ...
]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _infer_layers(fieldnames: List[str]) -> List[str]:
    layers: List[str] = []
    for name in fieldnames:
        if name.endswith("_cluster"):
            layers.append(name[: -len("_cluster")])
    return layers


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader]


def build_rows(path_a: Path, path_b: Path, model_a: str, model_b: str) -> List[Dict[str, Any]]:
    rows_a = _read_rows(path_a)
    rows_b = _read_rows(path_b)
    if not rows_a or not rows_b:
        return []

    layers_a = _infer_layers(list(rows_a[0].keys()))
    layers_b = _infer_layers(list(rows_b[0].keys()))
    layers = [l for l in layers_a if l in layers_b]

    rows_a_by_id = {str(r.get("real_id")): r for r in rows_a if r.get("real_id") is not None}
    rows_b_by_id = {str(r.get("real_id")): r for r in rows_b if r.get("real_id") is not None}

    out: List[Dict[str, Any]] = []
    for rid, ra in rows_a_by_id.items():
        rb = rows_b_by_id.get(rid)
        if rb is None:
            continue
        for layer in layers:
            ca = ra.get(f"{layer}_cluster")
            cb = rb.get(f"{layer}_cluster")
            if ca is None or cb is None or ca == "" or cb == "":
                continue
            out.append(
                {
                    "real_id": str(rid),
                    "layer": layer,
                    "from_model": model_a,
                    "to_model": model_b,
                    "from_cluster": int(ca),
                    "to_cluster": int(cb),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-sample transitions between two models within each layer")
    parser.add_argument(
        "--model-a",
        default="dimenet10conf",
        help="Model A name (source)",
    )
    parser.add_argument(
        "--model-b",
        default="dimenet_noaug",
        help="Model B name (target)",
    )
    parser.add_argument(
        "--input-a",
        type=Path,
        default=Path("latentvis/backend/data/clustering_metrics/cluster_assignments_with_real_id.csv"),
        help="CSV for model A",
    )
    parser.add_argument(
        "--input-b",
        type=Path,
        default=Path("latentvis/backend/data/clustering_metrics/cluster_assignments_with_real_id_dimenet_noaug.csv"),
        help="CSV for model B",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("latentvis/frontend/public/data/sankey/vertical_model_samples.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    rows = build_rows(args.input_a, args.input_b, args.model_a, args.model_b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
