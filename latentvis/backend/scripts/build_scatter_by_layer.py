#!/usr/bin/env python3
"""
Build per-layer scatterplot JSON files from two CSVs (10conf + noaug).

Output JSON format (one file per model + layer):
[
  {"real_id": "123", "x": 0.12, "y": -1.4, "cluster_id": 3},
  ...
]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_json(out_path: Path, rows: List[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def build_files(csv_path: Path, model_name: str, out_dir: Path) -> None:
    rows = _read_csv(csv_path)
    by_layer: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        layer = row.get("layer_name") or row.get("layer") or row.get("Layer")
        if not layer:
            continue
        rid = row.get("real_id", "")
        x = row.get("x", "")
        y = row.get("y", "")
        cluster = row.get("cluster_id", "")
        if x == "" or y == "":
            continue
        try:
            x_f = float(x)
            y_f = float(y)
        except ValueError:
            continue
        try:
            c_i = int(cluster) if cluster != "" else -1
        except ValueError:
            c_i = -1
        by_layer[layer].append(
            {"real_id": str(rid), "x": x_f, "y": y_f, "cluster_id": c_i}
        )

    for layer, items in by_layer.items():
        out_path = out_dir / f"{model_name}_{layer}.json"
        _write_json(out_path, items)
        print(f"Wrote {len(items)} points to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-layer scatterplot JSON files")
    parser.add_argument(
        "--tenconf",
        type=Path,
        default=Path("latentvis/backend/data/scatter_plots/scatter_by_layer.csv"),
        help="10conf CSV path",
    )
    parser.add_argument(
        "--noaug",
        type=Path,
        default=Path("latentvis/backend/data/scatter_plots/scatter_noaug_by_layer.csv"),
        help="noaug CSV path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("latentvis/frontend/public/data/scatterplot"),
        help="Output directory",
    )
    args = parser.parse_args()

    build_files(args.tenconf, "dimenet10conf", args.out_dir)
    build_files(args.noaug, "dimenet_noaug", args.out_dir)


if __name__ == "__main__":
    main()
