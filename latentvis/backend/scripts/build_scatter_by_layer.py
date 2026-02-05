#!/usr/bin/env python3
"""
Build per-layer scatterplot JSON files from two CSVs (10conf + noaug).

Output JSON format (one file per model + layer):
[
  {"real_id": "123", "x": 0.12, "y": -1.4, "cluster_id": 3},
  ...
]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
try:
    from typing import Dict, List
except ImportError:
    Dict = dict
    List = list


def _read_csv(path):
    # type: (Path) -> list
    """Read CSV file and return list of dicts."""
    with open(str(path), "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_json(out_path, rows):
    # type: (Path, list) -> None
    """Write JSON to file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def build_files(csv_path, model_name, out_dir):
    # type: (Path, str, Path) -> None
    """Build JSON files from CSV for a single model."""
    rows = _read_csv(csv_path)
    by_layer = defaultdict(list)  # type: dict

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


def main():
    # type: () -> None
    """Main entry point."""
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
