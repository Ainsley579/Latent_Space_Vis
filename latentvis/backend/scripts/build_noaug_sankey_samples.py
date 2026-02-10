#!/usr/bin/env python3
"""
Build per-sample Sankey transition rows for dimenet_noaug.

Output JSON format:
[
  {
    "real_id": "123",
    "from_layer": "emb",
    "to_layer": "output_block_0",
    "from_cluster": 1,
    "to_cluster": 2
  },
  ...
]
"""
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


def build_rows(path: Path) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows_out
        layers = _infer_layers(reader.fieldnames)
        for row in reader:
            rid = row.get("real_id")
            if rid is None:
                continue
            for i in range(len(layers) - 1):
                l1 = layers[i]
                l2 = layers[i + 1]
                c1 = row.get(f"{l1}_cluster")
                c2 = row.get(f"{l2}_cluster")
                if c1 is None or c2 is None or c1 == "" or c2 == "":
                    continue
                rows_out.append(
                    {
                        "real_id": str(rid),
                        "from_layer": l1,
                        "to_layer": l2,
                        "from_cluster": int(c1),
                        "to_cluster": int(c2),
                    }
                )
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-sample sankey rows for dimenet_noaug")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("latentvis/backend/data/clustering_metrics/cluster_assignments_with_real_id_dimenet_noaug.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("latentvis/frontend/public/data/sankey/noaug_sankey_samples.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    rows = build_rows(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
