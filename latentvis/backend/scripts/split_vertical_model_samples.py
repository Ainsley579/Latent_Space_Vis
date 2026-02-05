#!/usr/bin/env python3
"""
Split vertical_model_samples.json into per-layer JSON files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Split vertical model samples by layer")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("latentvis/frontend/public/data/sankey/vertical_model_samples.json"),
        help="Input JSON path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("latentvis/frontend/public/data/sankey"),
        help="Output directory",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = json.loads(args.input.read_text(encoding="utf-8"))
    by_layer: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        layer = str(r.get("layer", "unknown"))
        by_layer.setdefault(layer, []).append(r)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for layer, items in by_layer.items():
        out = args.out_dir / f"vertical_model_samples_{layer}.json"
        out.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(items)} rows -> {out}")


if __name__ == "__main__":
    main()
