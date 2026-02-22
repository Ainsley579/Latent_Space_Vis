#!/usr/bin/env python3
"""
Materialize legacy Sankey compatibility files from a dataset's clustering CSV.

Given:
  <dataset_root>/clustering_metrics/cluster_assignments_with_real_id.csv

Generate under:
  <dataset_root>/sankey/
    - <alias>_sankey_samples.json           (per-model layer-to-layer rows)
    - vertical_model_samples_<layer>.json   (cross-model rows within a layer)
    - vertical_model_samples.json           (all cross-model rows)

This is intentionally generic so the same command works for other datasets
under frontend/public/data/*.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def infer_layers(fieldnames: Sequence[str]) -> List[str]:
    layers: List[str] = []
    for name in fieldnames:
        if name.endswith("_cluster"):
            layers.append(name[: -len("_cluster")])
    return layers


def parse_int(value: str) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def model_alias(model: str) -> str:
    lower = model.lower()
    if "noaug" in lower:
        return "noaug"
    if "10conf" in lower:
        return "10conf"
    alias = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")
    return alias or "model"


def ordered_models(models: Iterable[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for m in models:
        if m and m not in seen:
            unique.append(m)
            seen.add(m)

    def rank(name: str) -> Tuple[int, str]:
        n = name.lower()
        if "10conf" in n:
            return (0, n)
        if "noaug" in n:
            return (1, n)
        return (2, n)

    return sorted(unique, key=rank)


def load_csv(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [r for r in reader]
    return fieldnames, rows


def build_model_sankey_rows(
    rows: Sequence[Dict[str, str]], layers: Sequence[str]
) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        model = (row.get("epoch") or row.get("model_type") or "").strip()
        real_id = row.get("real_id")
        if not model or real_id is None:
            continue
        for i in range(len(layers) - 1):
            l1 = layers[i]
            l2 = layers[i + 1]
            c1 = parse_int(row.get(f"{l1}_cluster", ""))
            c2 = parse_int(row.get(f"{l2}_cluster", ""))
            if c1 is None or c2 is None:
                continue
            out[model].append(
                {
                    "real_id": int(real_id) if str(real_id).isdigit() else str(real_id),
                    "from_layer": l1,
                    "to_layer": l2,
                    "from_cluster": c1,
                    "to_cluster": c2,
                    "from_model": model,
                    "to_model": model,
                }
            )
    return out


def build_vertical_rows(
    rows: Sequence[Dict[str, str]], layers: Sequence[str], source_model: str, target_model: str
) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    by_real_and_model: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        model = (row.get("epoch") or row.get("model_type") or "").strip()
        real_id = row.get("real_id")
        if not model or real_id is None:
            continue
        by_real_and_model[(str(real_id), model)] = row

    real_ids = sorted({rid for rid, _m in by_real_and_model.keys()}, key=lambda x: int(x) if x.isdigit() else x)
    all_rows: List[Dict[str, object]] = []
    per_layer: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for rid in real_ids:
        src = by_real_and_model.get((rid, source_model))
        tgt = by_real_and_model.get((rid, target_model))
        if src is None or tgt is None:
            continue
        rid_value: object = int(rid) if rid.isdigit() else rid
        for layer in layers:
            c1 = parse_int(src.get(f"{layer}_cluster", ""))
            c2 = parse_int(tgt.get(f"{layer}_cluster", ""))
            if c1 is None or c2 is None:
                continue
            item = {
                "real_id": rid_value,
                "from_layer": layer,
                "to_layer": layer,
                "from_model": source_model,
                "to_model": target_model,
                "from_cluster": c1,
                "to_cluster": c2,
            }
            all_rows.append(item)
            per_layer[layer].append(item)
    return all_rows, per_layer


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate legacy Sankey compatibility files under <dataset>/sankey from clustering CSV."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("latentvis/frontend/public/data/dimenet"),
        help="Dataset root directory, e.g. latentvis/frontend/public/data/dimenet",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional explicit clustering CSV path. Defaults to <dataset-root>/clustering_metrics/cluster_assignments_with_real_id.csv",
    )
    parser.add_argument(
        "--vertical-source-model",
        type=str,
        default=None,
        help="Source model for vertical_model_samples_*.json (default: auto pick)",
    )
    parser.add_argument(
        "--vertical-target-model",
        type=str,
        default=None,
        help="Target model for vertical_model_samples_*.json (default: auto pick)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    csv_path = args.csv or (dataset_root / "clustering_metrics" / "cluster_assignments_with_real_id.csv")
    sankey_dir = dataset_root / "sankey"

    if not csv_path.exists():
        raise FileNotFoundError(f"Clustering CSV not found: {csv_path}")

    fieldnames, rows = load_csv(csv_path)
    layers = infer_layers(fieldnames)
    if len(layers) < 1:
        raise ValueError(f"No *_cluster columns found in: {csv_path}")

    models = ordered_models((r.get("epoch") or r.get("model_type") or "" for r in rows))
    if not models:
        raise ValueError(f"No model label found in 'epoch' or 'model_type' column: {csv_path}")

    model_rows = build_model_sankey_rows(rows, layers)
    written: List[Path] = []

    for model in models:
        alias = model_alias(model)
        out_path = sankey_dir / f"{alias}_sankey_samples.json"
        write_json(out_path, model_rows.get(model, []))
        written.append(out_path)

        # Also write a full-name variant for generic datasets.
        full_alias = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")
        if full_alias and full_alias != alias:
            full_path = sankey_dir / f"{full_alias}_sankey_samples.json"
            write_json(full_path, model_rows.get(model, []))
            written.append(full_path)

    src_model = args.vertical_source_model or (models[0] if len(models) >= 1 else "")
    tgt_model = args.vertical_target_model or (models[1] if len(models) >= 2 else "")
    if src_model and tgt_model and src_model != tgt_model:
        all_vertical, per_layer = build_vertical_rows(rows, layers, src_model, tgt_model)
        all_path = sankey_dir / "vertical_model_samples.json"
        write_json(all_path, all_vertical)
        written.append(all_path)

        for layer in layers:
            layer_path = sankey_dir / f"vertical_model_samples_{layer}.json"
            write_json(layer_path, per_layer.get(layer, []))
            written.append(layer_path)

    print(f"Dataset root: {dataset_root}")
    print(f"CSV: {csv_path}")
    print(f"Layers ({len(layers)}): {', '.join(layers)}")
    print(f"Models ({len(models)}): {', '.join(models)}")
    print(f"Wrote {len(written)} files:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
