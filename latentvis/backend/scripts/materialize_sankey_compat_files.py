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


def reorder_layers(layers: Sequence[str], preferred: Sequence[str]) -> List[str]:
    if not preferred:
        return list(layers)
    seen = set()
    out: List[str] = []
    layer_set = set(layers)
    for layer in preferred:
        if layer in layer_set and layer not in seen:
            out.append(layer)
            seen.add(layer)
    for layer in layers:
        if layer not in seen:
            out.append(layer)
            seen.add(layer)
    return out


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
    if "100conf" in lower:
        return "100conf"
    alias = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")
    return alias or "model"


def ordered_models(models: Iterable[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for m in models:
        if m and m not in seen:
            unique.append(m)
            seen.add(m)

    def rank(name: str) -> Tuple[int, object]:
        n = name.lower()
        if "noaug" in n:
            return (0, n)
        if "10conf" in n:
            return (1, n)
        if "100conf" in n:
            return (2, n)
        # Numeric epochs (e.g. 0..99) should keep numeric order.
        if re.fullmatch(r"\d+", n):
            return (3, int(n))
        return (4, n)

    return sorted(unique, key=rank)


def cluster_id_from_label(label: str) -> int:
    m = re.search(r"_C(\d+)$", label)
    return int(m.group(1)) if m else 0


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


def build_vertical_rows_for_pair(
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


def build_vertical_rows_all_pairs(
    rows: Sequence[Dict[str, str]], layers: Sequence[str], ordered: Sequence[str]
) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    all_rows: List[Dict[str, object]] = []
    per_layer: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for i in range(len(ordered) - 1):
        src_model = ordered[i]
        tgt_model = ordered[i + 1]
        pair_rows, pair_per_layer = build_vertical_rows_for_pair(rows, layers, src_model, tgt_model)
        all_rows.extend(pair_rows)
        for layer, items in pair_per_layer.items():
            per_layer[layer].extend(items)

    return all_rows, per_layer


def build_edges_from_cluster_pairs(
    pairs: Sequence[Tuple[int, int]], source_group: str, target_group: str
) -> List[Dict[str, object]]:
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    source_totals: Dict[int, int] = defaultdict(int)
    for c1, c2 in pairs:
        counts[(c1, c2)] += 1
        source_totals[c1] += 1

    if not counts:
        return []

    labels = set()
    for c1, c2 in counts.keys():
        labels.add(f"{source_group}_C{c1}")
        labels.add(f"{target_group}_C{c2}")
    ordered_labels = sorted(
        labels,
        key=lambda s: (s[: s.rfind("_C")] if "_C" in s else s, cluster_id_from_label(s)),
    )
    label_to_id = {label: idx for idx, label in enumerate(ordered_labels)}

    edges: List[Dict[str, object]] = []
    for (c1, c2), value in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        src = f"{source_group}_C{c1}"
        tgt = f"{target_group}_C{c2}"
        total = source_totals.get(c1, 0)
        pct = round((100.0 * value / total), 4) if total > 0 else 0.0
        edges.append(
            {
                "source_node_id": label_to_id[src],
                "target_node_id": label_to_id[tgt],
                "source_label": src,
                "target_label": tgt,
                "value": value,
                "pct_of_source": pct,
            }
        )
    return edges


def build_all_sankey(
    rows: Sequence[Dict[str, str]], layers: Sequence[str], models: Sequence[str]
) -> Dict[str, Dict[str, Dict[str, object]]]:
    by_real_and_model: Dict[Tuple[str, str], Dict[str, str]] = {}
    rows_by_model: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        model = (row.get("epoch") or row.get("model_type") or "").strip()
        real_id = row.get("real_id")
        if not model or real_id is None:
            continue
        rows_by_model[model].append(row)
        by_real_and_model[(str(real_id), model)] = row

    real_ids = sorted({rid for rid, _m in by_real_and_model.keys()}, key=lambda x: int(x) if x.isdigit() else x)
    out: Dict[str, Dict[str, Dict[str, object]]] = {"epoch": {}, "layer": {}}

    for model in models:
        model_rows = rows_by_model.get(model, [])
        edges: List[Dict[str, object]] = []
        for i in range(len(layers) - 1):
            l1 = layers[i]
            l2 = layers[i + 1]
            pairs: List[Tuple[int, int]] = []
            for row in model_rows:
                c1 = parse_int(row.get(f"{l1}_cluster", ""))
                c2 = parse_int(row.get(f"{l2}_cluster", ""))
                if c1 is None or c2 is None:
                    continue
                pairs.append((c1, c2))
            edges.extend(build_edges_from_cluster_pairs(pairs, l1, l2))
        if edges:
            out["epoch"][model] = {"type": "spatial_epoch", "epoch": model, "edges": edges}

    for layer in layers:
        edges = []
        for i in range(len(models) - 1):
            src_model = models[i]
            tgt_model = models[i + 1]
            pairs: List[Tuple[int, int]] = []
            for rid in real_ids:
                src = by_real_and_model.get((rid, src_model))
                tgt = by_real_and_model.get((rid, tgt_model))
                if src is None or tgt is None:
                    continue
                c1 = parse_int(src.get(f"{layer}_cluster", ""))
                c2 = parse_int(tgt.get(f"{layer}_cluster", ""))
                if c1 is None or c2 is None:
                    continue
                pairs.append((c1, c2))
            edges.extend(build_edges_from_cluster_pairs(pairs, src_model, tgt_model))
        if edges:
            out["layer"][layer] = {"type": "spatial_layer", "layer": layer, "edges": edges}

    return out


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
    parser.add_argument(
        "--vertical-mode",
        choices=["auto", "adjacent"],
        default="auto",
        help="auto: explicit source/target if provided else first two; adjacent: generate transitions for all adjacent model pairs",
    )
    parser.add_argument(
        "--layer-order",
        type=str,
        default="",
        help="Optional comma-separated layer order override, e.g. ffn_layer_1,ffn_layer_2,fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3",
    )
    parser.add_argument(
        "--clustering-metrics-subdir",
        type=str,
        default="clustering_metrics",
        help="Subdir under dataset-root for cluster_assignments CSV (default: clustering_metrics; use clustering_metrics_current for interactive runs).",
    )
    parser.add_argument(
        "--sankey-subdir",
        type=str,
        default="sankey",
        help="Subdir under dataset-root for Sankey JSON output (default: sankey; use sankey_current for interactive runs).",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    cm_subdir = args.clustering_metrics_subdir or "clustering_metrics"
    sankey_subdir = args.sankey_subdir or "sankey"
    csv_path = args.csv or (dataset_root / cm_subdir / "cluster_assignments_with_real_id.csv")
    sankey_dir = dataset_root / sankey_subdir

    if not csv_path.exists():
        raise FileNotFoundError(f"Clustering CSV not found: {csv_path}")

    fieldnames, rows = load_csv(csv_path)
    layers = infer_layers(fieldnames)
    if args.layer_order.strip():
        preferred_layers = [s.strip() for s in args.layer_order.split(",") if s.strip()]
        layers = reorder_layers(layers, preferred_layers)
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

    # all_sankey["layer"] key order = layers order (CSV column order = model order when CSV from pipeline).
    all_sankey = build_all_sankey(rows, layers, models)
    all_sankey_path = sankey_dir / "all_sankey.json"
    write_json(all_sankey_path, all_sankey)
    written.append(all_sankey_path)

    sankey_samples: List[Dict[str, object]] = []
    for model in models:
        sankey_samples.extend(model_rows.get(model, []))
    sankey_samples_path = sankey_dir / "sankey_samples.json"
    write_json(sankey_samples_path, sankey_samples)
    written.append(sankey_samples_path)

    all_vertical: List[Dict[str, object]] = []
    per_layer: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    if args.vertical_mode == "adjacent":
        all_vertical, per_layer = build_vertical_rows_all_pairs(rows, layers, models)
    else:
        src_model = args.vertical_source_model or (models[0] if len(models) >= 1 else "")
        tgt_model = args.vertical_target_model or (models[1] if len(models) >= 2 else "")
        if src_model and tgt_model and src_model != tgt_model:
            all_vertical, per_layer = build_vertical_rows_for_pair(rows, layers, src_model, tgt_model)

    if all_vertical:
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
