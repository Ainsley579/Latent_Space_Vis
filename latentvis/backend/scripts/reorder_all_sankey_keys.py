#!/usr/bin/env python3
"""
Rewrite all_sankey.json so that the "layer" (and optionally "epoch") keys are in a specified order.
Use this to fix existing all_sankey files that were written with alphabetical or wrong order.
Frontend reads layer order from this file; order must match model layer order.

Usage:
  python3 reorder_all_sankey_keys.py <path/to/all_sankey.json> --layer-order "fingerprint,gnn_layer_1,..." [--reorder-epochs]
  python3 reorder_all_sankey_keys.py <path/to/all_sankey.json> --layer-order-from-csv [--reorder-epochs]
"""
import argparse
import csv
import json
from pathlib import Path


def _epoch_sort_key(x):
    if str(x).isdigit():
        return (0, int(x))
    return (1, str(x))


def reorder_layer_dict(layer_dict, preferred_order):
    """Return a new dict with keys in preferred_order (existing keys only), then any remaining."""
    existing = set(layer_dict.keys())
    out = {}
    for k in preferred_order:
        if k in existing:
            out[k] = layer_dict[k]
    for k in layer_dict:
        if k not in out:
            out[k] = layer_dict[k]
    return out


def reorder_epoch_dict(epoch_dict):
    """Return a new dict with epoch keys sorted numerically."""
    keys = sorted(epoch_dict.keys(), key=_epoch_sort_key)
    return {k: epoch_dict[k] for k in keys}


def _infer_layer_order_from_csv(all_sankey_path):
    """Find cluster_assignments_with_real_id.csv in the same case dir and return layer order from CSV header."""
    # all_sankey_path is e.g. .../case/sankey/all_sankey.json or .../case/clustering/gmm/sankey/all_sankey.json
    p = Path(all_sankey_path).resolve()
    for _ in range(5):
        p = p.parent
        if not p.name:
            break
        # Try clustering_metrics first (baseline), then clustering_metrics_current
        for sub in ("clustering_metrics", "clustering_metrics_current"):
            csv_path = p / sub / "cluster_assignments_with_real_id.csv"
            if csv_path.is_file():
                with csv_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    headers = list(reader.fieldnames or [])
                layers = [h[:-8] for h in headers if h.endswith("_cluster")]
                if layers:
                    return layers
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Reorder 'layer' and 'epoch' keys in all_sankey.json to match model order."
    )
    ap.add_argument(
        "all_sankey",
        type=Path,
        help="Path to all_sankey.json",
    )
    ap.add_argument(
        "--layer-order",
        type=str,
        default="",
        help="Comma-separated layer order, e.g. fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,ffn_layer_1,ffn_layer_2",
    )
    ap.add_argument(
        "--layer-order-from-csv",
        action="store_true",
        help="Infer layer order from cluster_assignments_with_real_id.csv in the same case dir (overrides --layer-order if found).",
    )
    ap.add_argument(
        "--reorder-epochs",
        action="store_true",
        help="Sort epoch keys numerically (default: keep file order).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print would-be key order and exit without writing.",
    )
    args = ap.parse_args()

    path = Path(args.all_sankey).resolve()
    if not path.is_file():
        raise FileNotFoundError("Not a file: {}".format(path))

    data = json.loads(path.read_text(encoding="utf-8"))
    layer_dict = data.get("layer") or {}
    epoch_dict = data.get("epoch") or {}

    preferred_layers = []
    if args.layer_order_from_csv:
        inferred = _infer_layer_order_from_csv(path)
        if inferred:
            preferred_layers = inferred
        else:
            print("Warning: --layer-order-from-csv could not find cluster_assignments CSV; using --layer-order if given.")
    if not preferred_layers and args.layer_order:
        preferred_layers = [s.strip() for s in args.layer_order.split(",") if s.strip()]
    if preferred_layers:
        new_layer = reorder_layer_dict(layer_dict, preferred_layers)
    else:
        new_layer = dict(layer_dict)
        if args.layer_order_from_csv and not args.dry_run:
            print("Warning: no layer order applied (CSV not found and --layer-order not set).")

    if args.reorder_epochs:
        new_epoch = reorder_epoch_dict(epoch_dict)
    else:
        new_epoch = dict(epoch_dict)

    if args.dry_run:
        print("layer keys:", list(new_layer.keys()))
        print("epoch keys:", list(new_epoch.keys()))
        return

    data["layer"] = new_layer
    data["epoch"] = new_epoch
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Updated {path}: layer order = {list(new_layer.keys())}")


if __name__ == "__main__":
    main()
