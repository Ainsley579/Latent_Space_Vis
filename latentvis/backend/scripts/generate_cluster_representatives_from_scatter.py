#!/usr/bin/env python3
"""
Generate cluster representatives (top-k closest/farthest) from scatter-by-layer CSV.

This uses 2D scatter coordinates to ensure "outlier" points look like outliers
in the scatterplot (consistent with frontend expectations).
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_scatter_df(scatter_csv, model_type):
    df = pd.read_csv(scatter_csv)
    if "model_type" not in df.columns:
        df["model_type"] = model_type
    return df


def build_representatives(tenconf_csv, noaug_csv, k_default):
    df_ten = _load_scatter_df(tenconf_csv, "dimenet10conf")
    df_no = _load_scatter_df(noaug_csv, "dimenet_noaug")
    df = pd.concat([df_ten, df_no], ignore_index=True, sort=False)

    required = {"real_id", "model_type", "layer_name", "cluster_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: {}".format(sorted(missing)))

    reps = []
    # Ensure numeric types
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").fillna(-1).astype(int)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])

    grouped = df.groupby(["model_type", "layer_name", "cluster_id"], sort=True)
    for (model_type, layer_name, cluster_id), group in grouped:
        if group.empty:
            continue
        # centroid in 2D
        cx = group["x"].mean()
        cy = group["y"].mean()
        d2 = (group["x"] - cx) ** 2 + (group["y"] - cy) ** 2
        group = group.assign(_d2=d2)
        group_sorted = group.sort_values("_d2", ascending=True)

        k = min(int(k_default), len(group_sorted))
        closest = group_sorted.head(k)["real_id"].astype(int).tolist()
        farthest = group_sorted.tail(k)["real_id"].astype(int).tolist()[::-1]

        reps.append(
            {
                "layer_name": layer_name,
                "model_type": model_type,
                "cluster_id": int(cluster_id),
                "top_k_closest_real_ids": closest,
                "top_k_farthest_real_ids": farthest,
                "k": k,
                "note": "Computed from 2D scatter distances (centroid in x/y space)",
            }
        )

    return reps


def main():
    parser = argparse.ArgumentParser(
        description="Generate cluster representatives from scatter_by_layer.csv"
    )
    parser.add_argument(
        "--tenconf",
        type=Path,
        default=Path("latentvis/backend/data/scatter_plots/scatter_by_layer.csv"),
        help="10conf scatter CSV with x/y coords",
    )
    parser.add_argument(
        "--noaug",
        type=Path,
        default=Path("latentvis/backend/data/scatter_plots/scatter_noaug_by_layer.csv"),
        help="noaug scatter CSV with x/y coords",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("latentvis/frontend/public/data/topk/cluster_representatives.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k to keep per cluster (default: 5)",
    )
    args = parser.parse_args()

    reps = build_representatives(args.tenconf, args.noaug, args.k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reps, indent=2), encoding="utf-8")
    print("Wrote {} entries to {}".format(len(reps), args.out))


if __name__ == "__main__":
    main()
