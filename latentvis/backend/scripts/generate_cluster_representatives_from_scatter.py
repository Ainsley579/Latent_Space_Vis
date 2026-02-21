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


def build_representatives(csv_first, csv_second, k_default, model_name_first="dimenet10conf", model_name_second="dimenet_noaug"):
    """Build cluster representatives from two scatter CSVs. Model names used only when CSV lacks 'model_type' column."""
    print("Loading scatter CSVs...", flush=True)
    df_ten = _load_scatter_df(csv_first, model_name_first)
    df_no = _load_scatter_df(csv_second, model_name_second)
    df = pd.concat([df_ten, df_no], ignore_index=True, sort=False)
    print("Loaded {} + {} rows, building representatives...".format(len(df_ten), len(df_no)), flush=True)

    required = {"real_id", "model_type", "layer_name", "cluster_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: {}".format(sorted(missing)))

    reps = []
    sorted_members_list = []
    # Ensure numeric types
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").fillna(-1).astype(int)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])

    grouped = df.groupby(["model_type", "layer_name", "cluster_id"], sort=True)
    n_groups = grouped.ngroups
    done = 0
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

        # Keep farthest examples disjoint from closest examples so KNN/KFN
        # do not overlap for the same (model, layer, cluster).
        closest_set = set(closest)
        farthest_candidates = group_sorted.sort_values("_d2", ascending=False)["real_id"].astype(int).tolist()
        farthest = []
        for rid in farthest_candidates:
            if rid in closest_set:
                continue
            farthest.append(rid)
            if len(farthest) >= k:
                break

        sorted_real_ids = group_sorted["real_id"].astype(int).tolist()
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
        sorted_members_list.append({
            "layer_name": layer_name,
            "model_type": model_type,
            "cluster_id": int(cluster_id),
            "sorted_real_ids": sorted_real_ids,
            "count": len(sorted_real_ids),
        })
        done += 1
        if done % 20 == 0 or done == n_groups:
            print("  scatter progress: {}/{} clusters".format(done, n_groups), flush=True)

    return reps, sorted_members_list


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
    parser.add_argument(
        "--model-name-1",
        default="dimenet10conf",
        help="Model/epoch label for first CSV when it has no 'model_type' column (default: dimenet10conf)",
    )
    parser.add_argument(
        "--model-name-2",
        default="dimenet_noaug",
        help="Model/epoch label for second CSV when it has no 'model_type' column (default: dimenet_noaug)",
    )
    args = parser.parse_args()
    print("generate_cluster_representatives_from_scatter: starting (CSV -> topk + sorted_members)", flush=True)

    reps, sorted_members_list = build_representatives(
        args.tenconf, args.noaug, args.k,
        model_name_first=args.model_name_1,
        model_name_second=args.model_name_2,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reps, indent=2), encoding="utf-8")
    sorted_out = args.out.parent / "cluster_sorted_members.json"
    sorted_out.write_text(json.dumps(sorted_members_list, indent=2), encoding="utf-8")
    print("Wrote {} entries to {}".format(len(reps), args.out))
    print("Wrote {} entries to {}".format(len(sorted_members_list), sorted_out))


if __name__ == "__main__":
    main()
