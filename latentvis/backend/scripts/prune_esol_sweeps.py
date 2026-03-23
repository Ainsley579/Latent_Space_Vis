#!/usr/bin/env python3
"""
Remove stale or incomplete ESOL sweep directories without changing the
existing frontend-facing path layout.

Expected layout:
  backend/data/<dataset>/<split>/sweeps/<method>/<tag>/
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Set


DATASETS = ("chemprop", "chemeleon")
SPLITS = ("train", "val", "test")
METHODS = ("pgmeans", "agglomerative", "knn", "hdbscan")
AGG_LINKAGES = ("ward", "complete", "average")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def expected_tags(method):
    if method == "pgmeans":
        return {
            f"kmin_{k_min}_kmax_{k_max}"
            for k_min in range(2, 11)
            for k_max in range(k_min, 11)
        }
    if method == "agglomerative":
        return {
            f"k_{k}_linkage_{linkage}"
            for linkage in AGG_LINKAGES
            for k in range(2, 11)
        }
    if method == "knn":
        return {f"k_{k}" for k in range(2, 11)}
    if method == "hdbscan":
        return {"default"}
    raise ValueError(f"Unsupported method: {method}")


def is_complete_dir(path):
    required_paths = (
        path / "sankey" / "all_sankey.json",
        path / "sankey" / "sankey_samples.json",
        path / "clustering_metrics" / "cluster_assignments_with_real_id.csv",
        path / "scatterplot",
    )
    return all(required.exists() for required in required_paths)


def iter_targets(dataset, split_arg, method_arg):
    splits = SPLITS if split_arg == "all" else (split_arg,)
    methods = METHODS if method_arg == "all" else (method_arg,)
    for split in splits:
        for method in methods:
            yield split, method


def main() -> int:
    args = parse_args()
    backend_dir = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get("LATENTVIS_DATA_ROOT", backend_dir / "data"))
    removed = 0

    for split, method in iter_targets(args.dataset, args.split, args.method):
        method_root = data_root / args.dataset / split / "sweeps" / method
        if not method_root.is_dir():
            print(f"[skip] missing root: {method_root}")
            continue

        keep = expected_tags(method)
        for child in sorted(p for p in method_root.iterdir() if p.is_dir()):
            reason = None
            if child.name not in keep:
                reason = "unexpected_tag"
            elif not is_complete_dir(child):
                reason = "incomplete"

            if reason is None:
                resume_dir = child / ".pipeline_resume"
                if resume_dir.is_dir():
                    print(f"[remove] resume_cache: {resume_dir}")
                    if not args.dry_run:
                        shutil.rmtree(resume_dir)
                continue

            print(f"[remove] {reason}: {child}")
            removed += 1
            if not args.dry_run:
                shutil.rmtree(child)

    print(f"[summary] removed={removed} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
