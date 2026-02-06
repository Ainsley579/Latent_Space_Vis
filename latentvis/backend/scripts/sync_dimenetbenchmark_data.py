#!/usr/bin/env python3
"""
Sync DimenetBenchmark data into frontend public data folder.

Copies files from latent-spaces/DimenetBenchmark/data into a dedicated folder
under latentvis/frontend/public/data without overwriting existing files.
"""

import argparse
import os
import shutil
from pathlib import Path


def sync_data(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.exists():
        raise ValueError("Source directory not found: {}".format(src_dir))

    copied = 0
    skipped = 0
    for root, _, files in os.walk(str(src_dir)):
        root_path = Path(root)
        rel_root = root_path.relative_to(src_dir)
        for filename in files:
            src_path = root_path / filename
            dst_path = dst_dir / rel_root / filename
            if dst_path.exists():
                skipped += 1
                continue
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_path), str(dst_path))
            copied += 1

    print("Copied {} files, skipped {} existing files".format(copied, skipped))


def main():
    default_src = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/DimenetBenchmark/data"
    default_dst = "/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/frontend/public/data/dimenet_benchmark"

    parser = argparse.ArgumentParser(description="Sync DimenetBenchmark data")
    parser.add_argument("--src", type=str, default=default_src, help="Source data directory")
    parser.add_argument("--dst", type=str, default=default_dst, help="Destination data directory")
    args = parser.parse_args()

    sync_data(args.src, args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
