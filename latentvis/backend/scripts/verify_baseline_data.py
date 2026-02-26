#!/usr/bin/env python3
"""
Verify baseline data layout for test and train splits (chemprop, chemeleon).
Reports presence of key dirs and file counts so you can confirm Slurm jobs completed
and that the four scatter types (pca, mds, umap, tsne) are present and distinct.
Only .py files; all output in English.
"""
from pathlib import Path
import sys

# Same root as flask_backend: frontend/public/data
BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = BACKEND_ROOT.parent / "frontend" / "public" / "data"
CASES = ["chemprop", "chemeleon"]
SPLITS = ["test", "train"]


def count_json(dir_path: Path, pattern: str = "*.json") -> int:
    if not dir_path.is_dir():
        return 0
    return len(list(dir_path.glob(pattern)))


def main():
    print("Data root:", DATA_ROOT)
    if not DATA_ROOT.is_dir():
        print("ERROR: Data root not found.")
        sys.exit(1)
    all_ok = True
    for case in CASES:
        for split in SPLITS:
            base = DATA_ROOT / case / split
            if not base.is_dir():
                print("[%s/%s] MISSING directory" % (case, split))
                all_ok = False
                continue
            scatter = base / "scatterplot"
            n_scatter = count_json(scatter)
            # Check that we have projection-specific files (pca, mds, umap, tsne) for at least one cell
            projections = {}
            if scatter.is_dir():
                for f in scatter.glob("*.json"):
                    name = f.stem
                    if "_" in name:
                        parts = name.split("_")
                        if len(parts) >= 3 and parts[-1] in ("pca", "mds", "umap", "tsne"):
                            proj = parts[-1]
                            projections[proj] = projections.get(proj, 0) + 1
            n_topk = count_json(base / "topk")
            n_sub = count_json(base / "substructures")
            n_sankey = count_json(base / "sankey")
            n_metrics = 0
            cm_dir = base / "clustering_metrics"
            if cm_dir.is_dir():
                n_metrics = len(list(cm_dir.glob("*.csv"))) + count_json(cm_dir)
            print("[%s/%s] scatter=%d topk=%d substructures=%d sankey=%d clustering_metrics=%d projections=%s"
                  % (case, split, n_scatter, n_topk, n_sub, n_sankey, n_metrics, dict(projections)))
            if n_scatter == 0:
                all_ok = False
            if projections and set(projections.keys()) != {"pca", "mds", "umap", "tsne"}:
                print("  -> WARNING: expected all four projections (pca, mds, umap, tsne) for distinct scatter plots.")
    print("Done." if all_ok else "Some checks failed.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
