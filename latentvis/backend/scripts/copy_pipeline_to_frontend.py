#!/usr/bin/env python3
"""
Copy pipeline output directories into the frontend public data root so the frontend
can serve them at /data/... . Merges multiple case outputs (e.g. chemprop + chemeleon)
so that one frontend instance can use both: each case gets a subdir (chemprop, chemeleon)
and we also merge into the root so the default view shows one case (last copied wins for
root; or we keep both in subdirs and document how to switch).

Usage:
  python copy_pipeline_to_frontend.py --pipeline-dir ./pipeline_output_chemprop
  python copy_pipeline_to_frontend.py --pipeline-dir ./pipeline_output_chemprop --pipeline-dir ./pipeline_output_chemeleon --merge-root chemprop
"""
import argparse
import shutil
from pathlib import Path


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            copy_tree(item, dst / item.name)
        else:
            shutil.copy2(item, dst / item.name)


def main():
    ap = argparse.ArgumentParser(description="Copy pipeline output to frontend public data")
    ap.add_argument("--pipeline-dir", action="append", required=True, help="Pipeline output dir (can repeat)")
    ap.add_argument("--frontend-data", type=Path, default=None, help="Frontend data root (default: latentvis/frontend/public/data)")
    ap.add_argument("--merge-root", type=str, default=None, help="Which case to copy into root (default: first); use 'none' to only copy to subdirs")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent
    if args.frontend_data is not None:
        data_root = Path(args.frontend_data)
    else:
        data_root = backend_dir.parent / "frontend" / "public" / "data"
    data_root = data_root.resolve()
    if not data_root.exists():
        data_root.mkdir(parents=True)

    for i, pd in enumerate(args.pipeline_dir):
        pipeline_dir = Path(pd).resolve()
        if not pipeline_dir.exists():
            print("Skip (not found): {}".format(pipeline_dir))
            continue
        # Infer case name from dir name, e.g. pipeline_output_chemprop -> chemprop
        case_name = pipeline_dir.name.replace("pipeline_output_", "").strip("_") or "case_{}".format(i)
        subdir = data_root / case_name
        print("Copying {} -> {}".format(pipeline_dir, subdir))
        if subdir.exists():
            shutil.rmtree(subdir)
        copy_tree(pipeline_dir, subdir)
        # Optionally merge this case into root so /data/sankey, /data/scatterplot etc. point here
        merge_into_root = (args.merge_root is None and i == 0) or (args.merge_root and args.merge_root.lower() == case_name.lower())
        if args.merge_root and args.merge_root.lower() == "none":
            merge_into_root = False
        if merge_into_root:
            for name in ("sankey", "scatterplot", "bic_aic", "topk", "substructures", "clustering_metrics"):
                src_sub = pipeline_dir / name
                dst_sub = data_root / name
                if src_sub.exists():
                    print("  Merge into root: {} -> {}".format(name, dst_sub))
                    if dst_sub.exists():
                        shutil.rmtree(dst_sub)
                    copy_tree(src_sub, dst_sub)
    print("Done. Frontend data root: {}".format(data_root))


if __name__ == "__main__":
    main()
