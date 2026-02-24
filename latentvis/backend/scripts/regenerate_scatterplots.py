#!/usr/bin/env python3
"""
Regenerate scatterplot JSONs (PCA/MDS/UMAP/t-SNE) for an existing case directory.

Why:
- Older runs used --skip-tsne, producing *_tsne.json identical to PCA.
- MDS now uses cosine distance (1 - cosine similarity) in dimensionality_reduction.py.

This script:
- Loads embeddings from the original .pt input via embedding_loader.load_embedding_input()
- Computes 2D coords for pca/mds/umap/tsne for each (epoch, layer) pair that exists in the case's scatterplot dirs
- Rewrites scatterplot JSONs while preserving the existing ordering of real_id and the existing cluster_id values
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# Ensure backend dir is on path when run as scripts/regenerate_scatterplots.py
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import numpy as np

from embedding_loader import load_embedding_input
from dimensionality_reduction import reduce_embedding


def _find_scatter_dirs(case_dir: Path) -> List[Path]:
    out: List[Path] = []
    root = case_dir / "scatterplot"
    if root.is_dir():
        out.append(root)
    clustering = case_dir / "clustering"
    if clustering.is_dir():
        for p in clustering.iterdir():
            d = p / "scatterplot"
            if d.is_dir():
                out.append(d)
    # de-dupe while preserving order
    seen = set()
    uniq = []
    for d in out:
        if str(d) in seen:
            continue
        seen.add(str(d))
        uniq.append(d)
    return uniq


def _pairs_from_scatter_dir(scatter_dir: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for f in sorted(scatter_dir.glob("*_pca.json")):
        stem = f.name[: -len("_pca.json")]
        if "_" not in stem:
            continue
        epoch, layer = stem.split("_", 1)
        pairs.append((epoch, layer))
    # unique
    seen = set()
    uniq = []
    for p in pairs:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _load_reference_points(scatter_dir: Path, epoch: str, layer: str) -> List[dict]:
    p = scatter_dir / f"{epoch}_{layer}_pca.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate PCA/MDS/UMAP/t-SNE scatterplots for a case directory")
    ap.add_argument("--case-dir", required=True, help="Case directory under frontend/public/data/<case>")
    ap.add_argument("--input-pt", required=True, help="Original embeddings .pt file")
    ap.add_argument(
        "--tsne-iters",
        type=int,
        default=350,
        help="t-SNE iterations (will be clamped internally based on n_samples)",
    )
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    case_dir = Path(args.case_dir)
    if not case_dir.is_dir():
        raise FileNotFoundError(str(case_dir))

    scatter_dirs = _find_scatter_dirs(case_dir)
    if not scatter_dirs:
        raise ValueError(f"No scatterplot directories found under {case_dir}")

    # Pairs to compute are driven by what's present on disk (keeps scope bounded)
    needed_pairs: List[Tuple[str, str]] = []
    needed_set = set()
    for sd in scatter_dirs:
        for pair in _pairs_from_scatter_dir(sd):
            if pair in needed_set:
                continue
            needed_set.add(pair)
            needed_pairs.append(pair)

    emb = load_embedding_input(args.input_pt)
    cells: Mapping[Tuple[str, str], np.ndarray] = emb["cells"]
    real_ids: Sequence = emb.get("real_ids") or list(range(emb["n_samples"]))
    rid_to_idx: Dict[str, int] = {str(rid): i for i, rid in enumerate(real_ids)}

    methods = ["pca", "mds", "umap", "tsne"]

    # Compute coords once per (epoch, layer)
    coords_by_pair: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for epoch, layer in needed_pairs:
        X = cells.get((epoch, layer))
        if X is None:
            raise KeyError(f"Missing embeddings for ({epoch}, {layer}) in {args.input_pt}")
        coords: Dict[str, np.ndarray] = {}
        coords["pca"] = reduce_embedding(X, method="pca", random_state=args.random_state)
        coords["mds"] = reduce_embedding(X, method="mds", random_state=args.random_state)
        coords["umap"] = reduce_embedding(X, method="umap", random_state=args.random_state)
        coords["tsne"] = reduce_embedding(
            X,
            method="tsne",
            random_state=args.random_state,
            n_iter=args.tsne_iters,
        )
        # Validate: ensure methods are not identical to PCA
        for m in ("mds", "umap", "tsne"):
            if _maxdiff(coords["pca"], coords[m]) == 0.0:
                raise RuntimeError(f"({epoch}, {layer}) {m} is identical to PCA (likely fallback/bug)")
        coords_by_pair[(epoch, layer)] = coords

    # Rewrite JSONs while preserving cluster_id from existing points
    for sd in scatter_dirs:
        for epoch, layer in _pairs_from_scatter_dir(sd):
            ref_points = _load_reference_points(sd, epoch, layer)
            rids = [str(p["real_id"]) for p in ref_points]
            clusters = [int(p.get("cluster_id", -1)) for p in ref_points]

            # Map to embedding indices (keep exact strings)
            idxs: List[int] = []
            missing: List[str] = []
            for rid in rids:
                i = rid_to_idx.get(rid)
                if i is None:
                    missing.append(rid)
                else:
                    idxs.append(i)
            if missing:
                raise RuntimeError(
                    f"{sd}: ({epoch}, {layer}) has {len(missing)} real_id(s) not found in embeddings (example: {missing[0]!r})"
                )

            C = coords_by_pair[(epoch, layer)]
            for method in methods:
                arr = C[method][idxs, :]
                out_points = []
                for (rid, cid), (x, y) in zip(zip(rids, clusters), arr.tolist()):
                    out_points.append({"real_id": rid, "x": float(x), "y": float(y), "cluster_id": int(cid)})
                (sd / f"{epoch}_{layer}_{method}.json").write_text(json.dumps(out_points, indent=2), encoding="utf-8")

            # Default scatter uses PCA for compatibility with existing UI defaults
            (sd / f"{epoch}_{layer}.json").write_text(
                (sd / f"{epoch}_{layer}_pca.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    print(f"Done. Updated scatterplots in {len(scatter_dirs)} directories under {case_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

