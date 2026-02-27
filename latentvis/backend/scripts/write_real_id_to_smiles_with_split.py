#!/usr/bin/env python3
"""
Write real_id_to_smiles.json with a "split" column (train/test) for frontend use.
Includes all train and test rows for the given ESOL .pt file(s). Use for both case studies.

Usage:
  # One case: output path required
  python scripts/write_real_id_to_smiles_with_split.py --input /path/to/chemprop-embeddings.pt --output ../frontend/public/data/chemprop/real_id_to_smiles.json

  # Both case studies (default .pt paths and frontend data dir)
  python scripts/write_real_id_to_smiles_with_split.py --both-cases
"""

import argparse
import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
from embedding_loader import load_real_id_smiles_both_splits

# Default paths when using --both-cases
DEFAULT_CHEMPROP_PT = Path("/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt")
DEFAULT_CHEMELEON_PT = Path("/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt")
FRONTEND_DATA = _BACKEND.resolve().parents[1] / "latentvis" / "frontend" / "public" / "data"


def main():
    ap = argparse.ArgumentParser(
        description="Write real_id_to_smiles.json with real_id, smiles, and split (train/test) for frontend."
    )
    ap.add_argument("--input", type=Path, help="Path to .pt file (ESOL format)")
    ap.add_argument("--output", type=Path, help="Output path for real_id_to_smiles.json")
    ap.add_argument(
        "--both-cases",
        action="store_true",
        help="Write for both chemprop and chemeleon using default .pt paths and frontend/public/data/<case>/real_id_to_smiles.json",
    )
    args = ap.parse_args()

    if args.both_cases:
        if args.input or args.output:
            print("Do not use --input/--output with --both-cases.")
            return 1
        cases = [
            ("chemprop", DEFAULT_CHEMPROP_PT, FRONTEND_DATA / "chemprop" / "real_id_to_smiles.json"),
            ("chemeleon", DEFAULT_CHEMELEON_PT, FRONTEND_DATA / "chemeleon" / "real_id_to_smiles.json"),
        ]
        for name, pt_path, out_path in cases:
            if not pt_path.exists():
                print("Skip {}: .pt not found: {}".format(name, pt_path))
                continue
            rows = load_real_id_smiles_both_splits(pt_path)
            if not rows:
                print("Skip {}: no real_id/smiles rows loaded.".format(name))
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            n_test = sum(1 for r in rows if r.get("split") == "test")
            n_train = sum(1 for r in rows if r.get("split") == "train")
            print("Wrote {} ({} rows: {} test, {} train).".format(out_path, len(rows), n_test, n_train))
        return 0

    if not args.input or not args.output:
        print("Provide --input and --output, or use --both-cases.")
        return 1
    if not args.input.exists():
        print("Input not found:", args.input)
        return 1
    rows = load_real_id_smiles_both_splits(args.input)
    if not rows:
        print("No real_id/smiles rows loaded (not an ESOL .pt with id and smiles in test/train?).")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    n_test = sum(1 for r in rows if r.get("split") == "test")
    n_train = sum(1 for r in rows if r.get("split") == "train")
    print("Wrote {} ({} rows: {} test, {} train).".format(args.output, len(rows), n_test, n_train))
    return 0


if __name__ == "__main__":
    sys.exit(main())
