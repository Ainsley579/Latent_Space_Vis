#!/usr/bin/env python3
"""
Write real_id_to_smiles.json from an ESOL-format .pt file (e.g. chemprop/chemeleon).
Use when the pipeline has already been run but real_id_to_smiles.json was not yet written.

Usage:
  python scripts/write_real_id_to_smiles.py --input /path/to/chemprop-embeddings.pt --output ../frontend/public/data/chemprop/real_id_to_smiles.json
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embedding_loader import load_embedding_input


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to .pt file")
    parser.add_argument("--output", type=Path, required=True, help="Output path for real_id_to_smiles.json")
    args = parser.parse_args()
    if not args.input.exists():
        print("Input not found:", args.input)
        return 1
    data = load_embedding_input(args.input)
    id_to_smiles = data.get("id_to_smiles")
    if not id_to_smiles:
        print("No id_to_smiles in loaded data (not an ESOL .pt with smiles?).")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({str(k): v for k, v in id_to_smiles.items()}, indent=2),
        encoding="utf-8",
    )
    print("Wrote {} ({} entries).".format(args.output, len(id_to_smiles)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
