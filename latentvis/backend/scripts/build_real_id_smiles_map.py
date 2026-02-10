#!/usr/bin/env python3
"""
Build a real_id -> SMILES CSV for the current case study.

This script uses:
  - cas_to_original_mapping.json (real_id -> original index)
  - HF-Literature.tsv (full dataset with SMILES columns)

It outputs a compact CSV with the columns:
  real_id, product_smiles, ligand_smiles, startingmat_smiles, solvent_smiles
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build real_id -> SMILES map CSV")
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=Path("latentvis/backend/data/smiles/cas_to_original_mapping.json"),
        help="Path to cas_to_original_mapping.json",
    )
    parser.add_argument(
        "--literature-tsv",
        type=Path,
        default=Path("latentvis/backend/data/smiles/HF-Literature.tsv"),
        help="Path to HF-Literature.tsv",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path(
            "latentvis/backend/data/smiles/dimenet-10conf_split_0.json"
        ),
        help="Path to split_0.json (train/val/test) for dimenet-10conf",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("latentvis/backend/data/smiles/real_id_smiles.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    if not args.mapping_json.exists():
        raise FileNotFoundError("Mapping JSON not found: {}".format(args.mapping_json))
    if not args.literature_tsv.exists():
        raise FileNotFoundError("Literature TSV not found: {}".format(args.literature_tsv))
    if not args.split_json.exists():
        raise FileNotFoundError("split_0.json not found: {}".format(args.split_json))

    mapping = json.loads(args.mapping_json.read_text())
    mapping = {int(k): int(v) for k, v in mapping.items()}

    # Build full dataset from split_0.json + HF-Literature.tsv (3518 rows)
    with args.split_json.open() as f:
        split_data = json.load(f)
    train = pd.DataFrame(split_data["train"])
    val = pd.DataFrame(split_data["val"])
    test = pd.DataFrame(split_data["test"])
    full = pd.concat([train, test, val], ignore_index=True)

    lit = pd.read_csv(args.literature_tsv, sep="\t")
    df = full.merge(lit, on="rxn_id", how="left")
    needed = {
        "product_1_smiles",
        "ligand_1_smiles",
        "startingmat_1_smiles",
        "solvent_1_smiles",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError("Missing SMILES columns in literature TSV: {}".format(sorted(missing)))

    rows = []
    for real_id, orig_idx in mapping.items():
        if orig_idx < 0 or orig_idx >= len(df):
            continue
        row = df.iloc[orig_idx]
        rows.append(
            {
                "real_id": real_id,
                "product_smiles": row.get("product_1_smiles"),
                "ligand_smiles": row.get("ligand_1_smiles"),
                "startingmat_smiles": row.get("startingmat_1_smiles"),
                "solvent_smiles": row.get("solvent_1_smiles"),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("real_id")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print("Wrote {} rows to {}".format(len(out_df), args.out))


if __name__ == "__main__":
    main()
