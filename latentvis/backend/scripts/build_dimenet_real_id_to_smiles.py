#!/usr/bin/env python3
"""
Build real_id_to_smiles.json for the CAS-deduplicated DimeNet case.

The DimeNet CAS embeddings are created by taking the concatenated
train + test + val rows and then dropping duplicate ligand_1_cas entries,
keeping the first occurrence. The resulting row order matches the
CAS-deduplicated embedding order used in latent-spaces.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path(
            "/global/cfs/cdirs/m636/nhadler/projects/Libra-ML-Modeling/model-training/butene/dimenet-noaug/data/jsons/split_0.json"
        ),
        help="Path to the DimeNet split_0.json file.",
    )
    parser.add_argument(
        "--literature-tsv",
        type=Path,
        default=Path("/global/cfs/cdirs/m636/nhadler/projects/Libra-ML-Modeling/data/HF-Literature.tsv"),
        help="Path to HF-Literature.tsv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for real_id_to_smiles.json.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV output path with columns real_id,smiles for pipeline substructure generation.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=152,
        help="Expected number of CAS-deduplicated rows. If the deduplicated table is larger, keep the first N rows.",
    )
    args = parser.parse_args()

    if not args.split_json.is_file():
        raise FileNotFoundError(f"split_0.json not found: {args.split_json}")
    if not args.literature_tsv.is_file():
        raise FileNotFoundError(f"HF-Literature.tsv not found: {args.literature_tsv}")

    with args.split_json.open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_df = pd.DataFrame(split_data.get("train", []))
    test_df = pd.DataFrame(split_data.get("test", []))
    val_df = pd.DataFrame(split_data.get("val", []))
    ordered_df = pd.concat([train_df, test_df, val_df], ignore_index=True)

    literature_df = pd.read_csv(args.literature_tsv, sep="\t")
    merged = ordered_df.merge(literature_df, on="rxn_id", how="left")
    if "ligand_1_cas" not in merged.columns or "ligand_1_smiles" not in merged.columns:
        raise ValueError("Expected ligand_1_cas and ligand_1_smiles columns after merge")

    dedup = merged.drop_duplicates(subset=["ligand_1_cas"], keep="first").reset_index(drop=True)
    if args.expected_count > 0:
        if len(dedup) < args.expected_count:
            raise ValueError(
                f"Deduplicated DimeNet metadata has only {len(dedup)} rows, expected at least {args.expected_count}"
            )
        if len(dedup) > args.expected_count:
            dedup = dedup.iloc[: args.expected_count].reset_index(drop=True)
    mapping = {}
    csv_rows = []
    for idx, smiles in enumerate(dedup["ligand_1_smiles"].tolist()):
        if pd.isna(smiles):
            continue
        smiles_str = str(smiles)
        mapping[str(idx)] = smiles_str
        csv_rows.append({"real_id": idx, "smiles": smiles_str})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"Wrote {len(mapping)} real_id->smiles entries to {args.output}")
    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows).to_csv(args.csv_output, index=False)
        print(f"Wrote {len(csv_rows)} real_id/smiles rows to {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
