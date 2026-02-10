#!/usr/bin/env python3
"""
Generate common substructures (MCS) for each cluster's K-nearest and K-farthest sets.

This script consumes:
  - cluster_representatives.json (top-k closest/farthest real_id per cluster)
  - a CSV file that maps real_id -> SMILES

It outputs a JSON list with one entry per (cluster, group) where group in {knn, kfn}.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


def _load_smiles_map(csv_path, id_column, smiles_column):
    df = pd.read_csv(csv_path)
    if id_column not in df.columns:
        raise ValueError("id_column not found in CSV: {}".format(id_column))
    if smiles_column not in df.columns:
        raise ValueError("smiles_column not found in CSV: {}".format(smiles_column))

    mapping = {}
    for _, row in df.iterrows():
        rid = row[id_column]
        smiles = row[smiles_column]
        if pd.isna(rid) or pd.isna(smiles):
            continue
        try:
            rid_int = int(rid)
        except Exception:
            continue
        mapping[rid_int] = str(smiles)
    return mapping


def _ids_to_mols(ids, smiles_map):
    mols = []
    missing = []
    invalid = []
    for rid in ids:
        smiles = smiles_map.get(int(rid))
        if not smiles:
            missing.append(int(rid))
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid.append(int(rid))
            continue
        mols.append(mol)
    return mols, missing, invalid


def _compute_mcs(mols, timeout, ring_only, complete_rings):
    if len(mols) < 2:
        return None, None
    res = rdFMCS.FindMCS(
        mols,
        timeout=int(timeout),
        ringMatchesRingOnly=bool(ring_only),
        completeRingsOnly=bool(complete_rings),
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
    )
    smarts = res.smartsString if res and res.smartsString else None
    smiles = None
    if smarts:
        try:
            smi_mol = Chem.MolFromSmarts(smarts)
            if smi_mol:
                smiles = Chem.MolToSmiles(smi_mol)
        except Exception:
            smiles = None
    return smarts, smiles


def main():
    parser = argparse.ArgumentParser(description="Generate cluster MCS substructures")
    parser.add_argument(
        "--topk-json",
        type=Path,
        default=Path("latentvis/frontend/public/data/topk/cluster_representatives.json"),
        help="Path to cluster_representatives.json",
    )
    parser.add_argument(
        "--smiles-csv",
        type=Path,
        required=True,
        help="CSV containing real_id and SMILES columns",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="real_id",
        help="Column name for real_id in the CSV",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="product_smiles",
        help="Column name for SMILES in the CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("latentvis/frontend/public/data/substructures/cluster_substructures.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="RDKit MCS timeout in seconds (per cluster)",
    )
    parser.add_argument(
        "--ring-only",
        action="store_true",
        help="Require ring-to-ring matching in MCS",
    )
    parser.add_argument(
        "--complete-rings",
        action="store_true",
        help="Require complete rings in MCS",
    )
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is not available. Install RDKit to run this script.")

    topk = json.loads(args.topk_json.read_text())
    smiles_map = _load_smiles_map(args.smiles_csv, args.id_column, args.smiles_column)

    out_entries = []
    for entry in topk:
        layer = entry.get("layer_name")
        model = entry.get("model_type")
        cluster_id = int(entry.get("cluster_id"))
        k = int(entry.get("k", 0))

        for group_name, id_key in [("knn", "top_k_closest_real_ids"), ("kfn", "top_k_farthest_real_ids")]:
            ids = entry.get(id_key, [])
            mols, missing, invalid = _ids_to_mols(ids, smiles_map)
            smarts, smiles = _compute_mcs(mols, args.timeout, args.ring_only, args.complete_rings)
            out_entries.append(
                {
                    "layer_name": layer,
                    "model_type": model,
                    "cluster_id": cluster_id,
                    "group": group_name,
                    "k": k,
                    "real_ids": [int(x) for x in ids],
                    "valid_molecules": len(mols),
                    "missing_real_ids": missing,
                    "invalid_smiles_real_ids": invalid,
                    "mcs_smarts": smarts,
                    "mcs_smiles": smiles,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_entries, indent=2), encoding="utf-8")
    print("Wrote {} entries to {}".format(len(out_entries), args.out))


if __name__ == "__main__":
    main()
