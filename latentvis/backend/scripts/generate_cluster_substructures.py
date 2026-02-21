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
import sys
from pathlib import Path

import pandas as pd

# Allow importing backend mcs_common when run from scripts/
_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND = _SCRIPT_DIR.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

try:
    from rdkit import Chem
    from mcs_common import common_substructure_smarts_smiles
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False
    common_substructure_smarts_smiles = None


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
    seen_canonical = set()
    for rid in ids:
        smiles = smiles_map.get(int(rid))
        if not smiles:
            missing.append(int(rid))
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid.append(int(rid))
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        mols.append(mol)
    return mols, missing, invalid


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
        default="smiles",
        help="Column name for SMILES in the CSV (default: smiles for generic case studies)",
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
    parser.add_argument(
        "--no-match-valences",
        action="store_false",
        dest="match_valences",
        default=True,
        help="Do not match atom valence in MCS (default: match valences)",
    )
    parser.add_argument(
        "--atom-compare",
        choices=["elements", "isotopes", "any"],
        default="elements",
        help="RDKit atom compare strategy for MCS",
    )
    parser.add_argument(
        "--bond-compare",
        choices=["order", "any"],
        default="order",
        help="RDKit bond compare strategy for MCS",
    )
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is not available. Install RDKit to run this script.")

    topk = json.loads(args.topk_json.read_text())
    smiles_map = _load_smiles_map(args.smiles_csv, args.id_column, args.smiles_column)

    total_expected = len(topk) * 2  # each entry has knn + kfn
    print("MCS substructures: processing {} cluster entries ({} knn+kfn pairs)...".format(len(topk), total_expected), flush=True)

    out_entries = []
    done = 0
    for entry in topk:
        layer = entry.get("layer_name")
        model = entry.get("model_type")
        cluster_id = int(entry.get("cluster_id"))
        k = int(entry.get("k", 0))

        for group_name, id_key in [("knn", "top_k_closest_real_ids"), ("kfn", "top_k_farthest_real_ids")]:
            ids = entry.get(id_key, [])
            mols, missing, invalid = _ids_to_mols(ids, smiles_map)
            # Use chemistry-expert MCS (mcs_common) to avoid overlapping/ambiguous substructure display
            smiles_list = [Chem.MolToSmiles(m, canonical=True) for m in mols]
            if common_substructure_smarts_smiles:
                smarts, smiles = common_substructure_smarts_smiles(
                    smiles_list,
                    timeout_s=args.timeout,
                    ring_matches_ring_only=args.ring_only,
                    complete_rings_only=args.complete_rings,
                    match_valences=args.match_valences,
                    atom_compare=args.atom_compare,
                    bond_compare=args.bond_compare,
                )
            else:
                smarts, smiles = None, None
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
            done += 1
            if done % 20 == 0 or done == total_expected:
                print("  progress: {}/{}".format(done, total_expected), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_entries, indent=2), encoding="utf-8")
    print("Wrote {} entries to {}".format(len(out_entries), args.out), flush=True)


if __name__ == "__main__":
    main()
