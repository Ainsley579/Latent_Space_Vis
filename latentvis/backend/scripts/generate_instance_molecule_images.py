#!/usr/bin/env python3
"""
Generate per-instance molecule PNGs from real_id + SMILES for one dataset.

Outputs:
  - {out_dir}/{safe_real_id}_{image_type}.png
  - {out_dir}/real_id_to_image_mapping.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


def _safe_real_id(real_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(real_id).strip())


def _read_real_ids_from_scatter(scatter_dir: Path) -> List[str]:
    if not scatter_dir.exists():
        raise FileNotFoundError(f"Scatterplot directory not found: {scatter_dir}")

    candidates = sorted(scatter_dir.glob("*_fingerprint.json"))
    if not candidates:
        candidates = sorted(scatter_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No scatterplot JSON files under: {scatter_dir}")

    # One scatter file already contains all instances for a model/layer cell.
    sample = candidates[0]
    rows = json.loads(sample.read_text())
    ids = [str(row.get("real_id", "")).strip() for row in rows if str(row.get("real_id", "")).strip()]
    if not ids:
        raise ValueError(f"No real_id found in scatter file: {sample}")
    return ids


def _build_smiles_maps(df: pd.DataFrame, id_column: str, smiles_columns: Iterable[str]) -> Dict[str, Dict[str, str]]:
    if id_column not in df.columns:
        raise ValueError(f"id_column not found: {id_column}")
    for c in smiles_columns:
        if c not in df.columns:
            raise ValueError(f"smiles column not found: {c}")

    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        rid = str(row[id_column]).strip()
        if not rid:
            continue
        entry: Dict[str, str] = {}
        for col in smiles_columns:
            val = row[col]
            if pd.isna(val):
                continue
            s = str(val).strip()
            if s:
                entry[col] = s
        if entry:
            out[rid] = entry
    return out


def _candidate_real_id_keys(real_id: str) -> Set[str]:
    key = str(real_id).strip()
    keys = {key}
    try:
        keys.add(str(int(float(key))))
    except Exception:
        pass
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-instance molecule PNG images")
    parser.add_argument("--dataset", required=True, help="Dataset key: chemprop / chemeleon / dimenet")
    parser.add_argument("--smiles-csv", type=Path, required=True, help="CSV containing real_id + smiles columns")
    parser.add_argument("--id-column", default="real_id", help="ID column in smiles CSV")
    parser.add_argument(
        "--smiles-columns",
        default="ligand_smiles",
        help="Comma-separated SMILES columns to render. First one will be aliased as ligand if named differently.",
    )
    parser.add_argument("--image-types", default="ligand", help="Comma-separated image type names, same length as --smiles-columns")
    parser.add_argument("--size", type=int, default=300, help="Square image size in px")
    parser.add_argument("--frontend-root", type=Path, default=Path("latentvis/frontend/public"), help="Frontend public root")
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is not available. Use an environment with RDKit installed.")
    if not args.smiles_csv.exists():
        raise FileNotFoundError(f"SMILES CSV not found: {args.smiles_csv}")

    smiles_columns = [s.strip() for s in args.smiles_columns.split(",") if s.strip()]
    image_types = [s.strip() for s in args.image_types.split(",") if s.strip()]
    if len(smiles_columns) != len(image_types):
        raise ValueError("--smiles-columns and --image-types must have same number of entries")

    dataset_root = args.frontend_root / "data" / args.dataset
    scatter_dir = dataset_root / "scatterplot"
    out_dir = dataset_root / "molecule_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    real_ids = _read_real_ids_from_scatter(scatter_dir)
    smiles_df = pd.read_csv(args.smiles_csv)
    smiles_map = _build_smiles_maps(smiles_df, args.id_column, smiles_columns)

    mapping: Dict[str, Dict[str, str]] = {}
    total_written = 0
    missing = 0
    invalid = 0

    for rid in real_ids:
        matched: Dict[str, str] = {}
        row_smiles: Dict[str, str] = {}
        for key in _candidate_real_id_keys(rid):
            if key in smiles_map:
                row_smiles = smiles_map[key]
                break
        if not row_smiles:
            missing += 1
            continue

        safe_id = _safe_real_id(rid)
        for smiles_col, image_type in zip(smiles_columns, image_types):
            smi = row_smiles.get(smiles_col)
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid += 1
                continue
            filename = f"{safe_id}_{image_type}.png"
            out_path = out_dir / filename
            img = Draw.MolToImage(mol, size=(args.size, args.size))
            img.save(out_path)
            matched[image_type] = filename
            total_written += 1

        if matched:
            mapping[rid] = matched

    (out_dir / "real_id_to_image_mapping.json").write_text(json.dumps(mapping, indent=2))
    print(f"Dataset: {args.dataset}")
    print(f"Real IDs in scatter: {len(real_ids)}")
    print(f"Mapped IDs: {len(mapping)}")
    print(f"Missing IDs in smiles CSV: {missing}")
    print(f"Invalid SMILES count: {invalid}")
    print(f"Images written: {total_written}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
