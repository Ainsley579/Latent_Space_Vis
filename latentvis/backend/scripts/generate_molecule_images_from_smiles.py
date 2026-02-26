#!/usr/bin/env python3
"""
Generate molecule PNG images from real_id_to_smiles.json for frontend KNN/molecule view.

Use for chemprop and chemeleon case studies so the frontend can show ligand images
under data/<case>/molecule_images/{sanitized_real_id}_ligand.png.

Usage:
  python scripts/generate_molecule_images_from_smiles.py --case-dir ../frontend/public/data/chemprop
  python scripts/generate_molecule_images_from_smiles.py --case-dir ../frontend/public/data/chemeleon
"""

import argparse
import json
import re
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

# Max filename stem length to avoid filesystem issues
MAX_FILENAME_STEM = 180


def sanitize_real_id_for_filename(real_id: str) -> str:
    """Produce a safe filename stem from real_id (may be molecule name with spaces, parens, etc.)."""
    s = str(real_id).strip()
    # Replace spaces and problematic chars with single underscore
    s = re.sub(r"[\s()\[\],;]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    # Keep alnum and -_
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    s = re.sub(r"_+", "_", s).strip("_") or "unknown"
    if len(s) > MAX_FILENAME_STEM:
        s = s[:MAX_FILENAME_STEM].rstrip("_")
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate molecule images from real_id_to_smiles.json")
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Case data directory (e.g. frontend/public/data/chemprop) containing real_id_to_smiles.json",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=240,
        help="Square image size in pixels (default 240)",
    )
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        print("RDKit not available. Install rdkit to run this script.")
        return 1

    json_path = args.case_dir / "real_id_to_smiles.json"
    if not json_path.exists():
        print("Missing {} - run the pipeline first so it writes real_id_to_smiles.json.".format(json_path))
        return 1

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        id_to_smiles = {str(r["real_id"]): r["smiles"] for r in raw if isinstance(r, dict) and "real_id" in r and "smiles" in r}
    else:
        id_to_smiles = {str(k): v for k, v in raw.items()}
    out_dir = args.case_dir / "molecule_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    failed = 0
    real_id_to_image = {}
    for real_id, smiles in id_to_smiles.items():
        if not smiles or not str(smiles).strip():
            failed += 1
            continue
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            failed += 1
            continue
        stem = sanitize_real_id_for_filename(real_id)
        real_id_to_image[str(real_id)] = stem
        out_path = out_dir / "{}_ligand.png".format(stem)
        try:
            img = Draw.MolToImage(mol, size=(args.size, args.size))
            img.save(out_path)
            written += 1
        except Exception:
            failed += 1

    mapping_path = out_dir / "real_id_to_image.json"
    mapping_path.write_text(json.dumps(real_id_to_image, indent=2), encoding="utf-8")
    print("Wrote {} images and {} to {} ({} failed).".format(written, mapping_path.name, out_dir, failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
