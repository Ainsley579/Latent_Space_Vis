#!/usr/bin/env python3
"""
Render MCS substructures as PNG images for frontend display.

Input: cluster_substructures.json
Output: images under latentvis/frontend/public/data/substructures/images/
"""

import argparse
import json
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate substructure PNG images")
    parser.add_argument(
        "--substructures-json",
        type=Path,
        default=Path("latentvis/frontend/public/data/substructures/cluster_substructures.json"),
        help="Path to cluster_substructures.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("latentvis/frontend/public/data/substructures/images"),
        help="Output directory for PNGs",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=240,
        help="Square image size (pixels)",
    )
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is not available. Install RDKit to run this script.")
    if not args.substructures_json.exists():
        raise FileNotFoundError("Substructures JSON not found: {}".format(args.substructures_json))

    data = json.loads(args.substructures_json.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for entry in data:
        model = str(entry.get("model_type", "unknown"))
        layer = str(entry.get("layer_name", "unknown"))
        cluster_id = int(entry.get("cluster_id", -1))
        group = str(entry.get("group", "group"))
        smarts = entry.get("mcs_smarts")
        smiles = entry.get("mcs_smiles")

        mol = None
        if smarts:
            mol = Chem.MolFromSmarts(smarts)
        if mol is None and smiles:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        fname = "{}_{}_cluster{}_{}.png".format(
            _safe_token(model),
            _safe_token(layer),
            cluster_id,
            _safe_token(group),
        )
        out_path = args.out_dir / fname
        img = Draw.MolToImage(mol, size=(args.size, args.size))
        img.save(out_path)
        written += 1

    print("Wrote {} images to {}".format(written, args.out_dir))


if __name__ == "__main__":
    main()
