#!/usr/bin/env python
from __future__ import annotations

import json
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR = ROOT / "frontend" / "public" / "molecule-images"
DATASETS = {
    "shared": ROOT / "backend" / "data" / "real_id_to_smiles.json",
    "dimenet": ROOT / "backend" / "data" / "dimenet" / "real_id_to_smiles.json",
}
IMAGE_SIZE = (320, 240)
SLUG_MAX_LENGTH = 72


def stable_hash_base36(value: str) -> str:
    hash_value = 2166136261
    for char in value:
        hash_value ^= ord(char)
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    if hash_value == 0:
        return "0"
    digits: list[str] = []
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    while hash_value:
        hash_value, remainder = divmod(hash_value, 36)
        digits.append(alphabet[remainder])
    return "".join(reversed(digits))


def slugify_real_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug or "molecule")[:SLUG_MAX_LENGTH]


def target_path(dataset_key: str, real_id: str) -> Path:
    slug = slugify_real_id(real_id)
    digest = stable_hash_base36(real_id)
    return PUBLIC_DIR / dataset_key / f"{slug}-{digest}.png"


def render_png(smiles: str) -> bytes | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdDepictor.SetPreferCoordGen(True)
    drawer = rdMolDraw2D.MolDraw2DCairo(*IMAGE_SIZE)
    options = drawer.drawOptions()
    options.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def clear_existing_pngs(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        return
    for path in dataset_dir.glob("*.png"):
        path.unlink()


def main() -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, object]] = {}

    for dataset_key, source_path in DATASETS.items():
        dataset_dir = PUBLIC_DIR / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        clear_existing_pngs(dataset_dir)

        with source_path.open() as handle:
            mapping = json.load(handle)

        generated = 0
        invalid: list[str] = []
        for real_id, smiles in mapping.items():
            text = str(smiles or "").strip()
            if not text:
                invalid.append(str(real_id))
                continue
            png = render_png(text)
            if png is None:
                invalid.append(str(real_id))
                continue
            output_path = target_path(dataset_key, str(real_id))
            output_path.write_bytes(png)
            generated += 1

        summary[dataset_key] = {
            "source": str(source_path.relative_to(ROOT)),
            "generated": generated,
            "invalid_count": len(invalid),
            "invalid_real_ids": invalid,
        }
        print(
            f"{dataset_key}: generated {generated} images from "
            f"{source_path.relative_to(ROOT)}; invalid={len(invalid)}"
        )

    summary_path = PUBLIC_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
