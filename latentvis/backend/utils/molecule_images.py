#!/usr/bin/env python3
"""
Utilities for mapping real IDs to molecule image files.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

DEFAULT_IMAGE_TYPES = [
    "primary",
    "product",
    "ligand",
    "solvent",
    "startingmat",
    "catalyst",
]


def load_index_mapping(mapping_path: Path) -> Dict[int, int]:
    """Load a JSON mapping file with integer keys and values."""
    with open(mapping_path, "r") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def build_real_id_image_mapping(
    real_to_original: Dict[int, int],
    image_dir: Path,
    image_types: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Build a real_id -> image mapping based on existing images for original IDs.

    Returns a dict keyed by real_id string with image filenames and original_id.
    """
    image_dir = Path(image_dir)
    image_types = list(image_types) if image_types else list(DEFAULT_IMAGE_TYPES)

    mapping: Dict[str, Dict[str, int]] = {}

    for real_id, original_id in sorted(real_to_original.items()):
        entry: Dict[str, int] = {"original_id": int(original_id)}
        for image_type in image_types:
            filename = f"{original_id}_{image_type}.png"
            if (image_dir / filename).exists():
                entry[image_type] = filename
        mapping[str(real_id)] = entry

    return mapping


def write_json(path: Path, payload: Dict[str, Dict[str, int]]) -> None:
    """Write mapping JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_summary_csv(path: Path, mapping: Dict[str, Dict[str, int]]) -> None:
    """Write a CSV summary of the mapping."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "real_id",
        "original_id",
        "primary_image",
        "product_image",
        "ligand_image",
        "solvent_image",
        "startingmat_image",
        "catalyst_image",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for real_id, entry in mapping.items():
            writer.writerow(
                {
                    "real_id": real_id,
                    "original_id": entry.get("original_id", ""),
                    "primary_image": entry.get("primary", ""),
                    "product_image": entry.get("product", ""),
                    "ligand_image": entry.get("ligand", ""),
                    "solvent_image": entry.get("solvent", ""),
                    "startingmat_image": entry.get("startingmat", ""),
                    "catalyst_image": entry.get("catalyst", ""),
                }
            )
