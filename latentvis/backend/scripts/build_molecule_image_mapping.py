#!/usr/bin/env python3
"""
Build real_id -> molecule image mapping for frontend.

Uses a CAS-to-original mapping to connect real_id indices (0..151) to
original dataset IDs, then maps to existing image files in public/molecule_images.
"""

import argparse
from pathlib import Path
import sys
import shutil

from latentvis.backend.utils.molecule_images import (
    build_real_id_image_mapping,
    load_index_mapping,
    write_json,
    write_summary_csv,
)


def _create_alias_images(mapping, image_dir, alias_dir):
    """Create real_id-based alias image files (0-151) for existing images."""
    image_dir = Path(image_dir)
    alias_dir = Path(alias_dir)
    alias_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    for real_id, entry in mapping.items():
        for image_type in ["primary", "product", "ligand", "solvent", "startingmat", "catalyst"]:
            filename = entry.get(image_type)
            if not filename:
                continue
            src = image_dir / filename
            if not src.exists():
                skipped += 1
                continue
            alias_name = "{}_{}.png".format(real_id, image_type)
            dst = alias_dir / alias_name
            if dst.exists():
                continue
            shutil.copy2(src, dst)
            created += 1

    print("Created {} alias images, skipped {} missing sources".format(created, skipped))


def _pick_default_mapping_path(project_root):
    candidates = [
        project_root / "scripts" / "GNNGMMAnalyzer" / "cas_to_original_mapping.json",
        project_root.parent / "latent-spaces" / "GNNGMMAnalyzer" / "cas_to_original_mapping.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def main():
    project_root = Path(__file__).resolve().parents[3]
    default_mapping_path = _pick_default_mapping_path(project_root)
    default_image_dir = project_root / "latentvis" / "frontend" / "public" / "molecule_images"
    default_output_json = default_image_dir / "real_id_to_image_mapping.json"
    default_output_csv = default_image_dir / "real_id_to_image_summary.csv"
    default_output_original = default_image_dir / "real_id_to_original_mapping.json"

    parser = argparse.ArgumentParser(description="Build real_id -> molecule image mapping")
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=default_mapping_path,
        help="Path to cas_to_original_mapping.json",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir,
        help="Directory containing molecule images",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_output_json,
        help="Output JSON mapping path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_output_csv,
        help="Output CSV summary path",
    )
    parser.add_argument(
        "--output-original-map",
        type=Path,
        default=default_output_original,
        help="Output real_id -> original_id JSON path",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=152,
        help="Expected number of real IDs (default: 152)",
    )
    parser.add_argument(
        "--materialize-aliases",
        action="store_true",
        help="Create real_id-based alias image files in alias dir",
    )
    parser.add_argument(
        "--alias-dir",
        type=Path,
        default=default_image_dir,
        help="Directory to write alias image files (default: image-dir)",
    )
    args = parser.parse_args()

    if not args.mapping_path.exists():
        print("Error: mapping file not found: {}".format(args.mapping_path))
        return 1

    real_to_original = load_index_mapping(args.mapping_path)
    if args.expected_count and len(real_to_original) != args.expected_count:
        print(
            "Warning: mapping contains {} entries, expected {}".format(
                len(real_to_original), args.expected_count
            )
        )

    mapping = build_real_id_image_mapping(real_to_original, args.image_dir)

    write_json(args.output_json, mapping)
    write_summary_csv(args.output_csv, mapping)
    write_json(args.output_original_map, {str(k): v for k, v in real_to_original.items()})

    if args.materialize_aliases:
        _create_alias_images(mapping, args.image_dir, args.alias_dir)

    print("Wrote image mapping JSON: {}".format(args.output_json))
    print("Wrote image mapping CSV: {}".format(args.output_csv))
    print("Wrote real_id -> original_id JSON: {}".format(args.output_original_map))
    return 0


if __name__ == "__main__":
    sys.exit(main())
