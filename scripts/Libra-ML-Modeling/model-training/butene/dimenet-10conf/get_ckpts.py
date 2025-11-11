#!/usr/bin/env python3
"""Script to collect all last.ckpt files and organize them in a checkpoints folder."""

import shutil
from pathlib import Path


def collect_checkpoints():
    """Find and copy all last.ckpt files to organized checkpoints folder."""
    # Define paths
    models_dir = Path("models")
    checkpoints_dir = Path("checkpoints")

    # Create checkpoints directory if it doesn't exist
    checkpoints_dir.mkdir(exist_ok=True)

    # Find all last.ckpt files
    last_ckpt_files = list(models_dir.rglob("last.ckpt"))

    print(f"Found {len(last_ckpt_files)} last.ckpt files")

    for ckpt_file in last_ckpt_files:
        try:
            # Extract split and seed from path
            # Path structure: models/split_XXX/seed_YYY/lightning_logs/version_Z/checkpoints/last.ckpt
            parts = ckpt_file.parts

            # Find split and seed folders
            split_folder = None
            seed_folder = None

            for part in parts:
                if part.startswith("split_"):
                    split_folder = part
                elif part.startswith("seed_"):
                    seed_folder = part

            if split_folder and seed_folder:
                # Create new filename: split_XXX_seed_YYY.ckpt
                new_filename = f"{split_folder}_{seed_folder}.ckpt"
                destination = checkpoints_dir / new_filename

                # Copy the file
                shutil.copy2(ckpt_file, destination)
                print(f"Copied: {ckpt_file} -> {destination}")
            else:
                print(f"Could not parse path: {ckpt_file}")

        except Exception as e:
            print(f"Error processing {ckpt_file}: {e}")

    print(f"\nCheckpoint collection complete. Files saved to: {checkpoints_dir.absolute()}")


if __name__ == "__main__":
    collect_checkpoints()
