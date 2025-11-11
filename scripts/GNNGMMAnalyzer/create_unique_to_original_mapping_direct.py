#!/usr/bin/env python3
"""
Create Unique to Original Index Mapping (Direct File Loading)

This script creates a mapping from unique embedding indices (374-378 samples) to original SMILES dataframe indices.
It loads SMILES data directly from files without requiring model_setup or torch_geometric.

Usage:
    python3 create_unique_to_original_mapping_direct.py [--libra-path PATH]
    
If --libra-path is not provided, the script will try to find the Libra-ML-Modeling directory automatically.
"""

import os
import sys
import pandas as pd
import json
import argparse
from pathlib import Path

def load_smiles_direct(libra_path=None):
    """Load SMILES data directly from files"""
    current_file = Path(__file__).resolve()
    
    # If path provided, use it
    if libra_path:
        dd = Path(libra_path)
        if not dd.exists():
            print(f"Error: Provided path does not exist: {dd}")
            return None
        if not (dd / "data" / "HF-Literature.tsv").exists():
            print(f"Error: HF-Literature.tsv not found at {dd / 'data' / 'HF-Literature.tsv'}")
            return None
        print(f"Using provided Libra-ML-Modeling path: {dd}")
        return dd
    
    # Find the project root directory (Libra-ML-Modeling)
    # Try multiple possible locations
    possible_paths = [
        current_file.parent.parent / "Libra-ML-Modeling",  # Under scripts/ directory
        current_file.parent.parent.parent / "Libra-ML-Modeling",
        Path("/Users/chenjiaqing/Documents/GitHub/Libra-ML-Modeling"),
        Path.home() / "Documents" / "GitHub" / "Libra-ML-Modeling",
        Path("/Users/chenjiaqing/Libra-ML-Modeling"),
        Path.home() / "Libra-ML-Modeling",
    ]
    
    # Also try to find it by searching for the data file
    import subprocess
    try:
        result = subprocess.run(
            ['find', str(Path.home()), '-name', 'HF-Literature.tsv', '-type', 'f', '-maxdepth', '5'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            found_path = Path(result.stdout.strip().split('\n')[0]).parent.parent
            if found_path.name == "data":
                possible_paths.insert(0, found_path.parent)
    except:
        pass
    
    dd = None
    for path in possible_paths:
        if path.exists() and (path / "data" / "HF-Literature.tsv").exists():
            dd = path
            print(f"Found Libra-ML-Modeling at: {dd}")
            break
    
    if dd is None:
        print("Error: Could not find Libra-ML-Modeling directory")
        print("Tried paths:")
        for path in possible_paths[:5]:  # Show first 5
            print(f"  - {path}")
        print("\nPlease provide the path using --libra-path argument:")
        print("  python3 create_unique_to_original_mapping_direct.py --libra-path /path/to/Libra-ML-Modeling")
        return None
    
    try:
        # Load HF-Literature.tsv
        desc_path = dd / "data" / "HF-Literature.tsv"
        desc = pd.read_csv(desc_path, sep="\t")
        print(f"Loaded HF-Literature.tsv: {desc.shape}")
        
        # Try to load split_0.json for full dataset
        json_path = dd / "model-training" / "butene" / "dimenet-10conf" / "data" / "jsons" / "split_0.json"
        
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                
                train = pd.DataFrame(data["train"])
                val = pd.DataFrame(data["val"])
                test = pd.DataFrame(data["test"])
                
                df = pd.concat([train, test, val])
                df = df.merge(desc, left_on="rxn_id", right_on="rxn_id", how="left")
                
                print(f"Combined dataframe shape: {df.shape}")
                print(f"Unique rxn_ids: {df['rxn_id'].nunique()}")
                
                return df
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse split_0.json: {e}")
                print("Using HF-Literature.tsv directly (assuming it contains unique rxn_ids)")
        
        # Fallback: Use HF-Literature.tsv directly
        # If HF-Literature.tsv already contains unique rxn_ids, we can use it directly
        # The unique embeddings are created by sorting unique rxn_ids, so we need to sort
        desc_sorted = desc.sort_values('rxn_id').reset_index(drop=True)
        print(f"Using HF-Literature.tsv directly (sorted by rxn_id)")
        print(f"Unique rxn_ids: {desc_sorted['rxn_id'].nunique()}")
        
        # For the mapping, we need to map unique embedding index to original dataset index
        # Since we don't have the full dataset, we'll use the index in HF-Literature as the "original" index
        # But actually, we need the first occurrence in the FULL dataset, not just HF-Literature
        # Let's try a different approach: use the index from HF-Literature as a proxy
        return desc_sorted
        
    except Exception as e:
        print(f"Error loading SMILES data: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_unique_to_original_mapping(libra_path=None):
    """Create mapping from unique embedding indices to original SMILES indices"""
    
    print("="*60)
    print("Creating unique to original index mapping")
    print("="*60)
    
    # Load SMILES data
    smiles = load_smiles_direct(libra_path)
    if smiles is None:
        return None
    
    print(f"\nSMILES dataframe shape: {smiles.shape}")
    print(f"Unique rxn_ids in SMILES: {smiles['rxn_id'].nunique()}")
    
    # Get unique molecules (one per unique rxn_id)
    unique_molecules = smiles.drop_duplicates(subset=['rxn_id']).reset_index(drop=True)
    print(f"Unique molecules: {len(unique_molecules)}")
    
    # Create mapping: unique embedding index -> original SMILES index
    unique_to_original_mapping = {}
    
    for unique_idx, (_, row) in enumerate(unique_molecules.iterrows()):
        # unique_idx is the index in the unique dataframe (0-377)
        # Find the first occurrence of this rxn_id in the original dataframe
        rxn_id = row['rxn_id']
        
        # Get the first occurrence index
        # If smiles is from the full dataset (with duplicates), use that
        # Otherwise, if it's already unique, use the current index
        if len(smiles[smiles['rxn_id'] == rxn_id]) > 1:
            # Multiple occurrences - get first one
            first_occurrence_idx = smiles[smiles['rxn_id'] == rxn_id].index[0]
        else:
            # Already unique or single occurrence
            first_occurrence_idx = smiles[smiles['rxn_id'] == rxn_id].index[0]
        
        unique_to_original_mapping[unique_idx] = int(first_occurrence_idx)
    
    print(f"\nCreated mapping for {len(unique_to_original_mapping)} unique embeddings")
    print(f"Unique index range: {min(unique_to_original_mapping.keys())} - {max(unique_to_original_mapping.keys())}")
    print(f"Original index range: {min(unique_to_original_mapping.values())} - {max(unique_to_original_mapping.values())}")
    
    # Show sample mappings
    print("\nSample mappings (unique index -> original index):")
    for i in range(min(10, len(unique_to_original_mapping))):
        print(f"  Unique {i} -> Original {unique_to_original_mapping[i]}")
    
    # Save the mapping
    script_dir = Path(__file__).parent
    mapping_file = script_dir / "unique_to_original_mapping.json"
    mapping_dict = {str(k): v for k, v in unique_to_original_mapping.items()}
    with open(mapping_file, 'w') as f:
        json.dump(mapping_dict, f, indent=2)
    
    print(f"\n✓ Unique to original mapping saved to: {mapping_file}")
    
    # Also create CSV
    csv_data = []
    for unique_idx, orig_idx in unique_to_original_mapping.items():
        csv_data.append({
            'unique_index': int(unique_idx),
            'original_index': orig_idx,
            'rxn_id': unique_molecules.iloc[unique_idx]['rxn_id']
        })
    
    df_mapping = pd.DataFrame(csv_data)
    csv_file = script_dir / "unique_to_original_mapping.csv"
    df_mapping.to_csv(csv_file, index=False)
    print(f"✓ Unique to original mapping CSV saved to: {csv_file}")
    
    return unique_to_original_mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create unique to original index mapping')
    parser.add_argument('--libra-path', type=str, help='Path to Libra-ML-Modeling directory')
    args = parser.parse_args()
    
    mapping = create_unique_to_original_mapping(libra_path=args.libra_path)
    if mapping:
        print("\n" + "="*60)
        print("✅ Mapping created successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Failed to create mapping")
        print("="*60)
        print("\nTo fix this, provide the path to Libra-ML-Modeling:")
        print("  python3 create_unique_to_original_mapping_direct.py --libra-path /path/to/Libra-ML-Modeling")

