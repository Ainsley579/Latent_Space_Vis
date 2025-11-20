#!/usr/bin/env python3
"""
Create Unique to Original Index Mapping

This script creates a mapping from unique embedding indices (374-378 samples) to original SMILES dataframe indices.
This is needed for Version 1 which uses unique embeddings.
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path

# Add the parent directory to path to import model_setup
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

def create_unique_to_original_mapping():
    """Create mapping from unique embedding indices to original SMILES indices"""
    
    print("Creating unique to original index mapping...")
    
    try:
        from model_setup import dimenet10conf
    except ImportError as e:
        print(f"Error: Could not import model_setup: {e}")
        print("Trying alternative approach...")
        return None
    
    # Load model and data
    model, data, smiles = dimenet10conf()
    print(f"SMILES dataframe shape: {smiles.shape}")
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
        first_occurrence_idx = smiles[smiles['rxn_id'] == rxn_id].index[0]
        unique_to_original_mapping[unique_idx] = int(first_occurrence_idx)
    
    print(f"Created mapping for {len(unique_to_original_mapping)} unique embeddings")
    print(f"Unique index range: {min(unique_to_original_mapping.keys())} - {max(unique_to_original_mapping.keys())}")
    print(f"Original index range: {min(unique_to_original_mapping.values())} - {max(unique_to_original_mapping.values())}")
    
    # Show sample mappings
    print("\nSample mappings (unique index -> original index):")
    for i in range(min(10, len(unique_to_original_mapping))):
        print(f"  Unique {i} -> Original {unique_to_original_mapping[i]}")
    
    # Save the mapping
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
    mapping = create_unique_to_original_mapping()
    if mapping:
        print("\n✅ Mapping created successfully!")
    else:
        print("\n❌ Failed to create mapping")
