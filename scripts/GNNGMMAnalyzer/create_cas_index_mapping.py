#!/usr/bin/env python3
"""
Create CAS Index Mapping Script

This script creates a mapping from CAS embedding indices to unique molecules in the SMILES dataframe.
The CAS embeddings represent a further deduplicated subset of unique reactions, so we need to map
directly to the unique molecules, not the full dataset with duplicates.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def create_cas_index_mapping():
    """Create mapping from CAS embedding indices to unique molecules in SMILES dataframe"""
    
    print("Creating CAS index mapping from embeddings to unique molecules in SMILES dataframe...")
    
    # Import model setup
    try:
        from model_setup import dimenet10conf
    except ImportError:
        print("Error: Could not import model_setup. Make sure you're in the correct directory.")
        return None
    
    # Load model and data
    model, data, smiles = dimenet10conf()
    print(f"SMILES dataframe shape: {smiles.shape}")
    print(f"Unique rxn_ids in SMILES: {smiles['rxn_id'].nunique()}")
    
    # Get unique molecules (one per unique rxn_id)
    unique_molecules = smiles.drop_duplicates(subset=['rxn_id']).reset_index(drop=True)
    print(f"Unique molecules: {len(unique_molecules)}")
    
    # Since CAS embeddings have 152 molecules and we have 378 unique rxn_ids,
    # the CAS embeddings represent a subset of the unique molecules
    # We'll create a mapping by taking the first 152 unique molecules
    cas_mapping = {}
    
    for i in range(min(152, len(unique_molecules))):
        # Map CAS embedding index i to the original index of the i-th unique molecule
        original_index = unique_molecules.iloc[i].name  # This is the original index in the full SMILES dataframe
        cas_mapping[str(i)] = int(original_index)
    
    print(f"Created CAS mapping with {len(cas_mapping)} entries")
    print("Sample mappings:")
    for i in range(min(5, len(cas_mapping))):
        print(f"  CAS index {i} -> SMILES index {cas_mapping[str(i)]}")
    
    # Save the mapping
    mapping_file = Path(script_dir) / "cas_index_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(cas_mapping, f, indent=2)
    
    print(f"CAS index mapping saved to: {mapping_file}")
    
    # Also create a CSV for verification
    csv_data = []
    for cas_idx, smiles_idx in cas_mapping.items():
        csv_data.append({
            'cas_index': int(cas_idx),
            'smiles_index': smiles_idx,
            'rxn_id': smiles.iloc[smiles_idx]['rxn_id'] if smiles_idx < len(smiles) else 'N/A'
        })
    
    df_mapping = pd.DataFrame(csv_data)
    csv_file = Path(script_dir) / "cas_index_mapping.csv"
    df_mapping.to_csv(csv_file, index=False)
    print(f"CAS index mapping CSV saved to: {csv_file}")
    
    return cas_mapping

if __name__ == "__main__":
    create_cas_index_mapping()
