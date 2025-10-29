#!/usr/bin/env python3
"""
Test CAS Index Mapping

This script tests the CAS index mapping to ensure it correctly maps to unique molecules.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def test_cas_mapping():
    """Test the CAS index mapping"""
    
    print("Testing CAS index mapping...")
    
    try:
        from model_setup import dimenet10conf
        
        # Load model and data
        model, data, smiles = dimenet10conf()
        print(f"SMILES dataframe shape: {smiles.shape}")
        print(f"Unique rxn_ids in SMILES: {smiles['rxn_id'].nunique()}")
        
        # Get unique molecules (one per unique rxn_id)
        unique_molecules = smiles.drop_duplicates(subset=['rxn_id']).reset_index(drop=True)
        print(f"Unique molecules: {len(unique_molecules)}")
        
        # Create mapping from CAS embedding indices to unique molecule indices
        mapping = {}
        for i in range(min(152, len(unique_molecules))):
            # Map CAS embedding index i to the original index of the i-th unique molecule
            original_index = unique_molecules.iloc[i].name  # Original index in full SMILES dataframe
            mapping[i] = int(original_index)
        
        print(f"Created CAS index mapping: {len(mapping)} entries")
        print("Sample mappings:")
        for i in range(min(10, len(mapping))):
            smiles_idx = mapping[i]
            rxn_id = smiles.iloc[smiles_idx]['rxn_id']
            print(f"  CAS index {i} -> SMILES index {smiles_idx} (rxn_id: {rxn_id})")
        
        # Verify no duplicates in the mapping
        mapped_indices = list(mapping.values())
        unique_mapped_indices = list(set(mapped_indices))
        print(f"\nVerification:")
        print(f"  Total mapped indices: {len(mapped_indices)}")
        print(f"  Unique mapped indices: {len(unique_mapped_indices)}")
        print(f"  No duplicates: {len(mapped_indices) == len(unique_mapped_indices)}")
        
        # Check if all mapped indices correspond to unique rxn_ids
        mapped_rxn_ids = [smiles.iloc[idx]['rxn_id'] for idx in mapped_indices]
        unique_mapped_rxn_ids = list(set(mapped_rxn_ids))
        print(f"  Mapped rxn_ids: {len(mapped_rxn_ids)}")
        print(f"  Unique mapped rxn_ids: {len(unique_mapped_rxn_ids)}")
        print(f"  All unique rxn_ids: {len(mapped_rxn_ids) == len(unique_mapped_rxn_ids)}")
        
        return mapping
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_cas_mapping()
