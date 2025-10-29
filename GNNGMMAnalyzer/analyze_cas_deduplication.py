#!/usr/bin/env python3
"""
CAS Deduplication Analysis Script

This script analyzes the CAS deduplication process to understand the indexing issue
and create the correct mapping between CAS embedding indices and original SMILES indices.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to the path to import model_setup
sys.path.append('/global/cfs/cdirs/m636/nhadler/projects/latent-spaces')

def load_smiles_data():
    """Load SMILES data"""
    try:
        from model_setup import dimenet10conf
        model, data, smiles = dimenet10conf()
        print(f"✓ Successfully loaded SMILES data")
        print(f"  SMILES shape: {smiles.shape}")
        print(f"  Unique rxn_ids: {smiles['rxn_id'].nunique()}")
        print(f"  Unique ligand_1_cas: {smiles['ligand_1_cas'].nunique()}")
        return smiles
    except Exception as e:
        print(f"✗ Error loading SMILES data: {e}")
        return None

def analyze_cas_deduplication(smiles):
    """Analyze the CAS deduplication process"""
    
    print("\n=== CAS DEDUPLICATION ANALYSIS ===")
    
    # This is the deduplication process from the notebook
    u_smiles = smiles.drop_duplicates(subset=["ligand_1_cas"])
    
    print(f"Original SMILES dataframe: {len(smiles)} rows")
    print(f"After CAS deduplication: {len(u_smiles)} rows")
    print(f"Deduplication removed: {len(smiles) - len(u_smiles)} rows")
    
    # Show the first few CAS values
    print(f"\nFirst 10 CAS values in original SMILES:")
    print(smiles['ligand_1_cas'].head(10).tolist())
    
    print(f"\nFirst 10 CAS values after deduplication:")
    print(u_smiles['ligand_1_cas'].head(10).tolist())
    
    # Check if the deduplication preserves order
    print(f"\nChecking if deduplication preserves order:")
    original_first_10_cas = smiles['ligand_1_cas'].head(10).tolist()
    dedup_first_10_cas = u_smiles['ligand_1_cas'].head(10).tolist()
    
    if original_first_10_cas == dedup_first_10_cas:
        print("✓ Deduplication preserves order - first 10 CAS values are identical")
    else:
        print("✗ Deduplication does NOT preserve order")
        print(f"  Original: {original_first_10_cas}")
        print(f"  Deduped:  {dedup_first_10_cas}")
    
    return u_smiles

def create_correct_mapping(smiles, u_smiles):
    """Create the correct mapping from CAS embedding indices to original SMILES indices"""
    
    print("\n=== CREATING CORRECT MAPPING ===")
    
    # The CAS embeddings are created from u_smiles (deduplicated by ligand_1_cas)
    # We need to map CAS embedding indices (0-151) to original SMILES indices
    
    # Create mapping: CAS embedding index -> original SMILES index
    cas_to_original_mapping = {}
    
    for cas_idx, (_, row) in enumerate(u_smiles.iterrows()):
        # cas_idx is the index in the deduplicated dataframe (0-151)
        # row.name is the original index in the full SMILES dataframe
        cas_to_original_mapping[cas_idx] = int(row.name)
    
    print(f"Created mapping for {len(cas_to_original_mapping)} CAS indices")
    print(f"CAS index range: {min(cas_to_original_mapping.keys())} - {max(cas_to_original_mapping.keys())}")
    print(f"Original index range: {min(cas_to_original_mapping.values())} - {max(cas_to_original_mapping.values())}")
    
    # Show sample mappings
    print(f"\nSample mappings (CAS index -> Original SMILES index):")
    for cas_idx in sorted(list(cas_to_original_mapping.keys())[:10]):
        orig_idx = cas_to_original_mapping[cas_idx]
        cas_value = u_smiles.iloc[cas_idx]['ligand_1_cas']
        print(f"  CAS {cas_idx:3d} -> Original {orig_idx:4d} (CAS: {cas_value})")
    
    return cas_to_original_mapping

def verify_mapping(cas_to_original_mapping, smiles, u_smiles):
    """Verify that the mapping is correct"""
    
    print("\n=== VERIFYING MAPPING ===")
    
    # Check that all CAS indices map to valid original indices
    valid_mappings = 0
    invalid_mappings = 0
    
    for cas_idx, orig_idx in cas_to_original_mapping.items():
        if orig_idx in smiles.index:
            valid_mappings += 1
        else:
            invalid_mappings += 1
            print(f"✗ CAS {cas_idx} maps to invalid original index {orig_idx}")
    
    print(f"✓ Valid mappings: {valid_mappings}")
    print(f"✗ Invalid mappings: {invalid_mappings}")
    
    # Check that CAS values match
    cas_mismatches = 0
    for cas_idx, orig_idx in cas_to_original_mapping.items():
        cas_value_from_dedup = u_smiles.iloc[cas_idx]['ligand_1_cas']
        cas_value_from_original = smiles.loc[orig_idx]['ligand_1_cas']
        
        if cas_value_from_dedup != cas_value_from_original:
            cas_mismatches += 1
            print(f"✗ CAS mismatch: CAS {cas_idx} -> Original {orig_idx}")
            print(f"  Dedup CAS: {cas_value_from_dedup}")
            print(f"  Original CAS: {cas_value_from_original}")
    
    print(f"✓ CAS value matches: {len(cas_to_original_mapping) - cas_mismatches}")
    print(f"✗ CAS value mismatches: {cas_mismatches}")
    
    return cas_mismatches == 0

def save_mapping(cas_to_original_mapping, output_path):
    """Save the mapping to a file"""
    
    print(f"\n=== SAVING MAPPING ===")
    
    # Save as JSON
    import json
    json_path = output_path / "cas_to_original_mapping.json"
    with open(json_path, 'w') as f:
        json.dump(cas_to_original_mapping, f, indent=2)
    print(f"✓ Saved JSON mapping to: {json_path}")
    
    # Save as CSV
    csv_path = output_path / "cas_to_original_mapping.csv"
    mapping_df = pd.DataFrame([
        {'cas_index': cas_idx, 'original_index': orig_idx}
        for cas_idx, orig_idx in cas_to_original_mapping.items()
    ])
    mapping_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV mapping to: {csv_path}")
    
    return json_path, csv_path

def main():
    """Main function"""
    print("CAS Deduplication Analysis Script")
    print("="*50)
    
    # Load SMILES data
    smiles = load_smiles_data()
    if smiles is None:
        return 1
    
    # Analyze deduplication
    u_smiles = analyze_cas_deduplication(smiles)
    
    # Create correct mapping
    cas_to_original_mapping = create_correct_mapping(smiles, u_smiles)
    
    # Verify mapping
    is_correct = verify_mapping(cas_to_original_mapping, smiles, u_smiles)
    
    if not is_correct:
        print("\n✗ Mapping verification failed!")
        return 1
    
    # Save mapping
    output_path = Path("/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzer")
    json_path, csv_path = save_mapping(cas_to_original_mapping, output_path)
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Mapping is correct and saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
