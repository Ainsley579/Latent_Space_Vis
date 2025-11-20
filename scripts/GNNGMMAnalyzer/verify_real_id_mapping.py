#!/usr/bin/env python3
"""
Verify Real ID Mapping

This script verifies that the generated CSV files use original dataset indices (0-3477)
for both Version 1 (378 unique embeddings) and Version 2 (152 CAS embeddings).
"""

import pandas as pd
from pathlib import Path

def verify_real_id_mapping():
    """Verify real ID mapping in generated files"""
    
    data_dir = Path("/Users/chenjiaqing/Documents/GitHub/Latent_Space_Vis/frontend/public/data")
    
    print("="*60)
    print("VERIFYING REAL ID MAPPING")
    print("="*60)
    
    # Version 1 files
    print("\n📊 VERSION 1 (Unique Embeddings - 374 samples):")
    print("-" * 60)
    
    files_v1 = [
        "cluster_transition_matrices_1_individual_real_id.csv",
        "sample_component_paths_1_with_real_id.csv"
    ]
    
    for filename in files_v1:
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            if 'real_id' in df.columns:
                real_ids = df['real_id']
                print(f"\n{filename}:")
                print(f"  Total rows: {len(df)}")
                print(f"  Unique real IDs: {real_ids.nunique()}")
                print(f"  Real ID range: {real_ids.min()} - {real_ids.max()}")
                
                # Check if using original dataset indices (should be 0-3477 range)
                if real_ids.max() > 1000:
                    print(f"  ✓ Using original dataset indices (range includes values > 1000)")
                else:
                    print(f"  ⚠ Using embedding indices (range 0-{real_ids.max()}, should be 0-3477)")
                    print(f"  ⚠ Need to create unique_to_original_mapping.json")
            elif 'sample_id' in df.columns:
                sample_ids = df['sample_id']
                print(f"\n{filename}:")
                print(f"  Total rows: {len(df)}")
                print(f"  Unique sample IDs: {sample_ids.nunique()}")
                print(f"  Sample ID range: {sample_ids.min()} - {sample_ids.max()}")
                
                if sample_ids.max() > 1000:
                    print(f"  ✓ Using original dataset indices")
                else:
                    print(f"  ⚠ Using embedding indices (should be 0-3477)")
        else:
            print(f"\n{filename}: NOT FOUND")
    
    # Version 2 files
    print("\n\n📊 VERSION 2 (CAS Embeddings - 152 samples):")
    print("-" * 60)
    
    files_v2 = [
        "cluster_transition_matrices_2_individual_real_id.csv",
        "sample_component_paths_2_with_real_id.csv"
    ]
    
    for filename in files_v2:
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            if 'real_id' in df.columns:
                real_ids = df['real_id']
                print(f"\n{filename}:")
                print(f"  Total rows: {len(df)}")
                print(f"  Unique real IDs: {real_ids.nunique()}")
                print(f"  Real ID range: {real_ids.min()} - {real_ids.max()}")
                
                if real_ids.max() > 1000:
                    print(f"  ✓ Using original dataset indices (0-3477 range)")
                else:
                    print(f"  ⚠ Using embedding indices")
            elif 'sample_id' in df.columns:
                sample_ids = df['sample_id']
                print(f"\n{filename}:")
                print(f"  Total rows: {len(df)}")
                print(f"  Unique sample IDs: {sample_ids.nunique()}")
                print(f"  Sample ID range: {sample_ids.min()} - {sample_ids.max()}")
                
                if sample_ids.max() > 1000:
                    print(f"  ✓ Using original dataset indices")
                else:
                    print(f"  ⚠ Using embedding indices")
        else:
            print(f"\n{filename}: NOT FOUND")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nExpected:")
    print("  Version 1: 374 unique embeddings -> original dataset indices (0-3477)")
    print("  Version 2: 152 CAS embeddings -> original dataset indices (0-3477)")
    print("\nTo fix Version 1, create unique_to_original_mapping.json by running:")
    print("  python3 create_unique_to_original_mapping.py")
    print("  (Requires model_setup.py with torch_geometric installed)")

if __name__ == "__main__":
    verify_real_id_mapping()

