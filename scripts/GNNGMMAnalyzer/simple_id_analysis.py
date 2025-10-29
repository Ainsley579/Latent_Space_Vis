#!/usr/bin/env python3
"""
Simplified ID Range Analysis Script (No RDKit dependency)

This script analyzes the ID ranges in GMM analysis results to identify
indexing issues between CAS embeddings and original SMILES data.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def analyze_id_ranges_simple(base_dir: Path, analysis_type: str):
    """Analyze ID ranges without requiring model_setup"""
    
    print("=== SIMPLIFIED ID RANGE ANALYSIS ===")
    
    # Load the CSV file
    csv_path = base_dir / analysis_type / "gmm_centers_and_neighbors.csv"
    if not csv_path.exists():
        print(f"✗ CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows from CSV")
    
    # Analyze neighbor_index (CAS embedding indices)
    neighbor_indices = df['neighbor_index'].dropna()
    print(f"\nNEIGHBOR INDEX (CAS embedding indices):")
    print(f"  Range: {neighbor_indices.min()} - {neighbor_indices.max()}")
    print(f"  Unique values: {neighbor_indices.nunique()}")
    print(f"  Sample values: {sorted(neighbor_indices.unique())[:10]}")
    
    # Analyze original_molecule_index (original SMILES indices)
    orig_indices = df['original_molecule_index'].dropna()
    print(f"\nORIGINAL MOLECULE INDEX (original SMILES indices):")
    print(f"  Range: {orig_indices.min()} - {orig_indices.max()}")
    print(f"  Unique values: {orig_indices.nunique()}")
    print(f"  Sample values: {sorted(orig_indices.unique())[:10]}")
    
    # Layer analysis
    layers = df['layer'].unique()
    print(f"\nLAYERS:")
    print(f"  Available layers: {sorted(layers)}")
    
    # Component analysis
    components = df['component'].unique()
    print(f"\nCOMPONENTS:")
    print(f"  Total components: {len(components)}")
    print(f"  Component range: {components.min()} - {components.max()}")
    
    # Create a mapping comparison
    print(f"\nMAPPING COMPARISON:")
    sample_df = df[['neighbor_index', 'original_molecule_index']].drop_duplicates().head(20)
    print("Sample mappings (neighbor_index -> original_molecule_index):")
    for _, row in sample_df.iterrows():
        print(f"  {int(row['neighbor_index'])} -> {int(row['original_molecule_index'])}")
    
    # Check for the issue
    print(f"\n" + "="*50)
    print("ISSUE ANALYSIS:")
    print("="*50)
    
    if orig_indices.min() == 0 and orig_indices.max() == 151:
        print("⚠️  PROBLEM IDENTIFIED:")
        print("   The original_molecule_index range is 0-151")
        print("   This suggests CAS embedding indices are being used")
        print("   instead of actual SMILES dataframe indices")
        print()
        print("   SOLUTION:")
        print("   The CAS embeddings contain 152 unique molecules")
        print("   These should be mapped to the first 152 unique molecules")
        print("   in the SMILES dataframe using their actual indices")
    else:
        print("✓ Original molecule indices appear to be correct")
        print(f"   Range: {orig_indices.min()} - {orig_indices.max()}")
    
    # Save analysis to file
    output_path = base_dir / "id_range_analysis.txt"
    with open(output_path, 'w') as f:
        f.write("=== ID RANGE ANALYSIS ===\n")
        f.write(f"Total rows: {len(df)}\n\n")
        f.write(f"NEIGHBOR INDEX (CAS embedding indices):\n")
        f.write(f"  Range: {neighbor_indices.min()} - {neighbor_indices.max()}\n")
        f.write(f"  Unique values: {neighbor_indices.nunique()}\n\n")
        f.write(f"ORIGINAL MOLECULE INDEX (original SMILES indices):\n")
        f.write(f"  Range: {orig_indices.min()} - {orig_indices.max()}\n")
        f.write(f"  Unique values: {orig_indices.nunique()}\n\n")
        
        f.write("Sample mappings (neighbor_index -> original_molecule_index):\n")
        for _, row in sample_df.iterrows():
            f.write(f"  {int(row['neighbor_index'])} -> {int(row['original_molecule_index'])}\n")
    
    print(f"\n✓ Analysis saved to: {output_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze ID ranges in GMM analysis results")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing GMM analysis results")
    parser.add_argument("--analysis_type", type=str, default="adaptive_with_zca",
                       choices=["fixed_no_zca", "fixed_with_zca", "adaptive_no_zca", "adaptive_with_zca"],
                       help="Type of analysis to analyze")
    
    args = parser.parse_args()
    
    print("Simplified ID Range Analysis Script")
    print("="*50)
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"✗ Base directory not found: {base_dir}")
        return 1
    
    analyze_id_ranges_simple(base_dir, args.analysis_type)
    return 0

if __name__ == "__main__":
    sys.exit(main())
