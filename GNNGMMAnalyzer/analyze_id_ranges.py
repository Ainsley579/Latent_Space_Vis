#!/usr/bin/env python3
"""
ID Range Analysis Script

This script analyzes the ID ranges in GMM analysis results to identify
indexing issues between CAS embeddings and original SMILES data.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Any

# Add the project root to the path to import model_setup
sys.path.append('/global/cfs/cdirs/m636/nhadler/projects')

def load_model_and_data():
    """Load the model and SMILES data"""
    try:
        from model_setup import dimenet10conf
        model, data, smiles = dimenet10conf()
        print(f"✓ Successfully loaded model and data")
        print(f"  SMILES dataframe shape: {smiles.shape}")
        print(f"  Unique rxn_ids: {smiles['rxn_id'].nunique()}")
        return model, data, smiles
    except Exception as e:
        print(f"✗ Error loading model and data: {e}")
        return None, None, None

def load_gmm_results(base_dir: Path, analysis_type: str) -> pd.DataFrame:
    """Load GMM results from CSV file"""
    try:
        csv_path = base_dir / analysis_type / "gmm_centers_and_neighbors.csv"
        if not csv_path.exists():
            print(f"✗ CSV file not found: {csv_path}")
            return None
            
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded GMM results from {csv_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"✗ Error loading GMM results: {e}")
        return None

def analyze_id_ranges(df: pd.DataFrame, smiles: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the ID ranges in the data"""
    analysis = {}
    
    # Original molecule index range
    orig_indices = df['original_molecule_index'].values
    analysis['original_index_range'] = (orig_indices.min(), orig_indices.max())
    analysis['original_index_count'] = len(np.unique(orig_indices))
    
    # Neighbor index range (embedding indices)
    neighbor_indices = df['neighbor_index'].values
    analysis['neighbor_index_range'] = (neighbor_indices.min(), neighbor_indices.max())
    analysis['neighbor_index_count'] = len(np.unique(neighbor_indices))
    
    # Check if original indices exist in SMILES dataframe
    smiles_indices = smiles.index.values
    analysis['smiles_index_range'] = (smiles_indices.min(), smiles_indices.max())
    analysis['smiles_index_count'] = len(smiles_indices)
    
    # Check overlap between original indices and SMILES indices
    orig_in_smiles = np.isin(orig_indices, smiles_indices)
    analysis['orig_indices_in_smiles'] = orig_in_smiles.sum()
    analysis['orig_indices_not_in_smiles'] = (~orig_in_smiles).sum()
    
    # Layer analysis
    layers = df['layer'].unique()
    analysis['layers'] = list(layers)
    
    # Component analysis
    analysis['total_components'] = df['component'].nunique()
    
    return analysis

def print_analysis_report(analysis: Dict[str, Any]):
    """Print a detailed analysis report"""
    print("\n" + "="*60)
    print("ID RANGE ANALYSIS REPORT")
    print("="*60)
    
    print(f"SMILES DataFrame Index Range: {analysis['smiles_index_range']}")
    print(f"SMILES DataFrame Index Count: {analysis['smiles_index_count']}")
    
    print(f"\nOriginal Molecule Index Range: {analysis['original_index_range']}")
    print(f"Original Molecule Index Count: {analysis['original_index_count']}")
    print(f"Original indices found in SMILES: {analysis['orig_indices_in_smiles']}")
    print(f"Original indices NOT found in SMILES: {analysis['orig_indices_not_in_smiles']}")
    
    print(f"\nNeighbor Index Range (Embedding): {analysis['neighbor_index_range']}")
    print(f"Neighbor Index Count: {analysis['neighbor_index_count']}")
    
    print(f"\nAvailable Layers: {analysis['layers']}")
    print(f"Total Components: {analysis['total_components']}")
    
    # Check for potential issues
    print("\n" + "-"*40)
    print("POTENTIAL ISSUES:")
    print("-"*40)
    
    if analysis['original_index_range'][0] == 0 and analysis['original_index_range'][1] == 151:
        print("⚠️  Original indices are 0-151, suggesting CAS embedding indices")
        print("   This may not match the actual SMILES dataframe indices")
    
    if analysis['neighbor_index_range'][0] == 0 and analysis['neighbor_index_range'][1] == 151:
        print("⚠️  Neighbor indices are 0-151, confirming CAS embedding indices")
    
    if analysis['orig_indices_not_in_smiles'] > 0:
        print(f"⚠️  {analysis['orig_indices_not_in_smiles']} original indices not found in SMILES dataframe")
        print("   This indicates an indexing mismatch")
    
    if analysis['original_index_count'] != analysis['neighbor_index_count']:
        print("⚠️  Mismatch between original and neighbor index counts")
    
    print("="*60)

def suggest_fix(analysis: Dict[str, Any], smiles: pd.DataFrame):
    """Suggest how to fix the indexing issue"""
    print("\n" + "="*60)
    print("SUGGESTED FIX")
    print("="*60)
    
    print("The issue is that the CAS embedding indices (0-151) are being used")
    print("as original_molecule_index, but these don't correspond to actual")
    print("indices in the SMILES dataframe.")
    print()
    print("SOLUTION:")
    print("1. The CAS embeddings contain 152 unique molecules")
    print("2. We need to map these to the first 152 unique molecules in SMILES")
    print("3. Use the actual SMILES dataframe indices, not sequential 0-151")
    print()
    
    # Show sample of SMILES indices
    unique_molecules = smiles.drop_duplicates(subset=['rxn_id']).reset_index(drop=True)
    print(f"First 10 unique molecule indices in SMILES:")
    for i in range(min(10, len(unique_molecules))):
        original_idx = unique_molecules.iloc[i].name
        rxn_id = unique_molecules.iloc[i]['rxn_id']
        print(f"  CAS index {i} -> SMILES index {original_idx} (rxn_id: {rxn_id})")
    
    print("="*60)

def main():
    """Main function to analyze ID ranges"""
    parser = argparse.ArgumentParser(description="Analyze ID ranges in GMM analysis results")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing GMM analysis results")
    parser.add_argument("--analysis_type", type=str, default="adaptive_with_zca",
                       choices=["fixed_no_zca", "fixed_with_zca", "adaptive_no_zca", "adaptive_with_zca"],
                       help="Type of analysis to analyze")
    
    args = parser.parse_args()
    
    print("ID Range Analysis Script")
    print("="*50)
    
    # Load data
    model, data, smiles = load_model_and_data()
    if smiles is None:
        return 1
    
    # Load GMM results
    df = load_gmm_results(Path(args.base_dir), args.analysis_type)
    if df is None:
        return 1
    
    # Analyze ID ranges
    analysis = analyze_id_ranges(df, smiles)
    print_analysis_report(analysis)
    suggest_fix(analysis, smiles)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())