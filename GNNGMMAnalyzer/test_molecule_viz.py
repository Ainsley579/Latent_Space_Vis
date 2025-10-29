#!/usr/bin/env python3
"""
Simple Molecule Visualization Test Script

Tests the molecule visualization functionality without requiring model_setup
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import PandasTools
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import math

def setup_rdkit():
    """Setup RDKit configuration for molecule drawing"""
    rdDepictor.SetPreferCoordGen(True)
    
    # Configure drawing options
    opts = Draw.MolDrawOptions()
    opts.legendFraction = 0.25
    opts.legendFontSize = 25
    
    return opts

def test_molecule_visualization():
    """Test molecule visualization with sample data"""
    
    print("=== MOLECULE VISUALIZATION TEST ===")
    
    # Setup RDKit
    setup_rdkit()
    
    # Load the CSV file
    base_dir = Path("/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzerResults/dimenet10conf_checkpoint0_fixed_comprehensive_analysis_with_correct_cas_indices")
    csv_path = base_dir / "adaptive_with_zca" / "gmm_centers_and_neighbors.csv"
    
    if not csv_path.exists():
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows from CSV")
    
    # Check if we have the required columns
    required_cols = ['layer', 'component', 'neighbor_rank', 'original_molecule_index', 'ligand_1_smiles']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"✗ Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    print(f"✓ All required columns present")
    
    # Filter for a specific layer and component
    target_layer = "emb"  # Use the first available layer
    layer_df = df[df["layer"] == target_layer].copy()
    
    if len(layer_df) == 0:
        print(f"✗ No data found for layer: {target_layer}")
        print(f"Available layers: {df['layer'].unique()}")
        return False
    
    print(f"✓ Found {len(layer_df)} molecules for layer {target_layer}")
    
    # Take first component
    first_component = layer_df['component'].iloc[0]
    comp_df = layer_df[layer_df['component'] == first_component].copy()
    
    print(f"✓ Testing with component {first_component}, {len(comp_df)} molecules")
    
    # Check if we have SMILES data
    if 'ligand_1_smiles' not in comp_df.columns:
        print("✗ No SMILES data found in CSV")
        print("This means we need to merge with the SMILES dataframe")
        return False
    
    # Add molecule column
    try:
        PandasTools.AddMoleculeColumnToFrame(
            comp_df,
            "ligand_1_smiles",
            "mol",
        )
        print("✓ Successfully added molecule column")
    except Exception as e:
        print(f"✗ Error adding molecule column: {e}")
        return False
    
    # Filter out invalid molecules
    valid_mols = comp_df[comp_df['mol'].notna()]
    print(f"✓ Found {len(valid_mols)} valid molecules")
    
    if len(valid_mols) == 0:
        print("✗ No valid molecules found")
        return False
    
    # Create a simple molecule grid
    try:
        mols = list(valid_mols['mol'])[:4]  # Take first 4 molecules
        legends = []
        
        for _, row in valid_mols.head(4).iterrows():
            legend = f"ID: {int(row['original_molecule_index'])}\nRank: {int(row['neighbor_rank'])}"
            legends.append(legend)
        
        # Draw molecules
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 600, 300, 300)
        drawer.DrawMolecules(mols, legends=legends)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        
        # Save test image
        output_dir = Path("/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzer/molecule_visualizations")
        output_dir.mkdir(exist_ok=True)
        
        test_image_path = output_dir / "test_molecule_visualization.png"
        with open(test_image_path, 'wb') as f:
            f.write(png)
        
        print(f"✓ Successfully created test molecule visualization")
        print(f"  Saved to: {test_image_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating molecule visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Molecule Visualization Test Script")
    print("="*50)
    
    success = test_molecule_visualization()
    
    if success:
        print("\n✓ Test completed successfully!")
        print("The molecule visualization functionality is working.")
    else:
        print("\n✗ Test failed!")
        print("There are issues that need to be resolved.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
