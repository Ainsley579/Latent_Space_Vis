#!/usr/bin/env python3
"""
Complete Molecule Visualization Script

This script loads GMM results, merges with SMILES data, and creates molecule visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

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

def load_smiles_data():
    """Load SMILES data by finding the model_setup module"""
    
    # Try different possible locations for model_setup
    possible_paths = [
        "/global/cfs/cdirs/m636/nhadler/projects",
        "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces",
        "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzer",
    ]
    
    for path in possible_paths:
        if path not in sys.path:
            sys.path.append(path)
    
    try:
        from model_setup import dimenet10conf
        model, data, smiles = dimenet10conf()
        print(f"✓ Successfully loaded SMILES data")
        print(f"  SMILES shape: {smiles.shape}")
        print(f"  Unique rxn_ids: {smiles['rxn_id'].nunique()}")
        return smiles
    except Exception as e:
        print(f"✗ Error loading SMILES data: {e}")
        return None

def create_molecule_visualization(df_smiles, target_layer="emb", output_dir=None):
    """Create molecule visualization for a specific layer"""
    
    print(f"Creating visualization for layer: {target_layer}")
    
    # Filter for target layer
    layer_df = df_smiles[df_smiles["layer"] == target_layer].copy()
    
    if len(layer_df) == 0:
        print(f"✗ No data found for layer: {target_layer}")
        return False
    
    print(f"✓ Found {len(layer_df)} molecules for layer {target_layer}")
    
    # Group by component
    all_canvases = []
    
    for comp, grp in layer_df.groupby("component"):
        grp = grp.sort_values(["neighbor_rank", "original_molecule_index"])
        
        # Get valid molecules
        valid_mols = grp[grp['mol'].notna()]
        
        if len(valid_mols) == 0:
            print(f"  Component {comp}: No valid molecules")
            continue
        
        print(f"  Component {comp}: {len(valid_mols)} valid molecules")
        
        # Take up to 4 molecules for visualization
        mols = list(valid_mols['mol'])[:4]
        legends = []
        
        for _, row in valid_mols.head(4).iterrows():
            legend = (
                f"ID: {int(row['original_molecule_index'])}\n"
                f"Rank: {int(row['neighbor_rank'])}\n"
                f"CAS: {row['ligand_1_cas']}"
            )
            legends.append(legend)
        
        # Draw molecules
        cols = 2
        rows = max(1, math.ceil(len(mols) / cols))
        panel_w, panel_h = 300, 300
        W, H = panel_w * cols, panel_h * rows
        
        drawer = rdMolDraw2D.MolDraw2DCairo(int(W), int(H), int(panel_w), int(panel_h))
        dopts = drawer.drawOptions()
        dopts.legendFontSize = 20
        try:
            dopts.legendFraction = 0.2
        except Exception:
            pass
        
        drawer.DrawMolecules(mols, legends=legends)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        grid = Image.open(BytesIO(png))
        
        # Add title
        w, h = grid.size
        title = f"Layer: {target_layer} | Component: {int(comp)}"
        
        title_h = 40
        canvas = Image.new("RGB", (w, h + title_h), color="white")
        canvas.paste(grid, (0, title_h))
        draw = ImageDraw.Draw(canvas)
        
        try:
            font = ImageFont.load_default(size=20)
        except Exception:
            font = None
        
        draw.text((10, 10), title, fill=(0, 0, 0), font=font)
        all_canvases.append(canvas)
    
    if not all_canvases:
        print("✗ No valid components found")
        return False
    
    # Combine all canvases
    max_w = max(img.width for img in all_canvases)
    sep = 10
    total_h = sum(img.height for img in all_canvases) + sep * (len(all_canvases) - 1)
    big = Image.new("RGB", (max_w, total_h), color="white")
    
    y = 0
    for img in all_canvases:
        x = (max_w - img.width) // 2
        big.paste(img, (x, y))
        y += img.height + sep
    
    # Save image
    if output_dir:
        output_path = output_dir / f"molecule_visualization_{target_layer}.png"
        big.save(output_path, format="PNG", dpi=(600, 600))
        print(f"✓ Saved visualization to: {output_path}")
    
    return True

def main():
    """Main function"""
    print("Complete Molecule Visualization Script")
    print("="*50)
    
    # Setup
    setup_rdkit()
    
    # Load SMILES data
    smiles = load_smiles_data()
    if smiles is None:
        print("✗ Cannot proceed without SMILES data")
        return 1
    
    # Load GMM results
    base_dir = Path("/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzerResults/comprehensive_analysis_20251024_234136")
    csv_path = base_dir / "adaptive_with_zca" / "gmm_centers_and_neighbors.csv"
    
    if not csv_path.exists():
        print(f"✗ CSV file not found: {csv_path}")
        return 1
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows from CSV")
    
    # Merge with SMILES data
    df_smiles = pd.merge(
        df,
        smiles,
        left_on="original_molecule_index",
        right_index=True,
        how="left",
    )
    
    print(f"✓ Merged with SMILES data: {len(df_smiles)} rows")
    
    # Add molecule column
    PandasTools.AddMoleculeColumnToFrame(
        df_smiles,
        "ligand_1_smiles",
        "mol",
    )
    
    print("✓ Added molecule column")
    
    # Create output directory inside the analysis results folder
    output_dir = base_dir / "molecule_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Create visualizations for each layer
    layers = sorted(df_smiles['layer'].unique())
    print(f"Creating visualizations for layers: {layers}")
    
    success_count = 0
    for layer in layers:
        try:
            success = create_molecule_visualization(df_smiles, layer, output_dir)
            if success:
                success_count += 1
        except Exception as e:
            print(f"✗ Error processing layer {layer}: {e}")
    
    print(f"\n✓ Successfully created {success_count}/{len(layers)} visualizations")
    print(f"  Output directory: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
