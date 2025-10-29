#!/usr/bin/env python3
"""
Molecule Visualization Script

This script replicates the functionality from the Jupyter notebook
and creates molecular visualizations for GMM analysis results.
It saves images to a specified output folder.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, PandasTools
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import math
import argparse
from typing import List, Dict, Any, Optional

# Add the project root to the path to import model_setup
sys.path.append('/global/cfs/cdirs/m636/nhadler/projects')

def setup_rdkit():
    """Configure RDKit for better molecular visualization"""
    rdDepictor.SetPreferCoordGen(True)
    
    # Set up drawing options
    opts = Draw.MolDrawOptions()
    opts.legendFraction = 0.25
    opts.legendFontSize = 25
    
    return opts

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

def load_gmm_results(base_dir: Path, analysis_type: str) -> Optional[pd.DataFrame]:
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

def merge_with_smiles(df: pd.DataFrame, smiles: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Merge GMM results with SMILES data"""
    try:
        df_smiles = pd.merge(
            df,
            smiles,
            left_on="original_molecule_index",
            right_index=True,
            how="left",
        )
        
        # Add molecule column
        PandasTools.AddMoleculeColumnToFrame(
            df_smiles,
            "ligand_1_smiles",
            "mol",
        )
        
        print(f"✓ Merged with SMILES data")
        print(f"  Final shape: {df_smiles.shape}")
        print(f"  CAS value counts: {df_smiles['ligand_1_cas'].value_counts().head()}")
        
        return df_smiles
    except Exception as e:
        print(f"✗ Error merging with SMILES: {e}")
        return None

def create_molecule_grid(
    layer_df: pd.DataFrame,
    target_layer: str,
    neighbor_rank_max: int = 4,
    mols_per_row: int = 4,
    sub_img_size: tuple = (300, 300)
) -> List[Image.Image]:
    """Create molecular visualization grids for each component"""
    
    def _to_pil(img):
        """Convert RDKit image to PIL Image"""
        if hasattr(img, "size") and hasattr(img, "save"):
            return img
        data = getattr(img, "data", None)
        if data is not None:
            return Image.open(BytesIO(data))
        if isinstance(img, (bytes, bytearray)):
            return Image.open(BytesIO(img))
        raise TypeError(f"Unsupported image type: {type(img)}")

    all_canvases = []
    
    for comp, grp in layer_df.groupby("component"):
        grp = grp.sort_values(["neighbor_rank", "original_molecule_index"])
        mols = list(grp["mol"])
        
        # Create legends with molecule information
        legends = []
        for _, row in grp.iterrows():
            legend = (
                f"ID: {int(row['original_molecule_index'])}\n"
                f"Rank: {int(row['neighbor_rank'])}\n"
                f"L:B = {row['product_1_product_2_ratio']:.2f}\n"
                f"CAS: {row['ligand_1_cas']}"
            )
            legends.append(legend)

        # Draw molecular grid
        cols = mols_per_row
        rows = max(1, math.ceil(len(mols) / cols))
        panel_w, panel_h = sub_img_size
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

        # Add title above the grid
        w, h = grid.size
        mean_lb = grp["product_1_product_2_ratio"].mean()
        if len(grp) > 1:
            std_lb = grp["product_1_product_2_ratio"].std()
            title = f"Layer: {target_layer} | Component: {int(comp)} | Avg L:B: {mean_lb:.2f} ± {std_lb:.2f}"
        else:
            title = f"Layer: {target_layer} | Component: {int(comp)} | Avg L:B: {mean_lb:.2f}"
            
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
        
    return all_canvases

def combine_images(all_canvases: List[Image.Image], output_path: Path) -> bool:
    """Combine all component images into one large image"""
    try:
        if not all_canvases:
            print("✗ No canvases to combine")
            return False
            
        max_w = max(img.width for img in all_canvases)
        sep = 10
        total_h = sum(img.height for img in all_canvases) + sep * (len(all_canvases) - 1)
        
        big = Image.new("RGB", (max_w, total_h), color="white")
        y = 0
        
        for img in all_canvases:
            x = (max_w - img.width) // 2
            big.paste(img, (x, y))
            y += img.height + sep
            
        # Save the combined image
        big.save(output_path, format="PNG", dpi=(600, 600))
        print(f"✓ Saved combined image to: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error combining images: {e}")
        return False

def analyze_id_ranges(df_smiles: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the ID ranges in the data"""
    analysis = {}
    
    # Original molecule index range
    orig_indices = df_smiles['original_molecule_index'].values
    analysis['original_index_range'] = (orig_indices.min(), orig_indices.max())
    analysis['original_index_count'] = len(np.unique(orig_indices))
    
    # Neighbor index range (embedding indices)
    neighbor_indices = df_smiles['neighbor_index'].values
    analysis['neighbor_index_range'] = (neighbor_indices.min(), neighbor_indices.max())
    analysis['neighbor_index_count'] = len(np.unique(neighbor_indices))
    
    # CAS value analysis
    cas_values = df_smiles['ligand_1_cas'].value_counts()
    analysis['cas_value_counts'] = cas_values.head(10).to_dict()
    analysis['unique_cas_count'] = len(cas_values)
    
    # Layer analysis
    layers = df_smiles['layer'].unique()
    analysis['layers'] = list(layers)
    
    return analysis

def print_analysis_report(analysis: Dict[str, Any]):
    """Print a detailed analysis report"""
    print("\n" + "="*60)
    print("ID RANGE ANALYSIS REPORT")
    print("="*60)
    
    print(f"Original Molecule Index Range: {analysis['original_index_range']}")
    print(f"Original Molecule Index Count: {analysis['original_index_count']}")
    
    print(f"\nNeighbor Index Range (Embedding): {analysis['neighbor_index_range']}")
    print(f"Neighbor Index Count: {analysis['neighbor_index_count']}")
    
    print(f"\nUnique CAS Values: {analysis['unique_cas_count']}")
    print("Top CAS values:")
    for cas, count in list(analysis['cas_value_counts'].items())[:5]:
        print(f"  {cas}: {count} occurrences")
    
    print(f"\nAvailable Layers: {analysis['layers']}")
    
    # Check for potential issues
    print("\n" + "-"*40)
    print("POTENTIAL ISSUES:")
    print("-"*40)
    
    if analysis['original_index_range'][0] == 0 and analysis['original_index_range'][1] == 151:
        print("⚠️  Original indices are 0-151, suggesting CAS embedding indices")
        print("   This may not match the actual SMILES dataframe indices")
    
    if analysis['neighbor_index_range'][0] == 0 and analysis['neighbor_index_range'][1] == 151:
        print("⚠️  Neighbor indices are 0-151, confirming CAS embedding indices")
    
    if analysis['original_index_count'] != analysis['neighbor_index_count']:
        print("⚠️  Mismatch between original and neighbor index counts")
    
    print("="*60)

def main():
    """Main function to run the molecular visualization"""
    parser = argparse.ArgumentParser(description="Create molecular visualizations from GMM analysis")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing GMM analysis results")
    parser.add_argument("--analysis_type", type=str, default="adaptive_with_zca",
                       choices=["fixed_no_zca", "fixed_with_zca", "adaptive_no_zca", "adaptive_with_zca"],
                       help="Type of analysis to visualize")
    parser.add_argument("--target_layer", type=str, default="u_mlp_l2",
                       help="Target layer to visualize")
    parser.add_argument("--neighbor_rank_max", type=int, default=4,
                       help="Maximum neighbor rank to include")
    parser.add_argument("--mols_per_row", type=int, default=4,
                       help="Number of molecules per row in grid")
    parser.add_argument("--output_dir", type=str, default="molecule_visualizations",
                       help="Output directory for images")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only analyze ID ranges, don't create images")
    
    args = parser.parse_args()
    
    print("Molecular Visualization Script")
    print("="*50)
    
    # Setup
    setup_rdkit()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    model, data, smiles = load_model_and_data()
    if smiles is None:
        return 1
    
    # Load GMM results
    df = load_gmm_results(base_dir, args.analysis_type)
    if df is None:
        return 1
    
    # Merge with SMILES
    df_smiles = merge_with_smiles(df, smiles)
    if df_smiles is None:
        return 1
    
    # Analyze ID ranges
    analysis = analyze_id_ranges(df_smiles)
    print_analysis_report(analysis)
    
    if args.analyze_only:
        print("Analysis complete. Exiting without creating images.")
        return 0
    
    # Filter for target layer
    layer_df = df_smiles[
        (df_smiles["layer"] == args.target_layer) &
        (df_smiles["neighbor_rank"] < args.neighbor_rank_max)
    ]
    
    print(f"\nVisualizing layer: {args.target_layer}")
    print(f"Components found: {layer_df['component'].nunique()}")
    print(f"Molecules to visualize: {len(layer_df)}")
    
    # Create molecular grids
    all_canvases = create_molecule_grid(
        layer_df,
        args.target_layer,
        args.neighbor_rank_max,
        args.mols_per_row
    )
    
    # Save combined image
    output_filename = f"{args.analysis_type}_{args.target_layer}_molecules.png"
    output_path = output_dir / output_filename
    
    success = combine_images(all_canvases, output_path)
    
    if success:
        print(f"\n✓ Visualization complete!")
        print(f"  Output saved to: {output_path}")
        print(f"  Components visualized: {len(all_canvases)}")
        return 0
    else:
        print(f"\n✗ Visualization failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())