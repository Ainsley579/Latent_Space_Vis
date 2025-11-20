#!/usr/bin/env python3
"""
Generate Molecule Structure Images from SMILES

This script generates molecule structure images for each data point in the CSV files
and saves them in a format that can be easily linked from the frontend.
"""

import pandas as pd
from pathlib import Path
import argparse
import json
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")
    print("Or: pip install rdkit-pypi")


def generate_molecule_image(smiles, output_path, size=(400, 400), title=None):
    """Generate a molecule image from SMILES string"""
    
    if not RDKIT_AVAILABLE:
        return False
    
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  Warning: Could not parse SMILES: {smiles[:50]}...")
            return False
        
        # Generate image
        img = Draw.MolToImage(mol, size=size)
        
        # Add title if provided
        if title:
            from PIL import Image, ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except:
                font = ImageFont.load_default()
            # Add title at the bottom
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (size[0] - text_width) // 2
            y = size[1] - text_height - 10
            draw.text((x, y), title, fill='black', font=font)
        
        # Save image
        img.save(output_path)
        return True
        
    except Exception as e:
        print(f"  Error generating image for {smiles[:50]}: {e}")
        return False


def generate_molecule_image_high_quality(smiles, output_path, size=(400, 400), title=None):
    """Generate a high-quality molecule image using rdMolDraw2D (with fallback)"""
    
    if not RDKIT_AVAILABLE:
        return False
    
    try:
        # Try using rdMolDraw2D for better quality
        try:
            from rdkit.Chem.Draw import rdMolDraw2D
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Use rdMolDraw2D for better quality
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            # Save image
            drawer.WriteDrawingText(str(output_path))
            
            # Add title if provided
            if title:
                from PIL import Image, ImageDraw, ImageFont
                img = Image.open(output_path)
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                except:
                    font = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), title, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (size[0] - text_width) // 2
                y = size[1] - text_height - 10
                draw.text((x, y), title, fill='black', font=font)
                img.save(output_path)
            
            return True
        except (ImportError, AttributeError):
            # Fallback to simple method if rdMolDraw2D not available
            return generate_molecule_image(smiles, output_path, size, title)
        
    except Exception as e:
        # Fallback to simple method
        return generate_molecule_image(smiles, output_path, size, title)


def extract_unique_molecules_from_csv(csv_path, id_column='real_id'):
    """Extract unique molecules from a CSV file"""
    
    print(f"\nReading: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check for id column
    if id_column not in df.columns:
        if 'sample_id' in df.columns:
            id_column = 'sample_id'
        else:
            print(f"  Warning: No ID column found, skipping")
            return {}
    
    # Get SMILES columns
    smiles_columns = [c for c in df.columns if 'smiles' in c.lower() and c != id_column]
    
    if not smiles_columns:
        print(f"  Warning: No SMILES columns found")
        return {}
    
    # Extract unique molecules
    molecules = {}
    
    for idx, row in df.iterrows():
        real_id = int(row[id_column])
        
        if real_id not in molecules:
            molecules[real_id] = {}
        
        # Store all SMILES for this real_id
        for col in smiles_columns:
            smiles = row[col]
            if pd.notna(smiles) and smiles and smiles.strip():
                # Use column name without '_smiles' as key
                key = col.replace('_smiles', '').replace('_1', '').replace('_2', '')
                molecules[real_id][key] = str(smiles)
    
    print(f"  Found {len(molecules)} unique IDs with molecule data")
    return molecules


def generate_images_for_molecules(molecules, output_dir, image_size=(400, 400), 
                                  primary_molecule='startingmat', include_all=True):
    """Generate images for all molecules"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different molecule types
    molecule_types = set()
    for mol_data in molecules.values():
        molecule_types.update(mol_data.keys())
    
    print(f"\nGenerating images for {len(molecules)} unique IDs...")
    print(f"  Molecule types found: {sorted(molecule_types)}")
    
    # Create mapping file (real_id -> image paths)
    mapping = {}
    
    generated_count = 0
    skipped_count = 0
    
    for real_id, mol_data in molecules.items():
        mapping[real_id] = {}
        
        # Generate primary molecule image (starting material by default)
        if primary_molecule in mol_data:
            smiles = mol_data[primary_molecule]
            img_path = output_dir / f"{real_id}_primary.png"
            if generate_molecule_image_high_quality(smiles, img_path, size=image_size, 
                                                   title=f"ID: {real_id}"):
                mapping[real_id]['primary'] = f"{real_id}_primary.png"
                generated_count += 1
            else:
                skipped_count += 1
        
        # Generate images for all other molecules if requested
        if include_all:
            for mol_type, smiles in mol_data.items():
                if mol_type == primary_molecule:
                    continue  # Already generated
                
                img_path = output_dir / f"{real_id}_{mol_type}.png"
                if generate_molecule_image_high_quality(smiles, img_path, size=image_size,
                                                      title=f"{mol_type} (ID: {real_id})"):
                    if mol_type not in mapping[real_id]:
                        mapping[real_id][mol_type] = {}
                    mapping[real_id][mol_type] = f"{real_id}_{mol_type}.png"
    
    # Save mapping file
    mapping_path = output_dir / "molecule_image_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n  Generated: {generated_count} primary images")
    print(f"  Skipped: {skipped_count} molecules")
    print(f"  Mapping saved to: {mapping_path}")
    
    return mapping


def process_all_csv_files(data_dir, output_dir, image_size=(400, 400), 
                          primary_molecule='product', include_all=True):
    """Process all CSV files and generate molecule images"""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    print("="*60)
    print("Generate Molecule Structure Images")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {image_size}")
    print(f"Primary molecule: {primary_molecule}")
    print("="*60)
    
    if not RDKIT_AVAILABLE:
        print("\n❌ Error: RDKit is not available!")
        print("Please install RDKit:")
        print("  conda install -c conda-forge rdkit")
        print("  OR")
        print("  pip install rdkit-pypi")
        return
    
    # Collect all unique molecules from all CSV files
    all_molecules = {}
    
    # Main CSV files
    csv_files = [
        data_dir / 'cluster_transition_matrices_1_individual_real_id.csv',
        data_dir / 'cluster_transition_matrices_2_individual_real_id.csv',
        data_dir / 'sample_component_paths_1_with_real_id.csv',
        data_dir / 'sample_component_paths_2_with_real_id.csv',
    ]
    
    for csv_file in csv_files:
        if csv_file.exists():
            molecules = extract_unique_molecules_from_csv(csv_file)
            # Merge into all_molecules
            for real_id, mol_data in molecules.items():
                if real_id not in all_molecules:
                    all_molecules[real_id] = {}
                all_molecules[real_id].update(mol_data)
    
    if not all_molecules:
        print("\n❌ No molecules found in CSV files!")
        return
    
    print(f"\nTotal unique IDs: {len(all_molecules)}")
    
    # Generate images
    mapping = generate_images_for_molecules(all_molecules, output_dir, image_size, 
                                          primary_molecule, include_all)
    
    # Create a summary CSV file for easy lookup
    summary_data = []
    for real_id, img_paths in mapping.items():
        row = {'real_id': real_id, 'primary_image': img_paths.get('primary', '')}
        for mol_type, path in img_paths.items():
            if mol_type != 'primary':
                row[f'{mol_type}_image'] = path
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "molecule_image_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary CSV saved to: {summary_path}")
    
    print("\n" + "="*60)
    print("✅ Molecule image generation complete!")
    print("="*60)
    print(f"\nUsage in frontend:")
    print(f"  Primary image: /data/molecule_images/{{real_id}}_primary.png")
    print(f"  Mapping file: /data/molecule_images/molecule_image_mapping.json")
    print(f"  Summary CSV: /data/molecule_images/molecule_image_summary.csv")
    print("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate Molecule Structure Images from SMILES")
    parser.add_argument('--data_dir', type=str, 
                       default='../../frontend/public/data',
                       help='Directory containing CSV files (default: frontend/public/data)')
    parser.add_argument('--output_dir', type=str,
                       default='../../frontend/public/data/molecule_images',
                       help='Output directory for images (default: frontend/public/data/molecule_images)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[400, 400],
                       help='Image size in pixels (default: 400 400)')
    parser.add_argument('--primary_molecule', type=str, default='product',
                       help='Primary molecule type to generate (default: product, options: product, startingmat, ligand, solvent)')
    parser.add_argument('--include_all', action='store_true', default=True,
                       help='Generate images for all molecule types (default: True)')
    parser.add_argument('--primary_only', action='store_true',
                       help='Only generate primary molecule images')
    
    args = parser.parse_args()
    
    if args.primary_only:
        args.include_all = False
    
    # Resolve paths
    script_dir = Path(__file__).parent
    if args.data_dir.startswith('../') or args.data_dir.startswith('../../'):
        data_dir = (script_dir / args.data_dir).resolve()
    else:
        data_dir = Path(args.data_dir)
    
    if args.output_dir.startswith('../') or args.output_dir.startswith('../../'):
        output_dir = (script_dir / args.output_dir).resolve()
    else:
        output_dir = Path(args.output_dir)
    
    # Process all CSV files
    process_all_csv_files(data_dir, output_dir, tuple(args.image_size), 
                         args.primary_molecule, args.include_all)


if __name__ == "__main__":
    main()

