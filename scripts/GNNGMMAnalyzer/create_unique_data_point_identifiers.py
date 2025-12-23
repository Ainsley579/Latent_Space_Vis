#!/usr/bin/env python3
"""
Create Unique Identifiers for Each Data Point

This script creates unique identifiers that combine reaction (rxn_id) and conformer (catalyst_key)
information, and generates a mapping file for the frontend to distinguish all data points.
"""

import pandas as pd
from pathlib import Path
import json
import argparse


def create_unique_identifiers(csv_path, output_dir):
    """Create unique identifiers for each data point"""
    
    print(f"\nProcessing: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create unique identifier combining rxn_id and catalyst_key
    # Replace underscores in catalyst_key to avoid conflicts
    df['unique_id'] = df['rxn_id'] + '_' + df['catalyst_key'].str.replace('_', '-', regex=False)
    
    # Also create a shorter display name
    df['display_name'] = df['rxn_id'] + ' (' + df['catalyst_key'].str.extract(r'conformer-(\d+)')[0].fillna('') + ')'
    
    # Check for real_id or sample_id column
    id_column = 'real_id' if 'real_id' in df.columns else 'sample_id'
    if id_column not in df.columns:
        print(f"  Warning: No ID column found, skipping")
        return {}, df
    
    # Create a comprehensive mapping
    mapping = {}
    for idx, row in df.iterrows():
        real_id = int(row[id_column])
        unique_id = row['unique_id']
        
        mapping[real_id] = {
            'unique_id': unique_id,
            'display_name': row['display_name'],
            'rxn_id': row['rxn_id'],
            'catalyst_key': row['catalyst_key'],
            'ddG': float(row['ddG']) if pd.notna(row['ddG']) else None,
            'split_label': row.get('split_label', 'unknown'),
            'startingmat_smiles': row.get('startingmat_1_smiles', ''),
            'product_smiles': row.get('product_1_smiles', ''),
            'ligand_smiles': row.get('ligand_1_smiles', ''),
        }
    
    # Save mapping
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mapping_file = output_dir / f"unique_identifiers_{Path(csv_path).stem}.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    # Also save as CSV for easy lookup
    csv_file = output_dir / f"unique_identifiers_{Path(csv_path).stem}.csv"
    id_col = 'real_id' if 'real_id' in df.columns else 'sample_id'
    df_export = df[[id_col, 'unique_id', 'display_name', 'rxn_id', 'catalyst_key', 'ddG', 'split_label']].copy()
    df_export.rename(columns={id_col: 'real_id'}, inplace=True)  # Standardize to real_id
    df_export.to_csv(csv_file, index=False)
    
    print(f"  Created mapping: {len(mapping)} entries")
    print(f"  Saved to: {mapping_file}")
    print(f"  Saved to: {csv_file}")
    
    return mapping, df


def create_combined_mapping(data_dir, output_dir):
    """Create combined mapping from all CSV files"""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Creating Unique Identifiers for All Data Points")
    print("="*60)
    
    # Process all CSV files
    csv_files = [
        data_dir / 'cluster_transition_matrices_1_individual_real_id.csv',
        data_dir / 'cluster_transition_matrices_2_individual_real_id.csv',
        data_dir / 'sample_component_paths_1_with_real_id.csv',
        data_dir / 'sample_component_paths_2_with_real_id.csv',
    ]
    
    all_mappings = {}
    all_dataframes = []
    
    for csv_file in csv_files:
        if csv_file.exists():
            mapping, df = create_unique_identifiers(csv_file, output_dir)
            all_mappings.update(mapping)
            all_dataframes.append(df)
    
    # Create combined mapping
    combined_mapping_file = output_dir / "unique_identifiers_combined.json"
    with open(combined_mapping_file, 'w') as f:
        json.dump(all_mappings, f, indent=2)
    
    # Create combined CSV
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['real_id'], keep='first')
        combined_csv = output_dir / "unique_identifiers_combined.csv"
        combined_df[['real_id', 'unique_id', 'display_name', 'rxn_id', 'catalyst_key', 'ddG', 'split_label']].to_csv(combined_csv, index=False)
    
    print(f"\n" + "="*60)
    print(f"Combined mapping created: {len(all_mappings)} unique data points")
    print(f"  Saved to: {combined_mapping_file}")
    print(f"  Saved to: {combined_csv}")
    print("="*60)
    
    # Create summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total unique real_ids: {len(all_mappings)}")
    
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True).drop_duplicates(subset=['real_id'], keep='first')
        print(f"  Unique rxn_ids: {combined_df['rxn_id'].nunique()}")
        print(f"  Unique catalyst_keys: {combined_df['catalyst_key'].nunique()}")
        print(f"  Unique combinations: {combined_df['unique_id'].nunique()}")
        print(f"  ddG range: {combined_df['ddG'].min():.2f} to {combined_df['ddG'].max():.2f}")
    
    return all_mappings


def create_frontend_lookup_table(data_dir, output_dir):
    """Create a lookup table optimized for frontend use"""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Load the combined mapping
    mapping_file = output_dir / "unique_identifiers_combined.json"
    if not mapping_file.exists():
        print("Error: Combined mapping not found. Run create_combined_mapping first.")
        return
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Create optimized lookup tables
    lookup_by_real_id = mapping  # Already keyed by real_id
    
    # Create reverse lookup by unique_id
    lookup_by_unique_id = {}
    for real_id, data in mapping.items():
        unique_id = data['unique_id']
        lookup_by_unique_id[unique_id] = {
            'real_id': int(real_id),
            **data
        }
    
    # Create lookup by rxn_id (grouped)
    lookup_by_rxn_id = {}
    for real_id, data in mapping.items():
        rxn_id = data['rxn_id']
        if rxn_id not in lookup_by_rxn_id:
            lookup_by_rxn_id[rxn_id] = []
        lookup_by_rxn_id[rxn_id].append({
            'real_id': int(real_id),
            **data
        })
    
    # Save lookup tables
    output = {
        'by_real_id': lookup_by_real_id,
        'by_unique_id': lookup_by_unique_id,
        'by_rxn_id': lookup_by_rxn_id,
    }
    
    lookup_file = output_dir / "frontend_lookup.json"
    with open(lookup_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nFrontend lookup table created:")
    print(f"  Saved to: {lookup_file}")
    print(f"  Contains:")
    print(f"    - by_real_id: {len(lookup_by_real_id)} entries")
    print(f"    - by_unique_id: {len(lookup_by_unique_id)} entries")
    print(f"    - by_rxn_id: {len(lookup_by_rxn_id)} entries (grouped)")
    
    return output


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create Unique Identifiers for Data Points")
    parser.add_argument('--data_dir', type=str, 
                       default='../../frontend/public/data',
                       help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str,
                       default='../../frontend/public/data/unique_identifiers',
                       help='Output directory for mapping files')
    
    args = parser.parse_args()
    
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
    
    # Create mappings
    create_combined_mapping(data_dir, output_dir)
    
    # Create frontend lookup table
    create_frontend_lookup_table(data_dir, output_dir)
    
    print("\n✅ Done! All data points now have unique identifiers.")
    print("\nUsage in frontend:")
    print("  const lookup = await fetch('/data/unique_identifiers/frontend_lookup.json')")
    print("  const data = lookup.by_real_id[realId]")
    print("  console.log(data.unique_id, data.display_name, data.ddG)")


if __name__ == "__main__":
    main()

