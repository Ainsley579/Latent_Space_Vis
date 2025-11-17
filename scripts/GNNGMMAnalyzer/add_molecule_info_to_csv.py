#!/usr/bin/env python3
"""
Add Molecule Information Columns to Cluster Transition Matrices CSV Files

This script adds molecule structure and information columns from the original dataset
to the cluster transition matrices CSV files. It handles both 10-conf and no-aug datasets.
"""

import json
import pandas as pd
from pathlib import Path
import argparse


def load_dataset_mapping(dataset_type='10-conf', libra_path=None):
    """Load dataset mapping from split_0.json and merge with HF-Literature.tsv"""
    
    if libra_path is None:
        script_dir = Path(__file__).parent
        libra_path = script_dir.parent / "Libra-ML-Modeling"
    
    libra_path = Path(libra_path)
    
    if not libra_path.exists():
        print(f"Error: Libra-ML-Modeling not found at {libra_path}")
        return None
    
    # Load split_0.json based on dataset type
    if dataset_type == '10-conf':
        json_path = libra_path / "model-training" / "butene" / "dimenet-10conf" / "data" / "jsons" / "split_0.json"
    elif dataset_type == 'no-aug':
        json_path = libra_path / "model-training" / "butene" / "dimenet-noaug" / "data" / "jsons" / "split_0.json"
    else:
        print(f"Error: Unknown dataset type: {dataset_type}")
        return None
    
    if not json_path.exists():
        print(f"Error: split_0.json not found at {json_path}")
        return None
    
    # Load HF-Literature.tsv
    desc_path = libra_path / "data" / "HF-Literature.tsv"
    if not desc_path.exists():
        print(f"Error: HF-Literature.tsv not found at {desc_path}")
        return None
    
    try:
        # Load split data
        with open(json_path, 'r') as f:
            split_data = json.load(f)
        
        # Get train/test/val dataframes
        train_df = pd.DataFrame(split_data['train'])
        val_df = pd.DataFrame(split_data['val'])
        test_df = pd.DataFrame(split_data['test'])
        
        print(f"\nLoaded {dataset_type} dataset:")
        print(f"  Train: {len(train_df)} entries")
        print(f"  Val: {len(val_df)} entries")
        print(f"  Test: {len(test_df)} entries")
        print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} entries")
        
        # Combine in order: train, val, test (this matches the original dataset order)
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Load HF-Literature.tsv
        desc = pd.read_csv(desc_path, sep="\t")
        print(f"\nLoaded HF-Literature.tsv: {desc.shape}")
        
        # Merge with HF-Literature to get molecule information
        merged_df = combined_df.merge(desc, on='rxn_id', how='left')
        
        print(f"Merged dataset: {merged_df.shape}")
        print(f"  Columns: {len(merged_df.columns)}")
        
        # Create mapping from index to all molecule info
        # The index in combined_df corresponds to real_id in CSV files
        index_to_info = {}
        for idx, row in merged_df.iterrows():
            index_to_info[idx] = row.to_dict()
        
        print(f"  Created mapping for {len(index_to_info)} indices")
        print(f"  Index range: {min(index_to_info.keys())} - {max(index_to_info.keys())}")
        
        return index_to_info, merged_df.columns.tolist()
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def select_molecule_columns(available_columns):
    """Select relevant molecule information columns to add"""
    
    # Priority columns for molecule structure and information
    priority_columns = [
        'rxn_id',
        'rxn_name',
        'catalyst_key',
        'ddG',
        'startingmat_1_smiles',
        'startingmat_1_cas',
        'startingmat_1_name',
        'product_1_smiles',
        'product_1_cas',
        'product_2_smiles',
        'product_2_cas',
        'catalyst_1_smiles',
        'catalyst_1_cas',
        'catalyst_1_name',
        'ligand_1_smiles',
        'ligand_1_cas',
        'ligand_1_name',
        'solvent_1_smiles',
        'solvent_1_cas',
        'reagent_1_smiles',
        'reagent_1_cas',
        'reagent_2_smiles',
        'reagent_2_cas',
        'temperature_deg_c',
        'time_h',
        'atmosphere',
        'reaction_conditions',
    ]
    
    # Select columns that exist in available_columns
    selected = [col for col in priority_columns if col in available_columns]
    
    # Add any other columns with 'smiles' or 'cas' in the name
    smiles_cas_cols = [col for col in available_columns if ('smiles' in col.lower() or 'cas' in col.lower()) and col not in selected]
    selected.extend(smiles_cas_cols)
    
    return selected


def add_molecule_info_to_csv(csv_path, index_to_info_mapping, molecule_columns, dataset_type='10-conf'):
    """Add molecule information columns to a CSV file based on real_id or sample_id"""
    
    print(f"\nProcessing: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Check if real_id or sample_id column exists
    id_column = None
    if 'real_id' in df.columns:
        id_column = 'real_id'
    elif 'sample_id' in df.columns:
        id_column = 'sample_id'
    else:
        print(f"  Warning: Neither 'real_id' nor 'sample_id' column found in {csv_path.name}, skipping")
        return False
    
    # Add molecule information columns
    print(f"  Adding {len(molecule_columns)} molecule information columns...")
    
    for col in molecule_columns:
        def get_molecule_value(real_id):
            if pd.isna(real_id):
                return None
            real_id_int = int(real_id)
            if real_id_int in index_to_info_mapping:
                value = index_to_info_mapping[real_id_int].get(col, None)
                # Handle list/array values (like reaction_conditions)
                if isinstance(value, list):
                    return str(value)  # Convert to string for CSV
                return value
            return None
        
        df[col] = df[id_column].apply(get_molecule_value)
    
    # Count non-null values for each new column
    print(f"  Added columns with data:")
    for col in molecule_columns:
        non_null = df[col].notna().sum()
        if non_null > 0:
            print(f"    {col}: {non_null}/{len(df)} non-null values")
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved to: {csv_path}")
    
    return True


def detect_dataset_type(csv_path, index_to_info_10conf, index_to_info_noaug):
    """Detect which dataset type a CSV file belongs to based on real_id ranges"""
    
    df = pd.read_csv(csv_path, nrows=100)
    
    id_column = None
    if 'real_id' in df.columns:
        id_column = 'real_id'
    elif 'sample_id' in df.columns:
        id_column = 'sample_id'
    else:
        return None
    
    # Check if real_ids are in 10-conf or no-aug range
    sample_ids = df[id_column].head(10).tolist()
    
    in_10conf = sum(1 for sid in sample_ids if int(sid) in index_to_info_10conf) if index_to_info_10conf else 0
    in_noaug = sum(1 for sid in sample_ids if int(sid) in index_to_info_noaug) if index_to_info_noaug else 0
    
    if in_10conf > in_noaug:
        return '10-conf'
    elif in_noaug > in_10conf:
        return 'no-aug'
    else:
        # Default to 10-conf if uncertain (most files are 10-conf)
        return '10-conf'


def process_all_csv_files(data_dir, dataset_type=None):
    """Process all cluster transition matrices CSV files"""
    
    data_dir = Path(data_dir)
    
    # Load both dataset mappings
    print("="*60)
    print("Loading Dataset Mappings")
    print("="*60)
    
    mapping_10conf, cols_10conf = load_dataset_mapping('10-conf')
    mapping_noaug, cols_noaug = load_dataset_mapping('no-aug')
    
    if mapping_10conf is None or mapping_noaug is None:
        print("Error: Could not load dataset mappings. Exiting.")
        return
    
    # Select molecule columns
    molecule_columns_10conf = select_molecule_columns(cols_10conf)
    molecule_columns_noaug = select_molecule_columns(cols_noaug)
    
    print(f"\nSelected molecule columns for 10-conf: {len(molecule_columns_10conf)}")
    print(f"  {molecule_columns_10conf[:10]}...")
    print(f"\nSelected molecule columns for no-aug: {len(molecule_columns_noaug)}")
    print(f"  {molecule_columns_noaug[:10]}...")
    
    # Files to process
    files_to_process = []
    
    # Main files
    files_to_process.extend([
        data_dir / 'cluster_transition_matrices_1_individual_real_id.csv',
        data_dir / 'cluster_transition_matrices_2_individual_real_id.csv',
        data_dir / 'sample_component_paths_1_with_real_id.csv',
        data_dir / 'sample_component_paths_2_with_real_id.csv',
    ])
    
    # Files in subdirectories
    subdirs = [
        data_dir / 'real_id_layer_transitions_1',
        data_dir / 'real_id_layer_transitions_2',
        data_dir / 'real_id_component_transitions_1',
        data_dir / 'real_id_component_transitions_2',
    ]
    
    for subdir in subdirs:
        if subdir.exists():
            for csv_file in subdir.glob('*.csv'):
                if 'real_id' in csv_file.name or 'real_id' in str(csv_file):
                    files_to_process.append(csv_file)
    
    # Process each file
    print("\n" + "="*60)
    print("Processing CSV Files")
    print("="*60)
    
    processed = 0
    skipped = 0
    
    for csv_path in files_to_process:
        if not csv_path.exists():
            print(f"\nSkipping (not found): {csv_path}")
            skipped += 1
            continue
        
        # Detect dataset type if not specified
        if dataset_type is None:
            detected_type = detect_dataset_type(csv_path, mapping_10conf, mapping_noaug)
        else:
            detected_type = dataset_type
        
        if detected_type == '10-conf':
            mapping = mapping_10conf
            mol_cols = molecule_columns_10conf
        elif detected_type == 'no-aug':
            mapping = mapping_noaug
            mol_cols = molecule_columns_noaug
        else:
            print(f"\nSkipping (unknown dataset type): {csv_path}")
            skipped += 1
            continue
        
        if add_molecule_info_to_csv(csv_path, mapping, mol_cols, detected_type):
            processed += 1
        else:
            skipped += 1
    
    print("\n" + "="*60)
    print(f"Processing complete: {processed} files processed, {skipped} files skipped")
    print("="*60)
    
    return processed, skipped


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Add Molecule Information to Cluster Transition Matrices CSV Files")
    parser.add_argument('--data_dir', type=str, 
                       default='../../frontend/public/data',
                       help='Directory containing CSV files (default: frontend/public/data)')
    parser.add_argument('--libra_path', type=str, default=None,
                       help='Path to Libra-ML-Modeling directory (default: auto-detect)')
    parser.add_argument('--dataset_type', type=str, default=None, choices=['10-conf', 'no-aug'],
                       help='Dataset type: 10-conf or no-aug (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    if args.data_dir.startswith('../') or args.data_dir.startswith('../../'):
        data_dir = (script_dir / args.data_dir).resolve()
    else:
        data_dir = Path(args.data_dir)
    
    print("="*60)
    print("Add Molecule Information to Cluster Transition Matrices")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Libra path: {args.libra_path or 'auto-detect'}")
    if args.dataset_type:
        print(f"Dataset type: {args.dataset_type}")
    else:
        print(f"Dataset type: auto-detect")
    print("="*60)
    
    # Process all CSV files
    process_all_csv_files(data_dir, args.dataset_type)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

