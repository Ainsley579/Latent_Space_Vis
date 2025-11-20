#!/usr/bin/env python3
"""
Add Split Labels (train/test/experimental) to Cluster Transition Matrices CSV Files

This script adds a "split_label" column to cluster transition matrices CSV files
based on the real_id mapping to train/test/val splits from split_0.json.
"""

import json
import pandas as pd
from pathlib import Path
import argparse


def create_index_to_split_mapping(libra_path=None):
    """Create mapping from original dataset index to split label (train/test/val)"""
    
    if libra_path is None:
        script_dir = Path(__file__).parent
        libra_path = script_dir.parent / "Libra-ML-Modeling"
    
    libra_path = Path(libra_path)
    
    if not libra_path.exists():
        print(f"Error: Libra-ML-Modeling not found at {libra_path}")
        return None
    
    # Load split_0.json
    json_path = libra_path / "model-training" / "butene" / "dimenet-10conf" / "data" / "jsons" / "split_0.json"
    
    if not json_path.exists():
        print(f"Error: split_0.json not found at {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            split_data = json.load(f)
        
        # Get train/test/val dataframes
        train_df = pd.DataFrame(split_data['train'])
        val_df = pd.DataFrame(split_data['val'])
        test_df = pd.DataFrame(split_data['test'])
        
        print(f"Loaded split data:")
        print(f"  Train: {len(train_df)} entries")
        print(f"  Val: {len(val_df)} entries")
        print(f"  Test: {len(test_df)} entries")
        print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} entries")
        
        # Combine in order: train, val, test (this matches the original dataset order)
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Add split labels (map val to experimental as requested)
        combined_df['split_label'] = 'train'
        combined_df.loc[len(train_df):len(train_df)+len(val_df)-1, 'split_label'] = 'experimental'
        combined_df.loc[len(train_df)+len(val_df):, 'split_label'] = 'test'
        
        # Create mapping from index to split label
        index_to_split = dict(zip(combined_df.index, combined_df['split_label']))
        
        print(f"\nSplit label distribution:")
        print(combined_df['split_label'].value_counts())
        print(f"\nIndex range: {min(index_to_split.keys())} - {max(index_to_split.keys())}")
        
        return index_to_split
        
    except Exception as e:
        print(f"Error loading split data: {e}")
        import traceback
        traceback.print_exc()
        return None


def add_split_labels_to_csv(csv_path, index_to_split_mapping, output_path=None):
    """Add split_label column to a CSV file based on real_id"""
    
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
    
    # Add split_label column
    def get_split_label(id_value):
        if pd.isna(id_value):
            return 'unknown'
        id_int = int(id_value)
        return index_to_split_mapping.get(id_int, 'experimental')
    
    df['split_label'] = df[id_column].apply(get_split_label)
    
    # Check for unknown labels (only warn about IDs not in mapping)
    unknown_count = (df['split_label'] == 'experimental').sum()
    # Count how many are actually unknown (not in mapping)
    unknown_ids = df[df[id_column].apply(lambda x: int(x) if not pd.isna(x) else -1) not in index_to_split_mapping][id_column].unique()
    actual_unknown = len(unknown_ids) if len(unknown_ids) > 0 else 0
    if actual_unknown > 0:
        print(f"  Warning: {actual_unknown} entries have real_id not in mapping (mapped to 'experimental')")
    
    # Save updated CSV
    if output_path is None:
        output_path = csv_path
    
    df.to_csv(output_path, index=False)
    print(f"  ✓ Added split_label column")
    print(f"  ✓ Saved to: {output_path}")
    print(f"  Split label distribution:")
    print(f"    {df['split_label'].value_counts().to_dict()}")
    
    return True


def process_all_cluster_transition_files(data_dir, index_to_split_mapping, version=None):
    """Process all cluster transition matrices CSV files"""
    
    data_dir = Path(data_dir)
    
    # Files to process
    files_to_process = []
    
    if version is None or version == 1:
        files_to_process.extend([
            data_dir / 'cluster_transition_matrices_1_individual_real_id.csv',
            data_dir / 'cluster_transition_matrices_1_with_real_id.csv',
            data_dir / 'sample_component_paths_1_with_real_id.csv',
        ])
    
    if version is None or version == 2:
        files_to_process.extend([
            data_dir / 'cluster_transition_matrices_2_individual_real_id.csv',
            data_dir / 'cluster_transition_matrices_2_with_real_id.csv',
            data_dir / 'sample_component_paths_2_with_real_id.csv',
        ])
    
    # Also process files in subdirectories
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
    processed = 0
    skipped = 0
    
    for csv_path in files_to_process:
        if csv_path.exists():
            if add_split_labels_to_csv(csv_path, index_to_split_mapping):
                processed += 1
            else:
                skipped += 1
        else:
            print(f"\nSkipping (not found): {csv_path}")
            skipped += 1
    
    print("\n" + "="*60)
    print(f"Processing complete: {processed} files processed, {skipped} files skipped")
    print("="*60)
    
    return processed, skipped


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Add Split Labels to Cluster Transition Matrices CSV Files")
    parser.add_argument('--data_dir', type=str, 
                       default='../../frontend/public/data',
                       help='Directory containing CSV files (default: frontend/public/data)')
    parser.add_argument('--libra_path', type=str, default=None,
                       help='Path to Libra-ML-Modeling directory (default: auto-detect)')
    parser.add_argument('--version', type=int, default=None, choices=[1, 2],
                       help='Process only version 1 or 2 (default: both)')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    if args.data_dir.startswith('../') or args.data_dir.startswith('../../'):
        data_dir = (script_dir / args.data_dir).resolve()
    else:
        data_dir = Path(args.data_dir)
    
    print("="*60)
    print("Add Split Labels to Cluster Transition Matrices")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Libra path: {args.libra_path or 'auto-detect'}")
    if args.version:
        print(f"Version: {args.version}")
    print("="*60)
    
    # Create index to split mapping
    print("\nCreating index to split label mapping...")
    index_to_split_mapping = create_index_to_split_mapping(args.libra_path)
    
    if index_to_split_mapping is None:
        print("Error: Could not create index to split mapping. Exiting.")
        return
    
    # Process all CSV files
    print("\nProcessing CSV files...")
    process_all_cluster_transition_files(data_dir, index_to_split_mapping, args.version)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

