#!/usr/bin/env python3
"""
Create Unique to Original Index Mapping from JSON (Handles Truncated JSON)

This script extracts rxn_ids from the split_0.json file even if it's truncated,
then creates the mapping from unique embedding indices to original dataset indices.
"""

import json
import re
import pandas as pd
from pathlib import Path

def extract_rxn_ids_from_json(json_path):
    """Extract rxn_id arrays from JSON file, handling truncation"""
    
    with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract rxn_id arrays using regex
    train_match = re.search(r'"train":\s*\{[^}]*"rxn_id":\s*\[(.*?)\]', content, re.DOTALL)
    
    train_rxn_ids = []
    
    if train_match:
        try:
            train_rxn_ids = json.loads('[' + train_match.group(1) + ']')
        except:
            pass
    
    return train_rxn_ids

def extract_rxn_ids_from_multiple_splits(json_dir, max_splits=10):
    """Extract rxn_ids from multiple split JSON files to get better coverage"""
    
    print(f"Extracting rxn_ids from multiple splits in: {json_dir}")
    
    all_rxn_ids = []
    
    for split_num in range(max_splits):
        json_path = json_dir / f"split_{split_num}.json"
        if not json_path.exists():
            continue
        
        try:
            train_rxn_ids = extract_rxn_ids_from_json(json_path)
            if train_rxn_ids:
                all_rxn_ids.extend(train_rxn_ids)
                print(f"  Split {split_num}: {len(train_rxn_ids)} rxn_ids")
        except Exception as e:
            print(f"  Split {split_num}: Error - {e}")
    
    print(f"  Total rxn_ids collected: {len(all_rxn_ids)}")
    unique_count = len(set(all_rxn_ids))
    print(f"  Unique rxn_ids: {unique_count}")
    
    return all_rxn_ids

def create_unique_to_original_mapping():
    """Create mapping from unique embedding indices to original SMILES indices"""
    
    print("="*60)
    print("Creating unique to original index mapping from JSON")
    print("="*60)
    
    script_dir = Path(__file__).parent
    libra_path = script_dir.parent / "Libra-ML-Modeling"
    
    if not libra_path.exists():
        print(f"Error: Libra-ML-Modeling not found at {libra_path}")
        return None
    
    # Extract rxn_ids from multiple split JSON files for better coverage
    json_dir = libra_path / "model-training" / "butene" / "dimenet-10conf" / "data" / "jsons"
    if not json_dir.exists():
        print(f"Error: JSON directory not found at {json_dir}")
        return None
    
    # Extract from split_0 as the primary dataset (this represents the original dataset)
    # Now that the JSON is not truncated, we can load train, val, and test
    split_0_path = json_dir / "split_0.json"
    
    try:
        with open(split_0_path, 'r') as f:
            split_0_data = json.load(f)
        
        # Extract rxn_ids from train, val, and test
        train_rxn_ids = []
        val_rxn_ids = []
        test_rxn_ids = []
        
        if 'train' in split_0_data:
            train_df = pd.DataFrame(split_0_data["train"])
            if 'rxn_id' in train_df.columns:
                train_rxn_ids = train_df['rxn_id'].tolist()
        
        if 'val' in split_0_data:
            val_df = pd.DataFrame(split_0_data["val"])
            if 'rxn_id' in val_df.columns:
                val_rxn_ids = val_df['rxn_id'].tolist()
        
        if 'test' in split_0_data:
            test_df = pd.DataFrame(split_0_data["test"])
            if 'rxn_id' in test_df.columns:
                test_rxn_ids = test_df['rxn_id'].tolist()
        
        # Combine all rxn_ids (train + val + test)
        split_0_rxn_ids = train_rxn_ids + val_rxn_ids + test_rxn_ids
        
        print(f"\nSplit 0 dataset:")
        print(f"  Train: {len(train_rxn_ids)} rxn_ids ({len(set(train_rxn_ids))} unique)")
        print(f"  Val: {len(val_rxn_ids)} rxn_ids ({len(set(val_rxn_ids))} unique)")
        print(f"  Test: {len(test_rxn_ids)} rxn_ids ({len(set(test_rxn_ids))} unique)")
        print(f"  Total: {len(split_0_rxn_ids)} rxn_ids ({len(set(split_0_rxn_ids))} unique)")
        
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse split_0.json: {e}")
        return None
    except Exception as e:
        print(f"Error: Could not extract rxn_ids from split_0.json: {e}")
        return None
    
    if not split_0_rxn_ids:
        print("Error: Could not extract rxn_ids from split_0.json")
        return None
    
    # Create dataframe with split_0 rxn_ids and their original indices
    # This represents the original dataset with duplicates (train + val + test)
    df = pd.DataFrame({
        'rxn_id': split_0_rxn_ids,
        'original_index': range(len(split_0_rxn_ids))
    })
    
    print(f"\nFull dataset shape: {df.shape}")
    print(f"Unique rxn_ids in dataset: {df['rxn_id'].nunique()}")
    
    # Load HF-Literature to get all possible unique rxn_ids
    desc_path = libra_path / "data" / "HF-Literature.tsv"
    if desc_path.exists():
        desc = pd.read_csv(desc_path, sep='\t')
        hf_unique_rxn_ids = sorted(desc['rxn_id'].unique())
        print(f"HF-Literature unique rxn_ids: {len(hf_unique_rxn_ids)}")
        
        # Check which rxn_ids from HF-Literature are in our dataset
        dataset_unique_rxn_ids = set(df['rxn_id'].unique())
        hf_in_dataset = [rxn_id for rxn_id in hf_unique_rxn_ids if rxn_id in dataset_unique_rxn_ids]
        hf_not_in_dataset = [rxn_id for rxn_id in hf_unique_rxn_ids if rxn_id not in dataset_unique_rxn_ids]
        
        print(f"HF-Literature rxn_ids in dataset: {len(hf_in_dataset)}")
        print(f"HF-Literature rxn_ids NOT in dataset: {len(hf_not_in_dataset)}")
        
        if hf_not_in_dataset:
            print(f"  Missing rxn_ids: {hf_not_in_dataset[:10]}...")
        
        # Use HF-Literature sorted unique rxn_ids as the reference
        # The unique embeddings (374) should correspond to a subset of these
        unique_rxn_ids = hf_unique_rxn_ids
    else:
        # Fallback: use dataset unique rxn_ids
        unique_rxn_ids = sorted(df['rxn_id'].unique())
    
    print(f"\nUsing {len(unique_rxn_ids)} unique rxn_ids for mapping")
    
    # Create mapping: unique embedding index -> original dataset index
    # Note: We have 374 unique embeddings, but may have more unique rxn_ids
    # We'll map based on what's available in the dataset
    unique_to_original_mapping = {}
    
    for unique_idx, rxn_id in enumerate(unique_rxn_ids):
        # Find first occurrence of this rxn_id in the full dataset
        matches = df[df['rxn_id'] == rxn_id]
        if len(matches) > 0:
            first_occurrence_idx = matches.index[0]
            unique_to_original_mapping[unique_idx] = int(first_occurrence_idx)
        else:
            # This rxn_id is not in the dataset (from val/test that we couldn't extract)
            # We'll skip it for now, but note that unique embeddings might have it
            pass
    
    print(f"\nCreated mapping for {len(unique_to_original_mapping)} unique embeddings")
    print(f"Unique index range: {min(unique_to_original_mapping.keys())} - {max(unique_to_original_mapping.keys())}")
    print(f"Original index range: {min(unique_to_original_mapping.values())} - {max(unique_to_original_mapping.values())}")
    
    # Show sample mappings
    print("\nSample mappings (unique index -> original index):")
    for i in range(min(10, len(unique_to_original_mapping))):
        print(f"  Unique {i} (rxn_id: {unique_rxn_ids[i]}) -> Original {unique_to_original_mapping[i]}")
    
    # Save the mapping
    mapping_file = script_dir / "unique_to_original_mapping.json"
    mapping_dict = {str(k): v for k, v in unique_to_original_mapping.items()}
    with open(mapping_file, 'w') as f:
        json.dump(mapping_dict, f, indent=2)
    
    print(f"\n✓ Unique to original mapping saved to: {mapping_file}")
    
    # Also create CSV
    csv_data = []
    for unique_idx, orig_idx in unique_to_original_mapping.items():
        csv_data.append({
            'unique_index': int(unique_idx),
            'original_index': orig_idx,
            'rxn_id': unique_rxn_ids[unique_idx]
        })
    
    df_mapping = pd.DataFrame(csv_data)
    csv_file = script_dir / "unique_to_original_mapping.csv"
    df_mapping.to_csv(csv_file, index=False)
    print(f"✓ Unique to original mapping CSV saved to: {csv_file}")
    
    return unique_to_original_mapping

if __name__ == "__main__":
    mapping = create_unique_to_original_mapping()
    if mapping:
        print("\n" + "="*60)
        print("✅ Mapping created successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Failed to create mapping")
        print("="*60)

