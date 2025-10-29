#!/usr/bin/env python3
"""
Create Index Mapping Script

This script creates a mapping from unique embedding indices to original SMILES dataframe indices.
The embeddings represent unique reactions (152 unique rxn_ids), while the SMILES dataframe 
contains all conformers/variants (3,518 total rows).

This mapping will be used to correctly match GMM results with molecules in the notebook.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def create_index_mapping():
    """Create mapping from unique embedding indices to original SMILES indices"""
    
    print("Creating index mapping from unique embeddings to original SMILES dataframe...")
    
    # Import model setup
    try:
        from model_setup import dimenet10conf
    except ImportError:
        # If model_setup is not available, try alternative approach
        print("Warning: model_setup not available, using alternative approach")
        return create_index_mapping_alternative()
    
    # Load model and data
    model, data, smiles = dimenet10conf()
    
    print(f"SMILES dataframe shape: {smiles.shape}")
    print(f"Unique rxn_ids in SMILES: {smiles['rxn_id'].nunique()}")
    
    # Create mapping: for each unique rxn_id, get the first occurrence index
    unique_rxn_ids = smiles['rxn_id'].unique()
    print(f"Number of unique rxn_ids: {len(unique_rxn_ids)}")
    
    # Create mapping from unique rxn_id to first occurrence index in SMILES dataframe
    rxn_id_to_first_index = {}
    for rxn_id in unique_rxn_ids:
        first_index = smiles[smiles['rxn_id'] == rxn_id].index[0]
        rxn_id_to_first_index[rxn_id] = first_index
    
    # Sort unique rxn_ids to create consistent ordering
    sorted_unique_rxn_ids = sorted(unique_rxn_ids)
    
    # Create the mapping: embedding_index -> original_smiles_index
    embedding_to_original_mapping = {}
    for embedding_idx, rxn_id in enumerate(sorted_unique_rxn_ids):
        original_idx = rxn_id_to_first_index[rxn_id]
        embedding_to_original_mapping[embedding_idx] = original_idx
    
    print(f"Created mapping for {len(embedding_to_original_mapping)} unique reactions")
    print(f"Embedding indices range: 0 to {max(embedding_to_original_mapping.keys())}")
    print(f"Original indices range: {min(embedding_to_original_mapping.values())} to {max(embedding_to_original_mapping.values())}")
    
    # Save the mapping
    mapping_df = pd.DataFrame([
        {'embedding_index': emb_idx, 'original_index': orig_idx, 'rxn_id': sorted_unique_rxn_ids[emb_idx]}
        for emb_idx, orig_idx in embedding_to_original_mapping.items()
    ])
    
    output_path = Path("index_mapping.csv")
    mapping_df.to_csv(output_path, index=False)
    print(f"Index mapping saved to: {output_path}")
    
    # Also save as a simple dictionary for easy loading
    import json
    mapping_dict = {str(k): v for k, v in embedding_to_original_mapping.items()}
    with open("index_mapping.json", "w") as f:
        json.dump(mapping_dict, f, indent=2)
    print("Index mapping dictionary saved to: index_mapping.json")
    
    return embedding_to_original_mapping

def create_index_mapping_alternative():
    """Alternative approach if model_setup is not available"""
    print("Using alternative approach to create index mapping...")
    
    # Try to load the SMILES data directly
    try:
        # Find the project root directory
        current_file = Path(__file__).resolve()
        dd = current_file.parent.parent.parent / "Libra-ML-Modeling"
        
        desc = pd.read_csv(f"{dd}/data/HF-Literature.tsv", sep="\t")
        jp = f"{dd}/model-training/butene/dimenet-10conf/data/jsons/split_0.json"
        
        import json
        with open(jp) as f:
            data = json.load(f)
        train = pd.DataFrame(data["train"])
        val = pd.DataFrame(data["val"])
        test = pd.DataFrame(data["test"])
        
        df = pd.concat([train, test, val])
        df = df.merge(desc, left_on="rxn_id", right_on="rxn_id", how="left")
        
        print(f"Loaded SMILES data shape: {df.shape}")
        print(f"Unique rxn_ids: {df['rxn_id'].nunique()}")
        
        # Create mapping
        unique_rxn_ids = df['rxn_id'].unique()
        sorted_unique_rxn_ids = sorted(unique_rxn_ids)
        
        embedding_to_original_mapping = {}
        for embedding_idx, rxn_id in enumerate(sorted_unique_rxn_ids):
            original_idx = df[df['rxn_id'] == rxn_id].index[0]
            embedding_to_original_mapping[embedding_idx] = original_idx
        
        print(f"Created mapping for {len(embedding_to_original_mapping)} unique reactions")
        
        # Save the mapping
        mapping_df = pd.DataFrame([
            {'embedding_index': emb_idx, 'original_index': orig_idx, 'rxn_id': sorted_unique_rxn_ids[emb_idx]}
            for emb_idx, orig_idx in embedding_to_original_mapping.items()
        ])
        
        output_path = Path("index_mapping.csv")
        mapping_df.to_csv(output_path, index=False)
        print(f"Index mapping saved to: {output_path}")
        
        return embedding_to_original_mapping
        
    except Exception as e:
        print(f"Error in alternative approach: {e}")
        return None

if __name__ == "__main__":
    mapping = create_index_mapping()
    if mapping:
        print("\nMapping created successfully!")
        print("Sample mappings:")
        for i in range(min(5, len(mapping))):
            print(f"  Embedding index {i} -> Original index {mapping[i]}")
    else:
        print("Failed to create mapping")
