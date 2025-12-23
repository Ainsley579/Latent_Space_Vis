#!/usr/bin/env python3
"""
Unified Entry Point for Embedding Processing Pipeline

This script provides a general interface to process ANY embeddings and generate
the two required output files:
1. scatter_plot_data.csv: id, x, y, color_value, layer
2. sankey_diagram_data.csv: real_id, from_layer, to_layer, from_cluster, to_cluster

Usage:
    # From files:
    python run_embedding_pipeline.py --embeddings_dir ./embeddings --layer_files emb.pt,output0.pt,output1.pt --layer_names emb,output0,output1
    
    # With ID mapping:
    python run_embedding_pipeline.py --embeddings_dir ./embeddings --id_mapping mapping.json
    
    # With training/test data:
    python run_embedding_pipeline.py --embeddings_dir ./embeddings --train_data train.json --test_data test.json
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from embedding_pipeline import EmbeddingPipeline
from embedding_pipeline_core import load_embeddings_from_files


def load_embeddings_from_args(args):
    """Load embeddings based on command-line arguments."""
    embeddings_dir = Path(args.embeddings_dir)
    
    if args.layer_files and args.layer_names:
        # Load from specified files
        layer_files = dict(zip(args.layer_names.split(','), args.layer_files.split(',')))
        embeddings = load_embeddings_from_files(embeddings_dir, layer_files)
    elif args.embeddings_dict:
        # Load from a dictionary file (JSON)
        with open(args.embeddings_dict, 'r') as f:
            embeddings_data = json.load(f)
        # Convert to numpy arrays
        embeddings = {}
        for layer_name, data in embeddings_data.items():
            embeddings[layer_name] = np.array(data)
    else:
        # Try to auto-detect common patterns
        print("Auto-detecting embedding files...")
        common_patterns = {
            'emb': ['emb.pt', 'emb_unique.pt', 'emb_unique_cas.pt'],
            'output_block_0': ['output0.pt', 'output0_unique.pt', 'output0_unique_cas.pt'],
            'output_block_1': ['output1.pt', 'output1_unique.pt', 'output1_unique_cas.pt'],
            'output_block_2': ['output2.pt', 'output2_unique.pt', 'output2_unique_cas.pt'],
            'output_block_3': ['output3.pt', 'output3_unique.pt', 'output3_unique_cas.pt'],
            'output_block_4': ['output4.pt', 'output4_unique.pt', 'output4_unique_cas.pt'],
        }
        
        embeddings = {}
        for layer_name, patterns in common_patterns.items():
            for pattern in patterns:
                filepath = embeddings_dir / pattern
                if filepath.exists():
                    try:
                        data = torch.load(filepath, map_location='cpu')
                        if isinstance(data, torch.Tensor):
                            data = data.squeeze().numpy()
                        else:
                            data = np.array(data)
                        if data.ndim > 2:
                            data = data.reshape(data.shape[0], -1)
                        embeddings[layer_name] = data
                        print(f"  ✓ Loaded {pattern} as {layer_name}")
                        break
                    except Exception as e:
                        print(f"  ✗ Error loading {pattern}: {e}")
        
        if not embeddings:
            raise ValueError("Could not auto-detect embedding files. Please specify --layer_files and --layer_names")
    
    return embeddings


def load_id_mapping(mapping_path):
    """Load ID mapping from JSON file."""
    if mapping_path is None:
        return None
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Convert string keys to integers
    return {int(k): int(v) for k, v in mapping.items()}


def load_train_test_data(train_path=None, test_path=None):
    """Load training and test data (optional)."""
    metadata = {}
    
    if train_path:
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        metadata['train_data_path'] = str(train_path)
        metadata['train_data_size'] = len(train_data) if isinstance(train_data, list) else 0
    
    if test_path:
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        metadata['test_data_path'] = str(test_path)
        metadata['test_data_size'] = len(test_data) if isinstance(test_data, list) else 0
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="General Embedding Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto-detection:
  python run_embedding_pipeline.py --embeddings_dir ./embeddings --output_dir ./output
  
  # Specify layer files explicitly:
  python run_embedding_pipeline.py --embeddings_dir ./embeddings \\
      --layer_files emb.pt,output0.pt,output1.pt \\
      --layer_names emb,output0,output1 \\
      --output_dir ./output
  
  # With ID mapping:
  python run_embedding_pipeline.py --embeddings_dir ./embeddings \\
      --id_mapping mapping.json \\
      --output_dir ./output
  
  # With training/test data:
  python run_embedding_pipeline.py --embeddings_dir ./embeddings \\
      --train_data train.json --test_data test.json \\
      --output_dir ./output
        """
    )
    
    # Required arguments
    parser.add_argument('--embeddings_dir', type=str, required=True,
                       help='Directory containing embedding files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for generated files')
    
    # Embedding specification (optional - will auto-detect if not provided)
    parser.add_argument('--layer_files', type=str, default=None,
                       help='Comma-separated list of embedding filenames (e.g., emb.pt,output0.pt)')
    parser.add_argument('--layer_names', type=str, default=None,
                       help='Comma-separated list of layer names (e.g., emb,output0)')
    parser.add_argument('--embeddings_dict', type=str, default=None,
                       help='Path to JSON file containing embeddings as a dictionary')
    
    # ID mapping
    parser.add_argument('--id_mapping', type=str, default=None,
                       help='Path to JSON file mapping embedding indices to real IDs')
    
    # GMM parameters
    parser.add_argument('--n_components', type=int, default=None,
                       help='Number of GMM components (auto-detect if not specified)')
    parser.add_argument('--zca_threshold', type=float, default=1e-5,
                       help='Threshold for ZCA whitening (default: 1e-5)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    # Layer order
    parser.add_argument('--layer_order', type=str, default=None,
                       help='Comma-separated list specifying layer order (default: sorted)')
    
    # Color values
    parser.add_argument('--color_values', type=str, default=None,
                       help='Path to numpy file (.npy) containing color values for scatter plots')
    
    # Training/test data (optional metadata)
    parser.add_argument('--train_data', type=str, default=None,
                       help='Path to training data file (JSON) - saved as metadata')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data file (JSON) - saved as metadata')
    
    # Additional metadata
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to JSON file with additional metadata')
    
    args = parser.parse_args()
    
    # Load embeddings
    print("Loading embeddings...")
    embeddings = load_embeddings_from_args(args)
    print(f"✓ Loaded {len(embeddings)} layers")
    
    # Load ID mapping
    id_mapping = load_id_mapping(args.id_mapping)
    if id_mapping:
        print(f"✓ Loaded ID mapping: {len(id_mapping)} entries")
    
    # Load color values
    color_values = None
    if args.color_values:
        color_values = np.load(args.color_values)
        print(f"✓ Loaded color values: shape {color_values.shape}")
    
    # Load training/test data
    train_test_metadata = load_train_test_data(args.train_data, args.test_data)
    
    # Load additional metadata
    additional_metadata = {}
    if args.metadata:
        with open(args.metadata, 'r') as f:
            additional_metadata = json.load(f)
    
    # Combine metadata
    metadata = {**train_test_metadata, **additional_metadata}
    
    # Set layer order
    layer_order = None
    if args.layer_order:
        layer_order = args.layer_order.split(',')
    
    # Create pipeline
    pipeline = EmbeddingPipeline(
        embeddings=embeddings,
        output_dir=args.output_dir,
        id_mapping=id_mapping,
        n_components=args.n_components,
        zca_threshold=args.zca_threshold,
        random_state=args.random_state,
        color_values=color_values,
        layer_order=layer_order,
        metadata=metadata
    )
    
    # Run pipeline
    pipeline.run()


if __name__ == "__main__":
    main()

