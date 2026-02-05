#!/usr/bin/env python3
"""
Unified entry point for generating all frontend data from embeddings.

Usage:
    # Generate from single embedding directory
    python generate_frontend_data.py \
        --embeddings-dir /path/to/embeddings \
        --model-name dimenet10conf \
        --output-dir frontend/public/data

    # Or with config file
    python generate_frontend_data.py \
        --config config.yaml \
        --output-dir frontend/public/data
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Import backend modules
from analyzer import (
    EmbeddingAnalyzer,
    DimensionalityReducer,
    SankeyFlowAnalyzer,
    ClusterRepresentativesExtractor
)
from generator import (
    ScatterGenerator,
    SankeyGenerator,
    ClusterRepresentativesGenerator
)
from utils import (
    EmbeddingLoader,
    DataValidator,
    ConfigManager
)


class FrontendDataPipeline(object):
    """Main pipeline for generating frontend data."""
    
    def __init__(self, model_name, model_type, output_dir=None, config=None):
        """
        Args:
            model_name: display name (e.g., dimenet10conf)
            model_type: type for clustering (same as model_name for now)
            output_dir: output directory for JSON files
            config: configuration dict
        """
        self.model_name = model_name
        self.model_type = model_type
        self.output_dir = Path(output_dir) if output_dir else Path('frontend/public/data')
        self.config = ConfigManager.merge_with_defaults(config or {})
        
        # Initialize components
        self.analyzer = EmbeddingAnalyzer(random_state=self.config['random_state'])
        self.reducer = DimensionalityReducer(
            method=self.config['reduction_method'],
            random_state=self.config['random_state']
        )
        self.scatter_gen = ScatterGenerator()
        self.sankey_gen = SankeyGenerator()
        self.reps_gen = ClusterRepresentativesGenerator()
        
        # Results storage
        self.all_scatter_rows = []
        self.all_sankey_transitions = []
        self.all_representatives = []
        
        print(f"\n[Pipeline] Initialized for model: {model_name}")
        print(f"[Pipeline] Output directory: {self.output_dir}")
    
    def process_layer(self, layer_name, embeddings):
        """
        Process a single layer: cluster, reduce, extract reps, compute transitions.
        
        Args:
            layer_name: layer name (emb, output_block_0, etc.)
            embeddings: numpy array of shape (n_samples, n_features)
            
        Returns:
            dict with processed results
        """
        print(f"\n[{layer_name}] Processing {embeddings.shape[0]} samples, {embeddings.shape[1]} dims...")
        
        # 1. Fit GMM and get clusters
        print(f"[{layer_name}] Fitting GMM...")
        gmm_result = self.analyzer.fit_gmm(
            embeddings,
            n_components_range=self.config['n_components_range']
        )
        optimal_n = gmm_result['optimal_n_components']
        optimal_gmm = gmm_result['optimal_gmm']
        
        print(f"[{layer_name}] Optimal components: {optimal_n}")
        
        # Get cluster assignments
        cluster_result = self.analyzer.get_cluster_assignments(embeddings, optimal_gmm)
        labels = cluster_result['labels']
        
        # 2. Reduce to 2D
        print(f"[{layer_name}] Reducing to 2D using {self.reducer.method}...")
        coords_2d = self.reducer.fit_reduce(embeddings)
        
        # 3. Generate scatter CSV rows
        print(f"[{layer_name}] Generating scatter data...")
        scatter_df = self.scatter_gen.generate_csv(
            embeddings, layer_name, self.model_name, labels, coords_2d
        )
        self.all_scatter_rows.append(scatter_df)
        
        # 4. Extract cluster representatives
        print(f"[{layer_name}] Extracting cluster representatives...")
        reps = ClusterRepresentativesExtractor.get_top_k(
            embeddings, labels, k=self.config['cluster_k']
        )
        reps_json = self.reps_gen.generate_json(reps, layer_name, self.model_name)
        self.all_representatives.extend(reps_json)
        
        return {
            'layer_name': layer_name,
            'optimal_n_components': optimal_n,
            'labels': labels,
            'coords_2d': coords_2d,
            'gmm': optimal_gmm,
            'scatter_df': scatter_df,
            'representatives': reps
        }
    
    def process_transitions(self, layer_results_list):
        """
        Compute transitions between consecutive layers.
        
        Args:
            layer_results_list: list of results from process_layer in order
        """
        print(f"\n[Transitions] Computing layer-to-layer transitions...")
        
        for i in range(len(layer_results_list) - 1):
            curr = layer_results_list[i]
            next_ = layer_results_list[i + 1]
            
            curr_layer = curr['layer_name']
            next_layer = next_['layer_name']
            
            print(f"[Transitions] {curr_layer} ({curr['optimal_n_components']}) -> {next_layer} ({next_['optimal_n_components']})")
            
            # Compute transitions
            transitions = SankeyFlowAnalyzer.compute_transitions(
                curr['labels'], next_['labels']
            )
            
            # Add component info
            for t in transitions:
                t['from_components'] = curr['optimal_n_components']
                t['to_components'] = next_['optimal_n_components']
            
            self.all_sankey_transitions.extend(transitions)
    
    def save_scatter_plots(self):
        """Save scatter plot JSON files."""
        print(f"\n[Output] Saving scatter plot JSONs...")
        
        output_dir = self.output_dir / 'scatterplot'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all rows
        combined_df = pd.concat(self.all_scatter_rows, ignore_index=True)
        
        # Group by layer and save JSON
        for layer_name in combined_df['layer_name'].unique():
            layer_data = combined_df[combined_df['layer_name'] == layer_name]
            
            json_file = output_dir / f"{self.model_name}_{layer_name}.json"
            json_data = []
            for _, row in layer_data.iterrows():
                json_data.append({
                    'real_id': str(int(row['real_id'])),
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'cluster_id': int(row['cluster_id'])
                })
            
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"  ✓ {json_file.name} ({len(json_data)} points)")
    
    def save_representatives(self):
        """Save cluster representatives JSON."""
        print(f"\n[Output] Saving cluster representatives...")
        
        output_dir = self.output_dir / 'cluster_representatives'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'cluster_representatives.json'
        
        # Save all representatives
        with open(output_file, 'w') as f:
            json.dump(self.all_representatives, f, indent=2)
        
        print(f"  ✓ {output_file.name} ({len(self.all_representatives)} clusters)")
    
    def run(self, embeddings_dict):
        """
        Run the full pipeline.
        
        Args:
            embeddings_dict: dict mapping layer_name -> embeddings array
        """
        # Validate
        valid, msg = DataValidator.validate_embeddings(embeddings_dict)
        if not valid:
            print(f"[ERROR] Invalid embeddings: {msg}")
            return False
        
        print(f"[Validation] {msg}")
        
        # Process each layer
        layer_results = []
        for layer_name in DataValidator.LAYERS:
            if layer_name not in embeddings_dict:
                print(f"[Warning] Skipping {layer_name}: not in embeddings")
                continue
            
            result = self.process_layer(layer_name, embeddings_dict[layer_name])
            layer_results.append(result)
        
        # Compute transitions
        self.process_transitions(layer_results)
        
        # Save outputs
        self.save_scatter_plots()
        self.save_representatives()
        
        print(f"\n[Pipeline] ✅ Complete!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate frontend data from embeddings'
    )
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        help='Directory containing embedding .npy files'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='dimenet10conf',
        help='Model name (directory naming)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        help='Model type for analysis (defaults to model-name)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='frontend/public/data',
        help='Output directory for JSON files'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Config file (JSON or YAML)'
    )
    parser.add_argument(
        '--reduction-method',
        type=str,
        choices=['pca', 'umap'],
        default='pca',
        help='Dimensionality reduction method'
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    user_config = {}
    if args.config:
        try:
            user_config = ConfigManager.load_config(args.config)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    
    # Override with command-line args
    if args.reduction_method:
        user_config['reduction_method'] = args.reduction_method
    
    model_type = args.model_type or args.model_name
    
    # Load embeddings
    if not args.embeddings_dir:
        print("Error: --embeddings-dir required")
        parser.print_help()
        sys.exit(1)
    
    embeddings_dir = Path(args.embeddings_dir)
    embeddings_dict = {}
    
    # Load all .npy files from directory
    for npy_file in sorted(embeddings_dir.glob('*.npy')):
        layer_name = npy_file.stem
        try:
            embeddings_dict[layer_name] = EmbeddingLoader.load_numpy(npy_file)
            print(f"Loaded {layer_name}: {embeddings_dict[layer_name].shape}")
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
    
    if not embeddings_dict:
        print(f"No embeddings found in {embeddings_dir}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = FrontendDataPipeline(
        model_name=args.model_name,
        model_type=model_type,
        output_dir=args.output_dir,
        config=user_config
    )
    
    success = pipeline.run(embeddings_dict)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
