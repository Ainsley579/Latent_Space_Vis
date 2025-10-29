#!/usr/bin/env python3
"""
Optimized Multi-Layer Embedding Extraction Script

This script addresses the bottleneck in save_embedding by extracting embeddings
from multiple layers in a single forward pass, significantly improving efficiency.

Usage:
    python optimized_embedding_extraction.py --model_type dimenet-10conf --checkpoint_index 0
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Force CPU usage to avoid CUDA issues
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Go up to latent-spaces directory
sys.path.append(parent_dir)

from model_setup import dimenet10conf, dimenet_model

class OptimizedEmbeddingExtractor:
    """Optimized embedding extraction for multiple layers"""
    
    def __init__(self, model_type="dimenet-10conf", checkpoint_index=0, 
                 output_dir="../GNNGMMAnalyzerResults/optimized_embeddings", sample_frac=1.0):
        self.model_type = model_type
        self.checkpoint_index = checkpoint_index
        self.output_dir = Path(output_dir)
        self.sample_frac = sample_frac
        self.model = None
        self.data = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model_and_data(self):
        """Load the GNN model and data"""
        print(f"Loading {self.model_type} model (checkpoint {self.checkpoint_index})...")
        
        if self.model_type == "dimenet-10conf":
            self.model, self.data, _ = dimenet10conf()
        else:
            self.model, self.data, _ = dimenet_model(self.model_type, self.checkpoint_index)
        
        print(f"Model loaded successfully. Data size: {len(self.data)} samples")
        
        # Sample data if needed
        if self.sample_frac < 1.0:
            n_samples = int(len(self.data) * self.sample_frac)
            indices = np.random.choice(len(self.data), n_samples, replace=False)
            self.data = [self.data[i] for i in indices]
            print(f"Sampled {n_samples} samples from {len(self.data)} total")
    
    def extract_all_layers_single_pass(self):
        """Extract embeddings from all layers in a single forward pass"""
        print("Extracting embeddings from all layers in single forward pass...")
        
        # Define layers to extract
        layers = [self.model.gnn.emb] + list(self.model.gnn.output_blocks)
        layer_names = ["emb"] + [f"output_block_{i}" for i in range(len(self.model.gnn.output_blocks))]
        
        # Initialize storage for all layers
        all_embeddings = {name: [] for name in layer_names}
        
        # Process each batch
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data, desc="Processing batches")):
                # Forward pass through all layers
                x = batch.x if hasattr(batch, 'x') else batch
                
                # Initial embedding layer
                x_emb = self.model.gnn.emb(x)
                all_embeddings["emb"].append(x_emb.sum(dim=0))
                
                # Pass through output blocks
                x_current = x_emb
                for i, output_block in enumerate(self.model.gnn.output_blocks):
                    x_current = output_block(x_current)
                    all_embeddings[f"output_block_{i}"].append(x_current.sum(dim=0))
        
        # Convert to tensors and save
        print("Saving embeddings...")
        for layer_name, embeddings_list in all_embeddings.items():
            # Stack embeddings
            embeddings_tensor = torch.stack(embeddings_list)
            
            # Save as both .pt and .npy
            torch.save(embeddings_tensor, self.output_dir / f"{layer_name}.pt")
            np.save(self.output_dir / f"{layer_name}.npy", embeddings_tensor.detach().cpu().numpy())
            
            print(f"  {layer_name}: {embeddings_tensor.shape}")
        
        print(f"Extracted embeddings from {len(layers)} layers")
        return all_embeddings
    
    def extract_with_hooks(self):
        """Alternative method using PyTorch hooks for even more efficiency"""
        print("Extracting embeddings using PyTorch hooks...")
        
        # Define layers to extract
        layers = [self.model.gnn.emb] + list(self.model.gnn.output_blocks)
        layer_names = ["emb"] + [f"output_block_{i}" for i in range(len(self.model.gnn.output_blocks))]
        
        # Storage for embeddings
        embeddings_storage = {name: [] for name in layer_names}
        
        # Hook function to capture embeddings
        def create_hook(name):
            def hook(module, input, output):
                embeddings_storage[name].append(output.sum(dim=0))
            return hook
        
        # Register hooks
        hooks = []
        for layer, name in zip(layers, layer_names):
            hook = layer.register_forward_hook(create_hook(name))
            hooks.append(hook)
        
        try:
            # Forward pass through all data
            with torch.no_grad():
                for batch in tqdm(self.data, desc="Processing with hooks"):
                    # Forward pass
                    x = batch.x if hasattr(batch, 'x') else batch
                    _ = self.model.gnn(x)
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Convert to tensors and save
        print("Saving embeddings from hooks...")
        for layer_name, embeddings_list in embeddings_storage.items():
            if embeddings_list:  # Check if we captured anything
                embeddings_tensor = torch.stack(embeddings_list)
                
                # Save as both .pt and .npy
                torch.save(embeddings_tensor, self.output_dir / f"{layer_name}_hooks.pt")
                np.save(self.output_dir / f"{layer_name}_hooks.npy", embeddings_tensor.detach().cpu().numpy())
                
                print(f"  {layer_name}: {embeddings_tensor.shape}")
        
        return embeddings_storage
    
    def benchmark_extraction_methods(self):
        """Benchmark different extraction methods"""
        print("Benchmarking extraction methods...")
        
        methods = {
            "Single Pass": self.extract_all_layers_single_pass,
            "PyTorch Hooks": self.extract_with_hooks
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            print(f"\nBenchmarking {method_name}...")
            start_time = time.time()
            
            try:
                embeddings = method_func()
                end_time = time.time()
                
                results[method_name] = {
                    'time': end_time - start_time,
                    'success': True,
                    'layers': len(embeddings),
                    'total_samples': len(list(embeddings.values())[0]) if embeddings else 0
                }
                
                print(f"  {method_name}: {end_time - start_time:.2f} seconds")
                
            except Exception as e:
                print(f"  {method_name} failed: {e}")
                results[method_name] = {
                    'time': None,
                    'success': False,
                    'error': str(e)
                }
        
        # Print benchmark results
        print("\nBenchmark Results:")
        print("=" * 50)
        for method_name, result in results.items():
            if result['success']:
                print(f"{method_name}: {result['time']:.2f}s ({result['layers']} layers, {result['total_samples']} samples)")
            else:
                print(f"{method_name}: FAILED - {result['error']}")
        
        return results
    
    def run_extraction(self, method="single_pass"):
        """Run the embedding extraction"""
        print("Starting optimized embedding extraction...")
        start_time = time.time()
        
        # Load model and data
        self.load_model_and_data()
        
        # Run extraction based on method
        if method == "single_pass":
            embeddings = self.extract_all_layers_single_pass()
        elif method == "hooks":
            embeddings = self.extract_with_hooks()
        elif method == "benchmark":
            results = self.benchmark_extraction_methods()
            return results
        else:
            raise ValueError(f"Unknown method: {method}")
        
        total_time = time.time() - start_time
        print(f"\nExtraction completed in {total_time:.2f} seconds")
        print(f"Results saved in: {self.output_dir}")
        
        return embeddings


def main():
    parser = argparse.ArgumentParser(description="Optimized Multi-Layer Embedding Extraction")
    parser.add_argument('--model_type', type=str, default='dimenet-10conf',
                       choices=['dimenet-10conf', 'dimenet-100conf', 'dimenet-noaug'],
                       help='Type of DimeNet model to analyze')
    parser.add_argument('--checkpoint_index', type=int, default=0,
                       help='Index of checkpoint to load (0-based)')
    parser.add_argument('--output_dir', type=str, default='../GNNGMMAnalyzerResults/optimized_embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--sample_frac', type=float, default=1.0,
                       help='Fraction of data to sample (1.0 = use all)')
    parser.add_argument('--method', type=str, default='single_pass',
                       choices=['single_pass', 'hooks', 'benchmark'],
                       help='Extraction method to use')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = OptimizedEmbeddingExtractor(
        model_type=args.model_type,
        checkpoint_index=args.checkpoint_index,
        output_dir=args.output_dir,
        sample_frac=args.sample_frac
    )
    
    # Run extraction
    extractor.run_extraction(method=args.method)


if __name__ == "__main__":
    main()
