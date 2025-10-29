#!/usr/bin/env python3
"""
Test script to demonstrate ZCA whitening functionality
"""

import os
import sys
from pathlib import Path

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from gnn_gmm_analysis import run_analysis_with_pre_saved_embeddings

def test_zca_whitening():
    """Test ZCA whitening with different thresholds"""
    
    # Example embeddings directory
    embeddings_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings"
    
    print("Testing ZCA whitening functionality...")
    print(f"Embeddings directory: {embeddings_dir}")
    
    # Test with different ZCA thresholds
    zca_thresholds = [1e-6, 1e-5, 1e-4, 1e-3]
    
    for threshold in zca_thresholds:
        print(f"\n{'='*60}")
        print(f"Testing with ZCA threshold: {threshold}")
        print(f"{'='*60}")
        
        try:
            analyzer = run_analysis_with_pre_saved_embeddings(
                embeddings_dir=embeddings_dir,
                experiment_name=f"zca_test_threshold_{threshold}",
                max_components=10,  # Reduced for faster testing
                zca_threshold=threshold
            )
            
            print(f"\nAnalysis completed successfully for threshold {threshold}!")
            print(f"Results saved in: {analyzer.output_dir}")
            
            # Print summary
            print(f"\nSummary for threshold {threshold}:")
            print(f"- Number of layers analyzed: {len(analyzer.layer_embeddings)}")
            print(f"- Optimal components: {analyzer.optimal_components}")
            
            # Print embedding shapes after ZCA whitening
            print(f"\nEmbedding shapes after ZCA whitening:")
            for layer_name, gmm_result in analyzer.gmm_results.items():
                original_shape = gmm_result['embeddings_original'].shape
                whitened_shape = gmm_result['embeddings_whitened'].shape
                print(f"  {layer_name}: {original_shape} -> {whitened_shape}")
            
        except Exception as e:
            print(f"Error during analysis with threshold {threshold}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("ZCA whitening test completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_zca_whitening()
