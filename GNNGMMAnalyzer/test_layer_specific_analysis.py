#!/usr/bin/env python3
"""
Test script for running GMM analysis with layer-specific components
"""

import os
import sys
from pathlib import Path

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from gnn_gmm_analysis_layer_specific import run_analysis_with_pre_saved_embeddings_layer_specific

def main():
    """Run GMM analysis with layer-specific components"""
    
    # Define paths
    embeddings_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings"
    output_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzerResults/dimenet10conf_checkpoint0_layer_specific_cas_zca_20250113_170000"
    experiment_name = "dimenet10conf_checkpoint0_layer_specific_cas_zca"
    
    print("Running GMM analysis with layer-specific components...")
    print(f"Embeddings directory: {embeddings_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Experiment name: {experiment_name}")
    
    try:
        # Run the analysis with layer-specific components
        analyzer = run_analysis_with_pre_saved_embeddings_layer_specific(
            embeddings_dir=embeddings_dir,
            output_dir=output_dir,
            experiment_name=experiment_name,
            max_components=30,
            zca_threshold=1e-5
        )
        
        print("\nLayer-specific analysis completed successfully!")
        print(f"Results saved in: {analyzer.output_dir}")
        
        # Print component counts for each layer
        print(f"\nComponent counts per layer:")
        for layer_name, n_comp in analyzer.layer_optimal_components.items():
            print(f"  {layer_name}: {n_comp} components")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
