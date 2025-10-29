#!/usr/bin/env python3
"""
Test script for layer-specific component analysis
This script runs GMM analysis where each layer can have different optimal number of components
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from gnn_gmm_analysis import GNNGMMAnalyzer

def run_layer_specific_analysis(embeddings_dir=None, max_components=20, 
                               experiment_name="layer_specific_components"):
    """Run layer-specific component analysis"""
    
    # Default embeddings directory
    if embeddings_dir is None:
        embeddings_dir = current_dir.parent / "dimenet10conf_embeddings"
    
    print(f"Running layer-specific component analysis...")
    print(f"Embeddings directory: {embeddings_dir}")
    print(f"Max components: {max_components}")
    print(f"Experiment name: {experiment_name}")
    
    # Create analyzer
    analyzer = GNNGMMAnalyzer(
        model_type="dimenet-10conf",
        checkpoint_index=0,
        output_dir=f"../GNNGMMAnalyzerResults/{experiment_name}",
        embeddings_dir=embeddings_dir
    )
    
    # Load embeddings
    print("Loading embeddings...")
    analyzer.load_pre_saved_embeddings()
    
    # Run layer-specific analysis
    print("Starting layer-specific component analysis...")
    layer_gmm_results, layer_optimal_components = analyzer.run_layer_specific_component_analysis(
        max_components=max_components
    )
    
    print("\n" + "="*60)
    print("LAYER-SPECIFIC COMPONENT ANALYSIS RESULTS")
    print("="*60)
    
    print("\nOptimal component counts per layer:")
    for layer_name, n_components in layer_optimal_components.items():
        print(f"  {layer_name}: {n_components} components")
    
    print(f"\nAnalysis completed successfully!")
    print(f"Results saved in: {analyzer.output_dir}")
    
    return analyzer, layer_gmm_results, layer_optimal_components

def main():
    """Main function to run layer-specific component analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run layer-specific GMM component analysis')
    parser.add_argument('--embeddings_dir', type=str, default=None,
                       help='Directory containing embedding files')
    parser.add_argument('--max_components', type=int, default=20,
                       help='Maximum number of components to test for each layer')
    parser.add_argument('--experiment_name', type=str, default="layer_specific_components",
                       help='Name for this experiment')
    
    args = parser.parse_args()
    
    try:
        analyzer, layer_gmm_results, layer_optimal_components = run_layer_specific_analysis(
            embeddings_dir=args.embeddings_dir,
            max_components=args.max_components,
            experiment_name=args.experiment_name
        )
        
        print("\nSummary:")
        print(f"- Number of layers analyzed: {len(layer_optimal_components)}")
        print(f"- Component count range: {min(layer_optimal_components.values())} - {max(layer_optimal_components.values())}")
        print(f"- Average components: {sum(layer_optimal_components.values()) / len(layer_optimal_components):.1f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
