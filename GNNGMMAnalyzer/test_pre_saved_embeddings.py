#!/usr/bin/env python3
"""
Flexible script for running GMM analysis with pre-saved embeddings
Supports multiple ways to modify paths at runtime
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from gnn_gmm_analysis import run_analysis_with_pre_saved_embeddings

def main():
    """Run GMM analysis with configurable paths"""
    
    parser = argparse.ArgumentParser(description="Run GMM analysis with pre-saved embeddings")
    parser.add_argument('--embeddings_dir', type=str, 
                       default="/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings",
                       help='Directory containing pre-saved embeddings (.pt files)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (auto-generated if not specified)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom name for this experiment')
    parser.add_argument('--max_components', type=int, default=30,
                       help='Maximum number of components to test for BIC/AIC')
    parser.add_argument('--model_type', type=str, default='dimenet-10conf',
                       help='Model type (for naming purposes)')
    parser.add_argument('--checkpoint_index', type=int, default=0,
                       help='Checkpoint index (for naming purposes)')
    parser.add_argument('--sample_frac', type=float, default=1.0,
                       help='Fraction of data to use (1.0 = use all, automatically set to 1.0 for unique files)')
    parser.add_argument('--zca_threshold', type=float, default=1e-5,
                       help='Threshold for ZCA whitening (default: 1e-5)')
    
    args = parser.parse_args()
    
    # Check if unique files are being used and set sample_frac to 1.0
    import os
    from pathlib import Path
    embeddings_path = Path(args.embeddings_dir)
    if embeddings_path.exists():
        # Check if unique files exist (including _cas variants)
        unique_files = ['emb_unique_cas.pt', 'emb_unique.pt', 'output0_unique_cas.pt', 'output0_unique.pt', 
                       'output1_unique_cas.pt', 'output1_unique.pt', 'output2_unique_cas.pt', 'output2_unique.pt',
                       'output3_unique_cas.pt', 'output3_unique.pt', 'output4_unique_cas.pt', 'output4_unique.pt']
        unique_files_exist = any((embeddings_path / f).exists() for f in unique_files)
        
        if unique_files_exist and args.sample_frac < 1.0:
            print(f"Note: Unique files detected, setting sample_frac to 1.0 (was {args.sample_frac})")
            args.sample_frac = 1.0
    
    print("Running GMM analysis with pre-saved embeddings...")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Output directory: {args.output_dir or 'Auto-generated'}")
    print(f"Experiment name: {args.experiment_name or 'Default'}")
    print(f"Max components: {args.max_components}")
    print(f"Sample fraction: {args.sample_frac}")
    print(f"ZCA threshold: {args.zca_threshold}")
    
    try:
        # Run the analysis with flexible parameters
        analyzer = run_analysis_with_pre_saved_embeddings(
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            max_components=args.max_components,
            model_type=args.model_type,
            checkpoint_index=args.checkpoint_index,
            sample_frac=args.sample_frac,
            experiment_name=args.experiment_name,
            zca_threshold=args.zca_threshold
        )
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved in: {analyzer.output_dir}")
        
        # Print some summary information
        print(f"\nSummary:")
        print(f"- Number of layers analyzed: {len(analyzer.layer_embeddings)}")
        print(f"- Optimal components: {analyzer.optimal_components}")
        print(f"- Layer names: {list(analyzer.layer_embeddings.keys())}")
        
        # Print embedding shapes
        print(f"\nEmbedding shapes:")
        for layer_name, embeddings in analyzer.layer_embeddings.items():
            print(f"  {layer_name}: {embeddings.shape}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def run_with_custom_paths():
    """Example function showing how to modify paths programmatically"""
    
    # Example 1: Different embeddings directory
    embeddings_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings"
    
    # Example 2: Custom output directory
    output_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzerResults/custom_experiment"
    
    # Example 3: Custom experiment name
    experiment_name = "my_custom_analysis"
    
    print("Running with custom paths...")
    
    analyzer = run_analysis_with_pre_saved_embeddings(
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        max_components=15
    )
    
    return analyzer

if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Run with default settings
        print("No command line arguments provided. Running with default settings...")
        print("Use --help to see available options.")
        main()
