#!/usr/bin/env python3
"""
Interactive script for running GMM analysis with pre-saved embeddings
Allows easy modification of paths and parameters
"""

import os
import sys
from pathlib import Path

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from gnn_gmm_analysis import run_analysis_with_pre_saved_embeddings

def get_user_input():
    """Get user input for paths and parameters"""
    
    print("=" * 60)
    print("GNN GMM Analysis with Pre-saved Embeddings")
    print("=" * 60)
    
    # Default paths
    default_embeddings = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings"
    default_output = None  # Will be auto-generated
    
    # Get embeddings directory
    print(f"\n1. Embeddings Directory:")
    print(f"   Default: {default_embeddings}")
    embeddings_dir = input("   Enter embeddings directory (or press Enter for default): ").strip()
    if not embeddings_dir:
        embeddings_dir = default_embeddings
    
    # Check if directory exists
    if not Path(embeddings_dir).exists():
        print(f"   WARNING: Directory {embeddings_dir} does not exist!")
        confirm = input("   Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("   Exiting...")
            return None
    
    # Get output directory
    print(f"\n2. Output Directory:")
    print(f"   Default: Auto-generated with timestamp")
    output_dir = input("   Enter output directory (or press Enter for auto-generated): ").strip()
    if not output_dir:
        output_dir = None
    
    # Get experiment name
    print(f"\n3. Experiment Name:")
    experiment_name = input("   Enter experiment name (or press Enter for default): ").strip()
    if not experiment_name:
        experiment_name = None
    
    # Get max components
    print(f"\n4. Maximum Components:")
    print(f"   Default: 20")
    max_components_input = input("   Enter max components (or press Enter for default): ").strip()
    try:
        max_components = int(max_components_input) if max_components_input else 20
    except ValueError:
        print("   Invalid input, using default: 20")
        max_components = 20
    
    # Get sample fraction
    print(f"\n5. Sample Fraction:")
    print(f"   Default: 1.0 (use all data)")
    sample_frac_input = input("   Enter sample fraction (or press Enter for default): ").strip()
    try:
        sample_frac = float(sample_frac_input) if sample_frac_input else 1.0
    except ValueError:
        print("   Invalid input, using default: 1.0")
        sample_frac = 1.0
    
    return {
        'embeddings_dir': embeddings_dir,
        'output_dir': output_dir,
        'experiment_name': experiment_name,
        'max_components': max_components,
        'sample_frac': sample_frac
    }

def main():
    """Main interactive function"""
    
    # Get user input
    params = get_user_input()
    if params is None:
        return
    
    # Display configuration
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY:")
    print("=" * 60)
    print(f"Embeddings Directory: {params['embeddings_dir']}")
    print(f"Output Directory: {params['output_dir'] or 'Auto-generated'}")
    print(f"Experiment Name: {params['experiment_name'] or 'Default'}")
    print(f"Max Components: {params['max_components']}")
    print(f"Sample Fraction: {params['sample_frac']}")
    
    # Confirm before running
    print("\n" + "=" * 60)
    confirm = input("Run analysis with these settings? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Analysis cancelled.")
        return
    
    try:
        # Run the analysis
        print("\nStarting analysis...")
        analyzer = run_analysis_with_pre_saved_embeddings(
            embeddings_dir=params['embeddings_dir'],
            output_dir=params['output_dir'],
            experiment_name=params['experiment_name'],
            max_components=params['max_components'],
            sample_frac=params['sample_frac']
        )
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved in: {analyzer.output_dir}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Number of layers analyzed: {len(analyzer.layer_embeddings)}")
        print(f"- Optimal components: {analyzer.optimal_components}")
        print(f"- Layer names: {list(analyzer.layer_embeddings.keys())}")
        
        # Print embedding shapes
        print(f"\nEmbedding shapes:")
        for layer_name, embeddings in analyzer.layer_embeddings.items():
            print(f"  {layer_name}: {embeddings.shape}")
            
    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
