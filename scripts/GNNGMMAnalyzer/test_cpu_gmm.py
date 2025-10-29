#!/usr/bin/env python3
"""
Test script for CPU-only GMM analysis
"""

import os
import sys
from pathlib import Path

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from cpu_gmm_analysis import run_cpu_gmm_analysis

def main():
    """Run CPU-only GMM analysis with default settings"""
    
    # Default embeddings directory
    embeddings_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings"
    
    print("Running CPU-only GMM analysis...")
    print(f"Embeddings directory: {embeddings_dir}")
    
    try:
        analyzer = run_cpu_gmm_analysis(
            embeddings_dir=embeddings_dir,
            experiment_name="cpu_test",
            max_components=15,
            sample_frac=0.1  # Use only 10% of data for testing
        )
        
        print("\n✅ Analysis completed successfully!")
        print(f"Results saved in: {analyzer.output_dir}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Number of layers: {len(analyzer.layer_embeddings)}")
        print(f"- Layer names: {list(analyzer.layer_embeddings.keys())}")
        
        for layer_name, embeddings in analyzer.layer_embeddings.items():
            print(f"- {layer_name}: {embeddings.shape}")
        
        print(f"\nOptimal components:")
        for layer_name, components in analyzer.optimal_components.items():
            print(f"- {layer_name}: BIC={components['bic']}, AIC={components['aic']}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
