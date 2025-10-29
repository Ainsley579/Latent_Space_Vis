#!/usr/bin/env python3
"""
Test script to demonstrate using _cas.pt embedding files
"""

import os
import sys
from pathlib import Path

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from gnn_gmm_analysis import run_analysis_with_pre_saved_embeddings

def test_cas_embeddings():
    """Test loading embeddings with _cas.pt files"""
    
    # Example embeddings directory containing _cas.pt files
    embeddings_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings"
    
    print("Testing _cas.pt embedding files...")
    print(f"Embeddings directory: {embeddings_dir}")
    
    # Check what files exist
    embeddings_path = Path(embeddings_dir)
    if embeddings_path.exists():
        print("\nAvailable embedding files:")
        for file in sorted(embeddings_path.glob("*.pt")):
            print(f"  {file.name}")
        
        # Check specifically for _cas files
        cas_files = list(embeddings_path.glob("*_cas.pt"))
        if cas_files:
            print(f"\nFound {len(cas_files)} _cas.pt files:")
            for file in cas_files:
                print(f"  {file.name}")
        else:
            print("\nNo _cas.pt files found")
    
    # Run analysis with _cas files (if they exist)
    try:
        analyzer = run_analysis_with_pre_saved_embeddings(
            embeddings_dir=embeddings_dir,
            experiment_name="cas_embeddings_test",
            max_components=15,
            zca_threshold=1e-5
        )
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved in: {analyzer.output_dir}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Number of layers analyzed: {len(analyzer.layer_embeddings)}")
        print(f"- Optimal components: {analyzer.optimal_components}")
        
        return analyzer
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_cas_embeddings()
