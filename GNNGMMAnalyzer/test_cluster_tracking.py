#!/usr/bin/env python3
"""
Quick test script for cluster tracking visualization
"""

import os
import sys
from pathlib import Path

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from gnn_gmm_analysis import run_analysis_with_pre_saved_embeddings

def main():
    """Test cluster tracking with improved visualization"""
    
    print("Testing improved cluster tracking visualization...")
    
    # Run analysis with improved cluster tracking
    analyzer = run_analysis_with_pre_saved_embeddings(
        embeddings_dir="/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings",
        experiment_name="improved_cluster_tracking",
        max_components=15
    )
    
    print(f"\nAnalysis completed!")
    print(f"Results saved in: {analyzer.output_dir}")
    print(f"Total molecules analyzed: {len(analyzer.layer_embeddings['emb'])}")

if __name__ == "__main__":
    main()
