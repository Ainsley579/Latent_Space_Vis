#!/usr/bin/env python3
"""
Example script to run GNN GMM analysis

This script demonstrates how to use the GNN GMM analysis with different configurations.
"""

import os
import sys
from pathlib import Path

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Go up to latent-spaces directory
sys.path.append(parent_dir)

from gnn_gmm_analysis import GNNGMMAnalyzer

def run_example_analysis():
    """Run example GMM analysis"""
    
    print("Running example GNN GMM analysis...")
    
    # Create analyzer for dimenet-10conf model
    analyzer = GNNGMMAnalyzer(
        model_type="dimenet-10conf",
        checkpoint_index=0,
        output_dir="../GNNGMMAnalyzerResults/example_gmm_results",
        sample_frac=0.5  # Use 50% of data for faster analysis
    )
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    print("Example analysis completed!")
    print("Check the '../GNNGMMAnalyzerResults/example_gmm_results' directory for outputs.")

def run_full_analysis():
    """Run full analysis on all data"""
    
    print("Running full GNN GMM analysis...")
    
    # Create analyzer for full analysis
    analyzer = GNNGMMAnalyzer(
        model_type="dimenet-10conf",
        checkpoint_index=0,
        output_dir="../GNNGMMAnalyzerResults/full_gmm_results",
        sample_frac=1.0  # Use all data
    )
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    print("Full analysis completed!")
    print("Check the '../GNNGMMAnalyzerResults/full_gmm_results' directory for outputs.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GNN GMM Analysis Example")
    parser.add_argument('--mode', type=str, default='example',
                       choices=['example', 'full'],
                       help='Analysis mode: example (50% data) or full (100% data)')
    
    args = parser.parse_args()
    
    if args.mode == 'example':
        run_example_analysis()
    elif args.mode == 'full':
        run_full_analysis()
