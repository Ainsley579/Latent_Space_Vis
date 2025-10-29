#!/usr/bin/env python3
"""
Test script to verify the no-overlap neighbor selection works correctly
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from comprehensive_gmm_analysis import ComprehensiveGMMAnalyzer

def test_no_overlap_analysis():
    """Test the no-overlap neighbor selection"""
    print("Testing no-overlap neighbor selection...")
    
    # Create a small test analyzer with limited components
    embeddings_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings"
    output_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzerResults/test_no_overlap_verification"
    
    analyzer = ComprehensiveGMMAnalyzer(
        embeddings_dir=embeddings_dir,
        base_output_dir=output_dir,
        max_components=5,  # Small number for testing
        zca_threshold=1e-5
    )
    
    # Test just one combination with a small number of components
    print("\nTesting fixed_no_zca combination...")
    gmm_results, center_data, output_dir = analyzer.run_combination(
        "fixed_no_zca", use_fixed_components=True, use_zca=False
    )
    
    # Analyze the results
    df_centers = pd.DataFrame(center_data)
    
    print(f"\nAnalysis Results:")
    print(f"Total data points: {len(df_centers)}")
    print(f"Total centers: {len(df_centers[df_centers['is_center'] == True])}")
    print(f"Unique neighbor indices: {len(set(df_centers['neighbor_index']))}")
    
    # Check for overlap within each layer
    print(f"\nChecking for overlap within each layer:")
    for layer in df_centers['layer'].unique():
        layer_data = df_centers[df_centers['layer'] == layer]
        unique_indices = len(set(layer_data['neighbor_index']))
        total_points = len(layer_data)
        
        print(f"  {layer}: {total_points} total points, {unique_indices} unique indices")
        
        if unique_indices == total_points:
            print(f"    ✅ No overlap detected")
        else:
            print(f"    ❌ Overlap detected: {total_points - unique_indices} duplicate indices")
            
            # Find duplicates
            index_counts = layer_data['neighbor_index'].value_counts()
            duplicates = index_counts[index_counts > 1]
            if len(duplicates) > 0:
                print(f"    Duplicate indices: {duplicates.to_dict()}")
    
    # Check component sizes
    print(f"\nComponent size analysis:")
    for layer in df_centers['layer'].unique():
        layer_data = df_centers[df_centers['layer'] == layer]
        print(f"  {layer}:")
        for comp in layer_data['component'].unique():
            comp_data = layer_data[layer_data['component'] == comp]
            comp_size = comp_data['component_size'].iloc[0]
            selected_points = len(comp_data)
            print(f"    Component {comp}: {comp_size} members, {selected_points} selected points")
    
    print(f"\nTest completed. Results saved in: {output_dir}")

if __name__ == "__main__":
    test_no_overlap_analysis()
