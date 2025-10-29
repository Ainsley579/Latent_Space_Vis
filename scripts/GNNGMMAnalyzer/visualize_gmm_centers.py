#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create 4 figures showing GMM center data point IDs for each combination
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_center_visualization(base_dir):
    """Create 4 figures showing GMM center data point IDs"""
    
    combinations = [
        "fixed_no_zca",
        "fixed_with_zca", 
        "adaptive_no_zca",
        "adaptive_with_zca"
    ]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GMM Center Data Point IDs Across All Combinations', fontsize=16, fontweight='bold')
    
    for idx, combination in enumerate(combinations):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Load the CSV file
        csv_path = Path(base_dir) / combination / "gmm_centers_and_neighbors.csv"
        
        if not csv_path.exists():
            print("Warning: {} not found".format(csv_path))
            continue
            
        df = pd.read_csv(csv_path)
        
        # Filter for centers only (is_center == True)
        centers_df = df[df['is_center'] == True].copy()
        
        # Create a matrix: rows = layers, columns = components
        layers = sorted(centers_df['layer'].unique())
        max_components = centers_df['component'].max() + 1
        
        # Create matrix to store center data point IDs
        center_matrix = np.full((len(layers), max_components), np.nan)
        
        for _, row_data in centers_df.iterrows():
            layer_idx = layers.index(row_data['layer'])
            component_idx = row_data['component']
            center_matrix[layer_idx, component_idx] = row_data['original_molecule_index']
        
        # Create heatmap
        im = ax.imshow(center_matrix, cmap='viridis', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(max_components))
        ax.set_xticklabels(['C{}'.format(i) for i in range(max_components)])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, rotation=0)
        
        ax.set_xlabel('GMM Component')
        ax.set_ylabel('GNN Layer')
        
        # Add title with combination info
        zca_status = "With ZCA" if "with_zca" in combination else "No ZCA"
        comp_status = "Fixed" if "fixed" in combination else "Adaptive"
        ax.set_title('{} + {}\n({} centers)'.format(comp_status, zca_status, len(centers_df)), fontweight='bold')
        
        # Add text annotations showing the data point IDs
        for i in range(len(layers)):
            for j in range(max_components):
                if not np.isnan(center_matrix[i, j]):
                    text = ax.text(j, i, '{}'.format(int(center_matrix[i, j])), 
                                 ha="center", va="center", 
                                 color="white" if center_matrix[i, j] > np.nanmax(center_matrix)/2 else "black",
                                 fontweight='bold', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Data Point ID')
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(base_dir) / 'gmm_centers_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Center visualization saved to: {}".format(output_path))
    
    # Also create individual figures for each combination
    create_individual_center_plots(base_dir, combinations)
    
    # Create summary statistics
    create_center_summary_stats(base_dir, combinations)

def create_individual_center_plots(base_dir, combinations):
    """Create individual detailed plots for each combination"""
    
    for combination in combinations:
        csv_path = Path(base_dir) / combination / "gmm_centers_and_neighbors.csv"
        
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        centers_df = df[df['is_center'] == True].copy()
        
        # Create detailed plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Heatmap of center data point IDs
        layers = sorted(centers_df['layer'].unique())
        max_components = centers_df['component'].max() + 1
        
        center_matrix = np.full((len(layers), max_components), np.nan)
        
        for _, row_data in centers_df.iterrows():
            layer_idx = layers.index(row_data['layer'])
            component_idx = row_data['component']
            center_matrix[layer_idx, component_idx] = row_data['original_molecule_index']
        
        im1 = ax1.imshow(center_matrix, cmap='viridis', aspect='auto')
        
        ax1.set_xticks(range(max_components))
        ax1.set_xticklabels(['C{}'.format(i) for i in range(max_components)])
        ax1.set_yticks(range(len(layers)))
        ax1.set_yticklabels(layers, rotation=0)
        
        ax1.set_xlabel('GMM Component')
        ax1.set_ylabel('GNN Layer')
        ax1.set_title('Center Data Point IDs\n{}'.format(combination), fontweight='bold')
        
        # Add text annotations
        for i in range(len(layers)):
            for j in range(max_components):
                if not np.isnan(center_matrix[i, j]):
                    text = ax1.text(j, i, '{}'.format(int(center_matrix[i, j])), 
                                  ha="center", va="center", 
                                  color="white" if center_matrix[i, j] > np.nanmax(center_matrix)/2 else "black",
                                  fontweight='bold', fontsize=9)
        
        plt.colorbar(im1, ax=ax1, label='Data Point ID')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of center data point IDs
        center_ids = centers_df['original_molecule_index'].values
        
        ax2.hist(center_ids, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax2.set_xlabel('Data Point ID')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Center Data Point IDs\n{}'.format(combination), fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = 'Total Centers: {}\n'.format(len(center_ids))
        stats_text += 'Unique Centers: {}\n'.format(len(set(center_ids)))
        stats_text += 'Min ID: {}\n'.format(min(center_ids))
        stats_text += 'Max ID: {}\n'.format(max(center_ids))
        stats_text += 'Mean ID: {:.1f}'.format(np.mean(center_ids))
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save individual figure
        output_path = Path(base_dir) / 'gmm_centers_{}.png'.format(combination)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Individual plot saved to: {}".format(output_path))

def create_center_summary_stats(base_dir, combinations):
    """Create summary statistics comparing centers across combinations"""
    
    summary_data = []
    
    for combination in combinations:
        csv_path = Path(base_dir) / combination / "gmm_centers_and_neighbors.csv"
        
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        centers_df = df[df['is_center'] == True].copy()
        
        center_ids = centers_df['original_molecule_index'].values
        
        summary_data.append({
            'combination': combination,
            'total_centers': len(center_ids),
            'unique_centers': len(set(center_ids)),
            'min_center_id': min(center_ids),
            'max_center_id': max(center_ids),
            'mean_center_id': np.mean(center_ids),
            'std_center_id': np.std(center_ids),
            'coverage_percentage': (len(set(center_ids)) / 152) * 100  # 152 total molecules
        })
    
    # Create summary DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(base_dir) / 'center_summary_statistics.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print("Center summary statistics saved to: {}".format(summary_path))
    
    # Print summary to console
    print("\n" + "="*80)
    print("CENTER SUMMARY STATISTICS:")
    print("="*80)
    for _, row in summary_df.iterrows():
        print("\n{}:".format(row['combination']))
        print("  Total centers: {}".format(row['total_centers']))
        print("  Unique center data points: {}".format(row['unique_centers']))
        print("  Coverage: {:.1f}% of all molecules".format(row['coverage_percentage']))
        print("  Center ID range: {} - {}".format(row['min_center_id'], row['max_center_id']))
        print("  Mean center ID: {:.1f} ± {:.1f}".format(row['mean_center_id'], row['std_center_id']))

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # Default to the most recent comprehensive analysis
        base_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzerResults/comprehensive_analysis_20251019_180106"
    
    print("Creating GMM center visualizations for: {}".format(base_dir))
    
    create_center_visualization(base_dir)
    
    print("\nAll visualizations completed!")

if __name__ == "__main__":
    main()
