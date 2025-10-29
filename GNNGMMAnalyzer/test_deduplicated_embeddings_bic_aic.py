#!/usr/bin/env python3
"""
Script for running GMM analysis with deduplicated embeddings (original .pt files)
Focuses on BIC and AIC component selection analysis
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

class DeduplicatedEmbeddingAnalyzer:
    """Analyzer for deduplicated embeddings focusing on BIC/AIC analysis"""
    
    def __init__(self, embeddings_dir, output_dir=None, max_components=20):
        self.embeddings_dir = Path(embeddings_dir)
        self.max_components = max_components
        self.layer_embeddings = {}
        self.bic_aic_results = {}
        
        # Create output directory
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"../GNNGMMAnalyzerResults/deduplicated_bic_aic_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
    def load_deduplicated_embeddings(self):
        """Load deduplicated embeddings (original .pt files)"""
        print(f"Loading deduplicated embeddings from: {self.embeddings_dir}")
        
        # Define expected layer files (using original files, not unique)
        layer_mappings = {
            'emb': 'emb.pt',
            'output_block_0': 'output0.pt',
            'output_block_1': 'output1.pt',
            'output_block_2': 'output2.pt',
            'output_block_3': 'output3.pt',
            'output_block_4': 'output4.pt'
        }
        
        # Load embeddings for each layer
        for layer_name, filename in layer_mappings.items():
            filepath = self.embeddings_dir / filename
            
            if not filepath.exists():
                print(f"  Warning: {filename} not found, skipping {layer_name}")
                continue
            
            try:
                # Load tensor and convert to numpy
                embeddings_tensor = torch.load(filepath, map_location='cpu')
                embeddings_np = embeddings_tensor.detach().cpu().numpy()
                
                # Reshape if needed (flatten extra dimensions)
                if embeddings_np.ndim > 2:
                    original_shape = embeddings_np.shape
                    # Flatten all dimensions except the first (batch dimension)
                    embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)
                    print(f"  Loading {filename} -> {layer_name} (shape: {original_shape} -> {embeddings_np.shape})")
                else:
                    print(f"  Loading {filename} -> {layer_name} (shape: {embeddings_np.shape})")
                
                self.layer_embeddings[layer_name] = embeddings_np
                
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                continue
        
        if not self.layer_embeddings:
            raise ValueError("No deduplicated embeddings were successfully loaded")
        
        print(f"Successfully loaded deduplicated embeddings from {len(self.layer_embeddings)} layers")
        return self.layer_embeddings
    
    def analyze_bic_aic_for_all_layers(self):
        """Analyze BIC and AIC for all layers"""
        print("Analyzing BIC and AIC for all layers...")
        
        for layer_name, embeddings in self.layer_embeddings.items():
            print(f"\nAnalyzing layer: {layer_name}")
            bic_aic_result = self._analyze_bic_aic_for_layer(layer_name, embeddings)
            self.bic_aic_results[layer_name] = bic_aic_result
        
        # Create summary analysis
        self._create_bic_aic_summary()
        
        return self.bic_aic_results
    
    def _analyze_bic_aic_for_layer(self, layer_name, embeddings):
        """Analyze BIC and AIC for a single layer"""
        print(f"  Analyzing BIC/AIC for {layer_name} with {embeddings.shape[0]} samples, {embeddings.shape[1]} features")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Test different numbers of components
        n_components_range = range(2, self.max_components + 1)
        bic_scores = []
        aic_scores = []
        
        for n_components in n_components_range:
            try:
                # Fit GMM
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42,
                    max_iter=200
                )
                gmm.fit(embeddings_scaled)
                
                # Calculate scores
                bic_score = gmm.bic(embeddings_scaled)
                aic_score = gmm.aic(embeddings_scaled)
                
                bic_scores.append(bic_score)
                aic_scores.append(aic_score)
                
                print(f"    {n_components} components: BIC={bic_score:.2f}, AIC={aic_score:.2f}")
                
            except Exception as e:
                print(f"    Warning: Failed to fit {n_components} components: {e}")
                bic_scores.append(float('inf'))
                aic_scores.append(float('inf'))
        
        # Find optimal components
        optimal_bic = n_components_range[np.argmin(bic_scores)]
        optimal_aic = n_components_range[np.argmin(aic_scores)]
        
        print(f"  Optimal components - BIC: {optimal_bic}, AIC: {optimal_aic}")
        
        # Store results
        result = {
            'layer_name': layer_name,
            'n_samples': embeddings.shape[0],
            'n_features': embeddings.shape[1],
            'n_components_range': list(n_components_range),
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'optimal_bic': optimal_bic,
            'optimal_aic': optimal_aic,
            'min_bic': min(bic_scores),
            'min_aic': min(aic_scores)
        }
        
        # Create individual layer plot
        self._plot_bic_aic_for_layer(result)
        
        return result
    
    def _plot_bic_aic_for_layer(self, result):
        """Create BIC/AIC plot for a single layer"""
        layer_name = result['layer_name']
        n_components_range = result['n_components_range']
        bic_scores = result['bic_scores']
        aic_scores = result['aic_scores']
        optimal_bic = result['optimal_bic']
        optimal_aic = result['optimal_aic']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # BIC plot
        ax1.plot(n_components_range, bic_scores, 'o-', linewidth=2, markersize=6, color='blue')
        ax1.axvline(optimal_bic, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal BIC: {optimal_bic}')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('BIC Score')
        ax1.set_title(f'BIC vs Components - {layer_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # AIC plot
        ax2.plot(n_components_range, aic_scores, 'o-', linewidth=2, markersize=6, color='green')
        ax2.axvline(optimal_aic, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal AIC: {optimal_aic}')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('AIC Score')
        ax2.set_title(f'AIC vs Components - {layer_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'bic_aic_{layer_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  BIC/AIC plot saved: bic_aic_{layer_name}.png")
    
    def _create_bic_aic_summary(self):
        """Create summary analysis of BIC/AIC results across all layers"""
        print("\nCreating BIC/AIC summary analysis...")
        
        # Prepare data for summary plots
        layer_names = list(self.bic_aic_results.keys())
        optimal_bic_values = [result['optimal_bic'] for result in self.bic_aic_results.values()]
        optimal_aic_values = [result['optimal_aic'] for result in self.bic_aic_results.values()]
        min_bic_values = [result['min_bic'] for result in self.bic_aic_results.values()]
        min_aic_values = [result['min_aic'] for result in self.bic_aic_results.values()]
        
        # Create summary plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Optimal components (BIC vs AIC)
        ax1.scatter(optimal_bic_values, optimal_aic_values, s=100, alpha=0.7, c=range(len(layer_names)), 
                   cmap='viridis', edgecolors='black', linewidth=2)
        ax1.set_xlabel('Optimal Components (BIC)')
        ax1.set_ylabel('Optimal Components (AIC)')
        ax1.set_title('Optimal Components: BIC vs AIC')
        ax1.grid(True, alpha=0.3)
        
        # Add layer labels
        for i, layer in enumerate(layer_names):
            ax1.annotate(layer, (optimal_bic_values[i], optimal_aic_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add diagonal line (BIC = AIC)
        max_val = max(max(optimal_bic_values), max(optimal_aic_values))
        min_val = min(min(optimal_bic_values), min(optimal_aic_values))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='BIC = AIC')
        ax1.legend()
        
        # Plot 2: Optimal components by layer
        x_pos = np.arange(len(layer_names))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, optimal_bic_values, width, label='BIC', alpha=0.8, color='blue')
        bars2 = ax2.bar(x_pos + width/2, optimal_aic_values, width, label='AIC', alpha=0.8, color='green')
        
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Optimal Components')
        ax2.set_title('Optimal Components by Layer')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Minimum BIC scores by layer
        ax3.bar(x_pos, min_bic_values, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Minimum BIC Score')
        ax3.set_title('Minimum BIC Scores by Layer')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(layer_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Minimum AIC scores by layer
        ax4.bar(x_pos, min_aic_values, alpha=0.7, color='green', edgecolor='black')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Minimum AIC Score')
        ax4.set_title('Minimum AIC Scores by Layer')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(layer_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bic_aic_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Export results to CSV
        self._export_bic_aic_to_csv()
        
        print("BIC/AIC summary analysis completed")
    
    def _export_bic_aic_to_csv(self):
        """Export BIC/AIC results to CSV files"""
        print("Exporting BIC/AIC results to CSV...")
        
        # Export summary results
        summary_data = []
        for layer_name, result in self.bic_aic_results.items():
            summary_data.append({
                'layer_name': layer_name,
                'n_samples': result['n_samples'],
                'n_features': result['n_features'],
                'optimal_bic': result['optimal_bic'],
                'optimal_aic': result['optimal_aic'],
                'min_bic': result['min_bic'],
                'min_aic': result['min_aic'],
                'bic_aic_difference': abs(result['optimal_bic'] - result['optimal_aic'])
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(self.output_dir / 'bic_aic_summary.csv', index=False)
        
        # Export detailed results for each layer
        for layer_name, result in self.bic_aic_results.items():
            detailed_data = []
            for i, n_comp in enumerate(result['n_components_range']):
                detailed_data.append({
                    'layer_name': layer_name,
                    'n_components': n_comp,
                    'bic_score': result['bic_scores'][i],
                    'aic_score': result['aic_scores'][i],
                    'is_optimal_bic': n_comp == result['optimal_bic'],
                    'is_optimal_aic': n_comp == result['optimal_aic']
                })
            
            df_detailed = pd.DataFrame(detailed_data)
            df_detailed.to_csv(self.output_dir / f'bic_aic_detailed_{layer_name}.csv', index=False)
        
        print("CSV export completed:")
        print("  - bic_aic_summary.csv")
        for layer_name in self.bic_aic_results.keys():
            print(f"  - bic_aic_detailed_{layer_name}.csv")
    
    def run_analysis(self):
        """Run the complete BIC/AIC analysis"""
        print("Starting deduplicated embeddings BIC/AIC analysis...")
        start_time = time.time()
        
        # Step 1: Load deduplicated embeddings
        self.load_deduplicated_embeddings()
        
        # Step 2: Analyze BIC/AIC for all layers
        self.analyze_bic_aic_for_all_layers()
        
        total_time = time.time() - start_time
        print(f"\nAnalysis completed in {total_time:.2f} seconds")
        print(f"Results saved in: {self.output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY OF BIC/AIC ANALYSIS")
        print("="*60)
        for layer_name, result in self.bic_aic_results.items():
            print(f"{layer_name}:")
            print(f"  Samples: {result['n_samples']}, Features: {result['n_features']}")
            print(f"  Optimal BIC: {result['optimal_bic']} components (score: {result['min_bic']:.2f})")
            print(f"  Optimal AIC: {result['optimal_aic']} components (score: {result['min_aic']:.2f})")
            print(f"  BIC-AIC difference: {abs(result['optimal_bic'] - result['optimal_aic'])} components")
            print()


def main():
    parser = argparse.ArgumentParser(description='Analyze BIC/AIC for deduplicated embeddings')
    parser.add_argument('--embeddings_dir', type=str,
                       default='/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings',
                       help='Directory containing deduplicated embedding files (.pt files)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (auto-generated if not specified)')
    parser.add_argument('--max_components', type=int, default=20,
                       help='Maximum number of components to test for BIC/AIC')
    
    args = parser.parse_args()
    
    print("Running deduplicated embeddings BIC/AIC analysis...")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Max components: {args.max_components}")
    print(f"Output directory: {args.output_dir or 'Auto-generated'}")
    
    # Create analyzer and run analysis
    analyzer = DeduplicatedEmbeddingAnalyzer(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        max_components=args.max_components
    )
    
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
