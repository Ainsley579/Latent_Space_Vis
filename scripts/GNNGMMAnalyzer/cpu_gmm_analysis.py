#!/usr/bin/env python3
"""
CPU-Only GMM Analysis Script (No PyTorch/CUDA dependencies)

This script performs GMM analysis on pre-saved embeddings without requiring PyTorch:
1. Loads embeddings from .npy files (converted from .pt files)
2. Baseline GMM fitting with BIC/AIC for optimal components
3. Layer-wise trajectory analysis
4. Visualization of evolution across layers
5. Quantification of cluster sharpening
6. Boundary analysis for ambiguous molecules

Usage:
    python cpu_gmm_analysis.py --embeddings_dir /path/to/embeddings --experiment_name "cpu_analysis"
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib.patches import Ellipse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Go up to latent-spaces directory
sys.path.append(parent_dir)

class CPUGMMAnalyzer:
    """CPU-only GMM analyzer that works with numpy arrays instead of PyTorch tensors"""
    
    def __init__(self, embeddings_dir, output_dir=None, experiment_name=None, 
                 model_type='dimenet-10conf', checkpoint_index=0):
        self.embeddings_dir = Path(embeddings_dir)
        self.model_type = model_type
        self.checkpoint_index = checkpoint_index
        
        # Set up output directory
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = f"{model_type}_checkpoint{checkpoint_index}_{experiment_name or 'cpu_gmm'}"
            self.output_dir = Path(f"cpu_gmm_results_{base_name}_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for analysis results
        self.layer_embeddings = {}
        self.layer_labels = {}
        self.gmm_results = {}
        self.optimal_components = {}
        self.trajectory_data = {}
        
        print(f"CPU GMM Analyzer initialized")
        print(f"Embeddings directory: {self.embeddings_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def load_embeddings_from_npy(self, sample_frac=1.0):
        """Load embeddings from .npy files (converted from .pt files)"""
        print("Loading embeddings from .npy files...")
        
        # Look for .npy files in the embeddings directory
        npy_files = list(self.embeddings_dir.glob("*.npy"))
        
        if not npy_files:
            print("No .npy files found. Converting .pt files to .npy...")
            self.convert_pt_to_npy()
            npy_files = list(self.embeddings_dir.glob("*.npy"))
        
        for npy_file in npy_files:
            layer_name = npy_file.stem  # filename without extension
            
            print(f"Loading {layer_name}...")
            embeddings = np.load(npy_file)
            
            # Apply sampling if requested
            if sample_frac < 1.0:
                n_samples = int(len(embeddings) * sample_frac)
                indices = np.random.choice(len(embeddings), n_samples, replace=False)
                embeddings = embeddings[indices]
            
            self.layer_embeddings[layer_name] = embeddings
            print(f"  Loaded {embeddings.shape[0]} samples with {embeddings.shape[1]} dimensions")
        
        print(f"Loaded {len(self.layer_embeddings)} layers")
        return self.layer_embeddings
    
    def convert_pt_to_npy(self):
        """Convert .pt files to .npy files for CPU-only processing"""
        print("Converting .pt files to .npy files...")
        
        pt_files = list(self.embeddings_dir.glob("*.pt"))
        
        if not pt_files:
            raise ValueError(f"No .pt files found in {self.embeddings_dir}")
        
        # Try to import torch only for conversion
        try:
            import torch
            print("PyTorch available for conversion")
        except ImportError:
            print("PyTorch not available. Please convert .pt files to .npy manually:")
            print("You can use: python -c \"import torch; import numpy as np; torch.load('file.pt').numpy().save('file.npy')\"")
            raise ImportError("PyTorch required for .pt to .npy conversion")
        
        for pt_file in pt_files:
            npy_file = pt_file.with_suffix('.npy')
            
            if npy_file.exists():
                print(f"  {npy_file.name} already exists, skipping")
                continue
            
            print(f"  Converting {pt_file.name} to {npy_file.name}")
            try:
                # Load tensor and convert to numpy
                tensor = torch.load(pt_file, map_location='cpu')
                if isinstance(tensor, torch.Tensor):
                    array = tensor.numpy()
                else:
                    # Handle case where tensor might be wrapped in a dict or list
                    array = np.array(tensor)
                
                np.save(npy_file, array)
                print(f"    Converted: {array.shape}")
                
            except Exception as e:
                print(f"    Error converting {pt_file.name}: {e}")
                continue
    
    def find_optimal_components(self, max_components=20):
        """Find optimal number of components using BIC/AIC"""
        print(f"Finding optimal components (max: {max_components})...")
        
        for layer_name, embeddings in self.layer_embeddings.items():
            print(f"  Analyzing {layer_name}...")
            
            # Standardize embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # Test different numbers of components
            n_components_range = range(1, min(max_components + 1, len(embeddings) // 10))
            bic_scores = []
            aic_scores = []
            silhouette_scores = []
            
            for n_components in n_components_range:
                try:
                    gmm = GaussianMixture(n_components=n_components, random_state=42)
                    gmm.fit(embeddings_scaled)
                    
                    bic_scores.append(gmm.bic(embeddings_scaled))
                    aic_scores.append(gmm.aic(embeddings_scaled))
                    
                    # Calculate silhouette score
                    labels = gmm.predict(embeddings_scaled)
                    if len(np.unique(labels)) > 1:
                        sil_score = silhouette_score(embeddings_scaled, labels)
                    else:
                        sil_score = 0
                    silhouette_scores.append(sil_score)
                    
                except Exception as e:
                    print(f"    Error with {n_components} components: {e}")
                    bic_scores.append(np.inf)
                    aic_scores.append(np.inf)
                    silhouette_scores.append(0)
            
            # Find optimal number of components
            optimal_bic = n_components_range[np.argmin(bic_scores)]
            optimal_aic = n_components_range[np.argmin(aic_scores)]
            optimal_sil = n_components_range[np.argmax(silhouette_scores)]
            
            self.optimal_components[layer_name] = {
                'bic': optimal_bic,
                'aic': optimal_aic,
                'silhouette': optimal_sil,
                'bic_scores': bic_scores,
                'aic_scores': aic_scores,
                'silhouette_scores': silhouette_scores,
                'n_components_range': list(n_components_range)
            }
            
            print(f"    Optimal components - BIC: {optimal_bic}, AIC: {optimal_aic}, Silhouette: {optimal_sil}")
        
        return self.optimal_components
    
    def fit_gmm_models(self, use_bic=True):
        """Fit GMM models with optimal components"""
        print("Fitting GMM models...")
        
        for layer_name, embeddings in self.layer_embeddings.items():
            print(f"  Fitting GMM for {layer_name}...")
            
            # Get optimal number of components
            if use_bic:
                n_components = self.optimal_components[layer_name]['bic']
            else:
                n_components = self.optimal_components[layer_name]['aic']
            
            # Standardize embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # Fit GMM
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(embeddings_scaled)
            
            # Get predictions and probabilities
            labels = gmm.predict(embeddings_scaled)
            probabilities = gmm.predict_proba(embeddings_scaled)
            
            self.gmm_results[layer_name] = {
                'gmm': gmm,
                'scaler': scaler,
                'labels': labels,
                'probabilities': probabilities,
                'n_components': n_components,
                'embeddings_scaled': embeddings_scaled
            }
            
            print(f"    Fitted GMM with {n_components} components")
        
        return self.gmm_results
    
    def analyze_layer_trajectories(self):
        """Analyze how clusters evolve across layers"""
        print("Analyzing layer trajectories...")
        
        # Get all unique labels across layers
        all_labels = set()
        for layer_name, results in self.gmm_results.items():
            all_labels.update(results['labels'])
        
        # Create trajectory matrix
        n_samples = len(next(iter(self.layer_embeddings.values())))
        n_layers = len(self.layer_embeddings)
        
        trajectory_matrix = np.zeros((n_samples, n_layers))
        layer_names = list(self.layer_embeddings.keys())
        
        for i, layer_name in enumerate(layer_names):
            trajectory_matrix[:, i] = self.gmm_results[layer_name]['labels']
        
        self.trajectory_data = {
            'matrix': trajectory_matrix,
            'layer_names': layer_names,
            'n_samples': n_samples,
            'n_layers': n_layers
        }
        
        print(f"Created trajectory matrix: {trajectory_matrix.shape}")
        return self.trajectory_data
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # 1. Component selection plots
        self._plot_component_selection()
        
        # 2. Layer-wise cluster visualizations
        self._plot_layer_clusters()
        
        # 3. Trajectory analysis
        self._plot_trajectory_analysis()
        
        # 4. Summary statistics
        self._plot_summary_statistics()
        
        # 5. Parallel coordinates plot
        self._plot_parallel_coordinates()
        
        print("Visualizations saved to:", self.output_dir)
    
    def _plot_component_selection(self):
        """Plot BIC/AIC/Silhouette scores for component selection"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, layer_name in enumerate(self.layer_embeddings.keys()):
            data = self.optimal_components[layer_name]
            n_components_range = data['n_components_range']
            
            # BIC plot
            axes[0].plot(n_components_range, data['bic_scores'], 'o-', label=layer_name)
            axes[0].set_xlabel('Number of Components')
            axes[0].set_ylabel('BIC Score')
            axes[0].set_title('BIC Score vs Components')
            axes[0].legend()
            
            # AIC plot
            axes[1].plot(n_components_range, data['aic_scores'], 'o-', label=layer_name)
            axes[1].set_xlabel('Number of Components')
            axes[1].set_ylabel('AIC Score')
            axes[1].set_title('AIC Score vs Components')
            axes[1].legend()
            
            # Silhouette plot
            axes[2].plot(n_components_range, data['silhouette_scores'], 'o-', label=layer_name)
            axes[2].set_xlabel('Number of Components')
            axes[2].set_ylabel('Silhouette Score')
            axes[2].set_title('Silhouette Score vs Components')
            axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_selection.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_layer_clusters(self):
        """Plot cluster visualizations for each layer"""
        for layer_name, embeddings in self.layer_embeddings.items():
            results = self.gmm_results[layer_name]
            labels = results['labels']
            embeddings_scaled = results['embeddings_scaled']
            
            # Use PCA for visualization
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings_scaled)
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Cluster plot
            scatter = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=labels, cmap='tab10', alpha=0.6)
            axes[0].set_title(f'{layer_name} - Clusters (PCA)')
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.colorbar(scatter, ax=axes[0])
            
            # Probability plot (max probability)
            max_probs = np.max(results['probabilities'], axis=1)
            scatter = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=max_probs, cmap='viridis', alpha=0.6)
            axes[1].set_title(f'{layer_name} - Max Probability')
            axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.colorbar(scatter, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{layer_name}_clusters.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_trajectory_analysis(self):
        """Plot trajectory analysis"""
        if not self.trajectory_data:
            return
        
        trajectory_matrix = self.trajectory_data['matrix']
        layer_names = self.trajectory_data['layer_names']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Trajectory heatmap
        im = axes[0, 0].imshow(trajectory_matrix.T, aspect='auto', cmap='tab10')
        axes[0, 0].set_title('Cluster Trajectories Across Layers')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Layer')
        axes[0, 0].set_yticks(range(len(layer_names)))
        axes[0, 0].set_yticklabels(layer_names)
        plt.colorbar(im, ax=axes[0, 0])
        
        # Cluster stability
        stability_scores = []
        for i in range(len(layer_names) - 1):
            stability = np.mean(trajectory_matrix[:, i] == trajectory_matrix[:, i + 1])
            stability_scores.append(stability)
        
        axes[0, 1].plot(range(1, len(layer_names)), stability_scores, 'o-')
        axes[0, 1].set_title('Cluster Stability Between Adjacent Layers')
        axes[0, 1].set_xlabel('Layer Transition')
        axes[0, 1].set_ylabel('Stability Score')
        axes[0, 1].grid(True)
        
        # Number of clusters per layer
        n_clusters = [len(np.unique(trajectory_matrix[:, i])) for i in range(len(layer_names))]
        axes[1, 0].bar(range(len(layer_names)), n_clusters)
        axes[1, 0].set_title('Number of Clusters per Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Number of Clusters')
        axes[1, 0].set_xticks(range(len(layer_names)))
        axes[1, 0].set_xticklabels(layer_names, rotation=45)
        
        # Cluster size distribution
        cluster_sizes = []
        for i, layer_name in enumerate(layer_names):
            labels = trajectory_matrix[:, i]
            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes.extend(counts)
        
        axes[1, 1].hist(cluster_sizes, bins=20, alpha=0.7)
        axes[1, 1].set_title('Cluster Size Distribution')
        axes[1, 1].set_xlabel('Cluster Size')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trajectory_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_summary_statistics(self):
        """Plot summary statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Optimal components
        layer_names = list(self.optimal_components.keys())
        bic_components = [self.optimal_components[layer]['bic'] for layer in layer_names]
        aic_components = [self.optimal_components[layer]['aic'] for layer in layer_names]
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, bic_components, width, label='BIC', alpha=0.7)
        axes[0, 0].bar(x + width/2, aic_components, width, label='AIC', alpha=0.7)
        axes[0, 0].set_title('Optimal Components per Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Number of Components')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(layer_names, rotation=45)
        axes[0, 0].legend()
        
        # Embedding dimensions
        dimensions = [embeddings.shape[1] for embeddings in self.layer_embeddings.values()]
        axes[0, 1].bar(layer_names, dimensions)
        axes[0, 1].set_title('Embedding Dimensions per Layer')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Dimensions')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sample counts
        sample_counts = [embeddings.shape[0] for embeddings in self.layer_embeddings.values()]
        axes[1, 0].bar(layer_names, sample_counts)
        axes[1, 0].set_title('Sample Counts per Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Silhouette scores
        silhouette_scores = [np.max(self.optimal_components[layer]['silhouette_scores']) 
                           for layer in layer_names]
        axes[1, 1].bar(layer_names, silhouette_scores)
        axes[1, 1].set_title('Best Silhouette Scores per Layer')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Silhouette Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parallel_coordinates(self):
        """Create parallel coordinates plot for layer-wise trajectory"""
        print("Creating parallel coordinates plot...")
        
        # Prepare data for parallel coordinates
        n_layers = len(self.layer_embeddings)
        n_samples = len(list(self.layer_embeddings.values())[0])
        
        # Create DataFrame with layer embeddings
        data_dict = {}
        for i, (layer_name, embeddings) in enumerate(self.layer_embeddings.items()):
            # Use first few principal components for visualization
            pca = PCA(n_components=min(3, embeddings.shape[1]))
            embeddings_pca = pca.fit_transform(embeddings)
            
            for j in range(embeddings_pca.shape[1]):
                data_dict[f'{layer_name}_PC{j+1}'] = embeddings_pca[:, j]
        
        df = pd.DataFrame(data_dict)
        
        # Sample data for visualization (too many points can be overwhelming)
        if len(df) > 1000:
            sample_indices = np.random.choice(len(df), 1000, replace=False)
            df_sample = df.iloc[sample_indices]
        else:
            df_sample = df
        
        # Create parallel coordinates plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot parallel coordinates
        for i in range(len(df_sample)):
            ax.plot(range(len(df_sample.columns)), df_sample.iloc[i], 
                   alpha=0.1, linewidth=0.5, color='blue')
        
        ax.set_xticks(range(len(df_sample.columns)))
        ax.set_xticklabels(df_sample.columns, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title('Layer-wise Trajectory: Parallel Coordinates Plot')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parallel_coordinates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Parallel coordinates plot saved")
    
    def save_results(self):
        """Save analysis results to files"""
        print("Saving results...")
        
        # Save optimal components
        components_df = pd.DataFrame(self.optimal_components).T
        components_df.to_csv(self.output_dir / 'optimal_components.csv')
        
        # Save trajectory data
        if self.trajectory_data:
            trajectory_df = pd.DataFrame(
                self.trajectory_data['matrix'],
                columns=self.trajectory_data['layer_names']
            )
            trajectory_df.to_csv(self.output_dir / 'trajectory_matrix.csv', index=False)
        
        # Save summary
        summary = {
            'experiment_name': self.experiment_name,
            'model_type': self.model_type,
            'checkpoint_index': self.checkpoint_index,
            'n_layers': len(self.layer_embeddings),
            'layer_names': list(self.layer_embeddings.keys()),
            'optimal_components': self.optimal_components,
            'output_dir': str(self.output_dir)
        }
        
        import json
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Results saved to: {self.output_dir}")
    
    def run_full_analysis(self, max_components=20, sample_frac=1.0):
        """Run complete GMM analysis pipeline"""
        print("Starting CPU-only GMM analysis...")
        
        # Load embeddings
        self.load_embeddings_from_npy(sample_frac=sample_frac)
        
        # Find optimal components
        self.find_optimal_components(max_components=max_components)
        
        # Fit GMM models
        self.fit_gmm_models(use_bic=True)
        
        # Analyze trajectories
        self.analyze_layer_trajectories()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        print("Analysis completed successfully!")
        return self

def run_cpu_gmm_analysis(embeddings_dir, output_dir=None, experiment_name=None,
                        max_components=20, model_type='dimenet-10conf', 
                        checkpoint_index=0, sample_frac=1.0):
    """Convenience function to run CPU-only GMM analysis"""
    
    analyzer = CPUGMMAnalyzer(
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        model_type=model_type,
        checkpoint_index=checkpoint_index
    )
    
    return analyzer.run_full_analysis(
        max_components=max_components,
        sample_frac=sample_frac
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-only GMM analysis")
    parser.add_argument('--embeddings_dir', type=str, required=True,
                       help='Directory containing embeddings (.npy or .pt files)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--max_components', type=int, default=20,
                       help='Maximum number of components to test')
    parser.add_argument('--model_type', type=str, default='dimenet-10conf',
                       help='Model type')
    parser.add_argument('--checkpoint_index', type=int, default=0,
                       help='Checkpoint index')
    parser.add_argument('--sample_frac', type=float, default=1.0,
                       help='Fraction of data to use')
    
    args = parser.parse_args()
    
    analyzer = run_cpu_gmm_analysis(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        max_components=args.max_components,
        model_type=args.model_type,
        checkpoint_index=args.checkpoint_index,
        sample_frac=args.sample_frac
    )
    
    print(f"\nAnalysis completed! Results saved in: {analyzer.output_dir}")
