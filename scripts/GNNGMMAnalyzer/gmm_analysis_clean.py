#!/usr/bin/env python
"""
Clean GMM Analysis Script with Parallel Coordinates
Python 2.7 compatible - No PyTorch/CUDA dependencies
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CleanGMMAnalyzer:
    """Clean GMM analyzer with parallel coordinates support"""
    
    def __init__(self, embeddings_dir, output_dir=None, experiment_name=None):
        self.embeddings_dir = embeddings_dir
        
        # Set up output directory
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = "gmm_analysis_{}".format(experiment_name or 'clean')
            self.output_dir = "{}_{}".format(base_name, timestamp)
        else:
            self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Storage for analysis results
        self.layer_embeddings = {}
        self.gmm_results = {}
        self.optimal_components = {}
        
        print("Clean GMM Analyzer initialized")
        print("Embeddings directory: {}".format(self.embeddings_dir))
        print("Output directory: {}".format(self.output_dir))
    
    def check_files(self):
        """Check what files are available"""
        pt_pattern = os.path.join(self.embeddings_dir, "*.pt")
        npy_pattern = os.path.join(self.embeddings_dir, "*.npy")
        
        pt_files = glob.glob(pt_pattern)
        npy_files = glob.glob(npy_pattern)
        
        print("Found {} .pt files".format(len(pt_files)))
        print("Found {} .npy files".format(len(npy_files)))
        
        if pt_files:
            print(".pt files:")
            for f in pt_files:
                print("  - {}".format(os.path.basename(f)))
        
        if npy_files:
            print(".npy files:")
            for f in npy_files:
                print("  - {}".format(os.path.basename(f)))
        
        return pt_files, npy_files
    
    def provide_conversion_instructions(self):
        """Provide instructions for converting .pt files"""
        print("\n" + "="*60)
        print("CONVERSION INSTRUCTIONS")
        print("="*60)
        print("\nTo convert your .pt files to .npy files, run:")
        print("\ncd {}".format(self.embeddings_dir))
        print("python -c \"")
        print("import torch")
        print("import numpy as np")
        print("import os")
        print("")
        print("files = ['emb.pt', 'output0.pt', 'output1.pt', 'output2.pt', 'output3.pt', 'output4.pt']")
        print("")
        print("for f in files:")
        print("    if os.path.exists(f):")
        print("        print('Converting {}...'.format(f))")
        print("        tensor = torch.load(f, map_location='cpu')")
        print("        array = tensor.numpy()")
        print("        np.save(f.replace('.pt', '.npy'), array)")
        print("        print('Saved: {} (shape: {})'.format(f.replace('.pt', '.npy'), array.shape))")
        print("\"")
    
    def load_embeddings(self, sample_frac=1.0):
        """Load embeddings from .npy files"""
        print("Loading embeddings from .npy files...")
        
        npy_pattern = os.path.join(self.embeddings_dir, "*.npy")
        npy_files = glob.glob(npy_pattern)
        
        if not npy_files:
            print("No .npy files found. Need to convert .pt files first.")
            self.provide_conversion_instructions()
            return {}
        
        for npy_file in npy_files:
            layer_name = os.path.basename(npy_file).replace('.npy', '')
            print("Loading {}...".format(layer_name))
            embeddings = np.load(npy_file)
            
            # Apply sampling if requested
            if sample_frac < 1.0:
                n_samples = int(len(embeddings) * sample_frac)
                indices = np.random.choice(len(embeddings), n_samples, replace=False)
                embeddings = embeddings[indices]
            
            self.layer_embeddings[layer_name] = embeddings
            print("  Loaded {} samples with {} dimensions".format(embeddings.shape[0], embeddings.shape[1]))
        
        print("Loaded {} layers".format(len(self.layer_embeddings)))
        return self.layer_embeddings
    
    def find_optimal_components(self, max_components=20):
        """Find optimal number of components using BIC/AIC"""
        print("Finding optimal components (max: {})...".format(max_components))
        
        for layer_name, embeddings in self.layer_embeddings.items():
            print("  Analyzing {}...".format(layer_name))
            
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
                    print("    Error with {} components: {}".format(n_components, e))
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
            
            print("    Optimal components - BIC: {}, AIC: {}, Silhouette: {}".format(optimal_bic, optimal_aic, optimal_sil))
        
        return self.optimal_components
    
    def fit_gmm_models(self, use_bic=True):
        """Fit GMM models with optimal components"""
        print("Fitting GMM models...")
        
        for layer_name, embeddings in self.layer_embeddings.items():
            print("  Fitting GMM for {}...".format(layer_name))
            
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
            
            print("    Fitted GMM with {} components".format(n_components))
        
        return self.gmm_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations including parallel coordinates"""
        print("Creating visualizations...")
        
        # 1. Component selection plots
        self._plot_component_selection()
        
        # 2. Layer-wise cluster visualizations
        self._plot_layer_clusters()
        
        # 3. Parallel coordinates plot
        self._plot_parallel_coordinates()
        
        print("Visualizations saved to: {}".format(self.output_dir))
    
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
        plt.savefig(os.path.join(self.output_dir, 'component_selection.png'), dpi=300, bbox_inches='tight')
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
            axes[0].set_title('{} - Clusters (PCA)'.format(layer_name))
            axes[0].set_xlabel('PC1 ({:.2%})'.format(pca.explained_variance_ratio_[0]))
            axes[0].set_ylabel('PC2 ({:.2%})'.format(pca.explained_variance_ratio_[1]))
            plt.colorbar(scatter, ax=axes[0])
            
            # Probability plot (max probability)
            max_probs = np.max(results['probabilities'], axis=1)
            scatter = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=max_probs, cmap='viridis', alpha=0.6)
            axes[1].set_title('{} - Max Probability'.format(layer_name))
            axes[1].set_xlabel('PC1 ({:.2%})'.format(pca.explained_variance_ratio_[0]))
            axes[1].set_ylabel('PC2 ({:.2%})'.format(pca.explained_variance_ratio_[1]))
            plt.colorbar(scatter, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, '{}_clusters.png'.format(layer_name)), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_parallel_coordinates(self):
        """Create parallel coordinates plot for layer-wise trajectory"""
        print("Creating parallel coordinates plot...")
        
        # Prepare data for parallel coordinates
        data_dict = {}
        for i, (layer_name, embeddings) in enumerate(self.layer_embeddings.items()):
            # Use first few principal components for visualization
            pca = PCA(n_components=min(3, embeddings.shape[1]))
            embeddings_pca = pca.fit_transform(embeddings)
            
            for j in range(embeddings_pca.shape[1]):
                data_dict['{}_PC{}'.format(layer_name, j+1)] = embeddings_pca[:, j]
        
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
        plt.savefig(os.path.join(self.output_dir, 'parallel_coordinates.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Parallel coordinates plot saved")
    
    def save_results(self):
        """Save analysis results to files"""
        print("Saving results...")
        
        # Save optimal components
        components_df = pd.DataFrame(self.optimal_components).T
        components_df.to_csv(os.path.join(self.output_dir, 'optimal_components.csv'))
        
        print("Results saved to: {}".format(self.output_dir))
    
    def run_full_analysis(self, max_components=20, sample_frac=1.0):
        """Run complete GMM analysis pipeline"""
        print("Starting clean GMM analysis...")
        
        # Check files first
        pt_files, npy_files = self.check_files()
        
        if not npy_files:
            print("No .npy files found. Cannot run analysis.")
            self.provide_conversion_instructions()
            return self
        
        # Load embeddings
        self.load_embeddings(sample_frac=sample_frac)
        
        if not self.layer_embeddings:
            print("No embeddings loaded. Exiting.")
            return self
        
        # Find optimal components
        self.find_optimal_components(max_components=max_components)
        
        # Fit GMM models
        self.fit_gmm_models(use_bic=True)
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        print("Analysis completed successfully!")
        return self

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Clean GMM analysis with parallel coordinates")
    parser.add_argument('--embeddings_dir', type=str, 
                       default="/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings",
                       help='Directory containing embeddings (.npy files)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default="clean_analysis",
                       help='Experiment name')
    parser.add_argument('--max_components', type=int, default=15,
                       help='Maximum number of components to test')
    parser.add_argument('--sample_frac', type=float, default=1.0,
                       help='Fraction of data to use (1.0 = use all)')
    
    args = parser.parse_args()
    
    print("Clean GMM Analysis with Parallel Coordinates")
    print("=" * 50)
    print("Embeddings directory: {}".format(args.embeddings_dir))
    print("Max components: {}".format(args.max_components))
    print("Sample fraction: {}".format(args.sample_frac))
    print("Experiment name: {}".format(args.experiment_name))
    
    analyzer = CleanGMMAnalyzer(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )
    
    analyzer.run_full_analysis(
        max_components=args.max_components,
        sample_frac=args.sample_frac
    )
    
    print("\nAnalysis completed! Results saved in: {}".format(analyzer.output_dir))
    
    # List generated files
    output_files = glob.glob(os.path.join(analyzer.output_dir, "*"))
    print("\nGenerated files:")
    for f in output_files:
        print("  - {}".format(os.path.basename(f)))

if __name__ == "__main__":
    main()
