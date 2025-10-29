#!/usr/bin/env python
"""
Run GMM analysis - Python 2.7 compatible
"""

import os
import sys
import glob
import argparse

def check_files(embeddings_dir):
    """Check what files are available"""
    
    pt_pattern = os.path.join(embeddings_dir, "*.pt")
    npy_pattern = os.path.join(embeddings_dir, "*.npy")
    
    pt_files = glob.glob(pt_pattern)
    npy_files = glob.glob(npy_pattern)
    
    print("Checking embeddings directory: {}".format(embeddings_dir))
    print("Found {} .pt files".format(len(pt_files)))
    print("Found {} .npy files".format(len(npy_files)))
    
    if pt_files:
        print("\n.pt files found:")
        for f in pt_files:
            print("  - {}".format(os.path.basename(f)))
    
    if npy_files:
        print("\n.npy files found:")
        for f in npy_files:
            print("  - {}".format(os.path.basename(f)))
    
    return pt_files, npy_files

def provide_conversion_instructions(embeddings_dir):
    """Provide instructions for converting .pt files"""
    print("\n" + "="*60)
    print("CONVERSION INSTRUCTIONS")
    print("="*60)
    print("\nTo convert your .pt files to .npy files, run this command:")
    print("\ncd {}".format(embeddings_dir))
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
    
    print("\nOr use this one-liner:")
    print("python -c \"import torch; import numpy as np; [np.save(f.replace('.pt', '.npy'), torch.load(f, map_location='cpu').numpy()) for f in ['emb.pt', 'output0.pt', 'output1.pt', 'output2.pt', 'output3.pt', 'output4.pt'] if __import__('os').path.exists(f)]\"")

def run_simple_analysis(embeddings_dir, experiment_name="test"):
    """Run a simple analysis if .npy files are available"""
    npy_pattern = os.path.join(embeddings_dir, "*.npy")
    npy_files = glob.glob(npy_pattern)
    
    if not npy_files:
        print("No .npy files found. Cannot run analysis.")
        return False
    
    print("\nFound .npy files! Running simple analysis...")
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.mixture import GaussianMixture
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Create output directory
        output_dir = "simple_gmm_results_{}".format(experiment_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load embeddings
        embeddings_data = {}
        for npy_file in npy_files:
            layer_name = os.path.basename(npy_file).replace('.npy', '')
            print("Loading {}...".format(layer_name))
            embeddings = np.load(npy_file)
            embeddings_data[layer_name] = embeddings
            print("  Shape: {}".format(embeddings.shape))
        
        # Run GMM analysis on first layer as example
        if embeddings_data:
            first_layer = embeddings_data.keys()[0]
            embeddings = embeddings_data[first_layer]
            
            print("\nRunning GMM analysis on {}...".format(first_layer))
            
            # Use subset for faster analysis
            if len(embeddings) > 1000:
                indices = np.random.choice(len(embeddings), 1000, replace=False)
                embeddings_sample = embeddings[indices]
            else:
                embeddings_sample = embeddings
            
            # Standardize
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings_sample)
            
            # Fit GMM
            gmm = GaussianMixture(n_components=5, random_state=42)
            gmm.fit(embeddings_scaled)
            
            # Get predictions
            labels = gmm.predict(embeddings_scaled)
            probabilities = gmm.predict_proba(embeddings_scaled)
            
            print("  Fitted GMM with 5 components")
            print("  BIC score: {:.2f}".format(gmm.bic(embeddings_scaled)))
            print("  AIC score: {:.2f}".format(gmm.aic(embeddings_scaled)))
            
            # Create visualization
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings_scaled)
            
            plt.figure(figsize=(12, 5))
            
            # Cluster plot
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=labels, cmap='tab10', alpha=0.6)
            plt.title('{} - Clusters (PCA)'.format(first_layer))
            plt.xlabel('PC1 ({:.2%})'.format(pca.explained_variance_ratio_[0]))
            plt.ylabel('PC2 ({:.2%})'.format(pca.explained_variance_ratio_[1]))
            plt.colorbar(scatter)
            
            # Probability plot
            plt.subplot(1, 2, 2)
            max_probs = np.max(probabilities, axis=1)
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=max_probs, cmap='viridis', alpha=0.6)
            plt.title('{} - Max Probability'.format(first_layer))
            plt.xlabel('PC1 ({:.2%})'.format(pca.explained_variance_ratio_[0]))
            plt.ylabel('PC2 ({:.2%})'.format(pca.explained_variance_ratio_[1]))
            plt.colorbar(scatter)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '{}_clusters.png'.format(first_layer)), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("  Visualization saved: {}_clusters.png".format(first_layer))
            
            # Create parallel coordinates plot
            if len(embeddings_data) > 1:
                print("\nCreating parallel coordinates plot...")
                
                # Prepare data
                data_dict = {}
                for layer_name, emb in embeddings_data.items():
                    pca = PCA(n_components=min(2, emb.shape[1]))
                    emb_pca = pca.fit_transform(emb)
                    
                    for j in range(emb_pca.shape[1]):
                        data_dict['{}_PC{}'.format(layer_name, j+1)] = emb_pca[:, j]
                
                # Sample data
                n_samples = min(500, len(embeddings_data.values()[0]))
                sample_indices = np.random.choice(len(embeddings_data.values()[0]), n_samples, replace=False)
                
                plt.figure(figsize=(15, 8))
                
                # Plot parallel coordinates
                for i in sample_indices:
                    values = [data_dict[col][i] for col in data_dict.keys()]
                    plt.plot(range(len(values)), values, alpha=0.1, linewidth=0.5, color='blue')
                
                plt.xticks(range(len(data_dict.keys())), data_dict.keys(), rotation=45, ha='right')
                plt.ylabel('Value')
                plt.title('Layer-wise Trajectory: Parallel Coordinates Plot')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'parallel_coordinates.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print("  Parallel coordinates plot saved: parallel_coordinates.png")
        
        print("\nAnalysis completed successfully!")
        print("Results saved in: {}".format(output_dir))
        
        # List generated files
        output_files = glob.glob(os.path.join(output_dir, "*"))
        print("\nGenerated files:")
        for f in output_files:
            print("  - {}".format(os.path.basename(f)))
        
        return True
        
    except Exception as e:
        print("\nError during analysis: {}".format(e))
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run GMM analysis or provide conversion instructions")
    parser.add_argument('--embeddings_dir', type=str, 
                       default="/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings",
                       help='Directory containing embeddings')
    parser.add_argument('--experiment_name', type=str, default="test",
                       help='Experiment name')
    
    args = parser.parse_args()
    
    print("Checking embeddings directory...")
    pt_files, npy_files = check_files(args.embeddings_dir)
    
    if npy_files:
        print("\n.npy files found! Running analysis...")
        success = run_simple_analysis(args.embeddings_dir, args.experiment_name)
        
        if success:
            print("\nAnalysis completed successfully!")
        else:
            print("\nAnalysis failed. Check the error messages above.")
    else:
        print("\nNo .npy files found. Need to convert .pt files first.")
        provide_conversion_instructions(args.embeddings_dir)
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Convert your .pt files to .npy files using the command above")
        print("2. Run this script again: python run_gmm_py2.py")
        print("3. Or run: python run_gmm_py2.py --embeddings_dir {}".format(args.embeddings_dir))

if __name__ == "__main__":
    main()
