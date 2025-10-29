#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Sankey diagrams showing data point flow between different numbers of components
for each layer embedding. Uses fixed + With ZCA approach with component counts 2-12.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# Add the current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import plotly for Sankey diagrams
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available, Sankey diagrams will be skipped")
    PLOTLY_AVAILABLE = False

class ComponentFlowAnalyzer:
    """Analyze data point flow between different numbers of components"""
    
    def __init__(self, embeddings_dir, output_dir, zca_threshold=1e-5):
        self.embeddings_dir = embeddings_dir
        self.output_dir = Path(output_dir)
        self.zca_threshold = zca_threshold
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load embeddings
        self.layer_embeddings = self._load_embeddings()
        
        # Component counts to test
        self.component_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
    def _load_embeddings(self):
        """Load all layer embeddings"""
        print("Loading embeddings...")
        embeddings_path = Path(self.embeddings_dir)
        
        # Define expected layer files and their names (prioritize _cas files)
        layer_mappings = {
            'emb': {'unique_cas': 'emb_unique_cas.pt', 'unique': 'emb_unique.pt', 'original': 'emb.pt'},
            'output_block_0': {'unique_cas': 'output0_unique_cas.pt', 'unique': 'output0_unique.pt', 'original': 'output0.pt'},
            'output_block_1': {'unique_cas': 'output1_unique_cas.pt', 'unique': 'output1_unique.pt', 'original': 'output1.pt'},
            'output_block_2': {'unique_cas': 'output2_unique_cas.pt', 'unique': 'output2_unique.pt', 'original': 'output2.pt'},
            'output_block_3': {'unique_cas': 'output3_unique_cas.pt', 'unique': 'output3_unique.pt', 'original': 'output3.pt'},
            'output_block_4': {'unique_cas': 'output4_unique_cas.pt', 'unique': 'output4_unique.pt', 'original': 'output4.pt'},
            'u_mlp_input': {'unique_cas': 'u_mlp_input_cas.pt', 'unique': 'u_mlp_input.pt', 'original': 'u_mlp_input.pt'},
            'u_mlp_l1': {'unique_cas': 'u_mlp_l1_cas.pt', 'unique': 'u_mlp_l1.pt', 'original': 'u_mlp_l1.pt'},
            'u_mlp_l2': {'unique_cas': 'u_mlp_l2_cas.pt', 'unique': 'u_mlp_l2.pt', 'original': 'u_mlp_l2.pt'}
        }
        
        layer_embeddings = {}
        
        for layer_name in ['emb', 'output_block_0', 'output_block_1', 'output_block_2', 'output_block_3', 'output_block_4', 
                           'u_mlp_input', 'u_mlp_l1', 'u_mlp_l2']:
            
            # Try unique_cas file first, then unique file, then fallback to original
            unique_cas_filename = layer_mappings[layer_name]['unique_cas']
            unique_filename = layer_mappings[layer_name]['unique']
            original_filename = layer_mappings[layer_name]['original']
            
            unique_cas_filepath = embeddings_path / unique_cas_filename
            unique_filepath = embeddings_path / unique_filename
            original_filepath = embeddings_path / original_filename
            
            if unique_cas_filepath.exists():
                filepath = unique_cas_filepath
                filename = unique_cas_filename
                print("  Loading {} -> {} (unique_cas)".format(filename, layer_name))
            elif unique_filepath.exists():
                filepath = unique_filepath
                filename = unique_filename
                print("  Loading {} -> {} (unique)".format(filename, layer_name))
            elif original_filepath.exists():
                filepath = original_filepath
                filename = original_filename
                print("  Loading {} -> {} (original)".format(filename, layer_name))
            else:
                print("  Warning: Neither {}, {}, nor {} found".format(unique_cas_filename, unique_filename, original_filename))
                continue
            
            try:
                import torch
                embeddings_tensor = torch.load(filepath, map_location='cpu')
                embeddings_np = embeddings_tensor.detach().cpu().numpy()
                
                if embeddings_np.ndim > 2:
                    embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)
                
                layer_embeddings[layer_name] = embeddings_np
                print("    Shape: {}".format(embeddings_np.shape))
                
            except Exception as e:
                print("    Error loading {}: {}".format(filename, e))
        
        print("Successfully loaded embeddings from {} layers".format(len(layer_embeddings)))
        return layer_embeddings
    
    def adaptive_zca_whitening(self, X, threshold=1e-5, epsilon=1e-5):
        """Apply ZCA whitening"""
        sigma = np.cov(X, rowvar=False)
        U, S, V = np.linalg.svd(sigma)
        
        k = 0
        for v in S:
            if v > threshold:
                k += 1
        
        top_U = U[:, :k]
        top_S = S[:k]
        red_zca = np.dot(np.diag(1.0 / np.sqrt(top_S + epsilon)), top_U.T)
        X_whitened = X @ red_zca.T
        
        return X_whitened, red_zca
    
    def fit_gmm_with_components(self, embeddings, n_components):
        """Fit GMM with specified number of components"""
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply ZCA whitening
        embeddings_processed, zca_transform = self.adaptive_zca_whitening(embeddings_scaled, threshold=self.zca_threshold)
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        gmm.fit(embeddings_processed)
        
        # Get cluster assignments
        labels = gmm.predict(embeddings_processed)
        
        return labels, gmm, scaler, zca_transform, embeddings_processed
    
    def create_sankey_for_layer(self, layer_name, embeddings):
        """Create Sankey diagram for a single layer with different component counts"""
        if not PLOTLY_AVAILABLE:
            print("Skipping Sankey diagram for {}: plotly not available".format(layer_name))
            return
        
        print("Creating Sankey diagram for layer: {}".format(layer_name))
        
        # Fit GMMs for different component counts
        gmm_results = {}
        for n_components in self.component_counts:
            print("  Fitting GMM with {} components...".format(n_components))
            labels, gmm, scaler, zca_transform, embeddings_processed = self.fit_gmm_with_components(embeddings, n_components)
            gmm_results[n_components] = {
                'labels': labels,
                'gmm': gmm,
                'scaler': scaler,
                'zca_transform': zca_transform,
                'embeddings_processed': embeddings_processed
            }
        
        # Create Sankey diagram showing flow between consecutive component counts
        self._create_component_flow_sankey(layer_name, gmm_results)
        
        # Create transition matrices for analysis
        self._create_transition_matrices(layer_name, gmm_results)
    
    def _create_component_flow_sankey(self, layer_name, gmm_results):
        """Create Sankey diagram showing flow between different component counts"""
        print("  Creating component flow Sankey diagram...")
        
        # Prepare data for Sankey diagram
        source_nodes = []
        target_nodes = []
        values = []
        labels = []
        
        # Create node labels for each component count
        node_id = 0
        component_node_mapping = {}
        
        for n_components in self.component_counts:
            component_node_mapping[n_components] = {}
            for comp_idx in range(n_components):
                node_label = "{} Components\nCluster {}".format(n_components, comp_idx)
                labels.append(node_label)
                component_node_mapping[n_components][comp_idx] = node_id
                node_id += 1
        
        # Calculate transitions between consecutive component counts
        for i in range(len(self.component_counts) - 1):
            current_n_components = self.component_counts[i]
            next_n_components = self.component_counts[i + 1]
            
            current_labels = gmm_results[current_n_components]['labels']
            next_labels = gmm_results[next_n_components]['labels']
            
            # Calculate transition counts
            for src_comp in range(current_n_components):
                src_mask = current_labels == src_comp
                if np.sum(src_mask) > 0:
                    next_clusters = next_labels[src_mask]
                    for dst_comp in range(next_n_components):
                        count = np.sum(next_clusters == dst_comp)
                        if count > 0:  # Only include non-zero transitions
                            source_nodes.append(component_node_mapping[current_n_components][src_comp])
                            target_nodes.append(component_node_mapping[next_n_components][dst_comp])
                            values.append(count)
        
        # Create color palette for different component counts
        colors = px.colors.qualitative.Set3
        node_colors = []
        for n_components in self.component_counts:
            layer_color = colors[n_components % len(colors)]
            for comp_idx in range(n_components):
                node_colors.append(layer_color)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=source_nodes,
                target=target_nodes,
                value=values,
                color='rgba(0,0,0,0.2)'  # Semi-transparent links
            )
        )])
        
        fig.update_layout(
            title_text="Data Point Flow Between Component Counts: {}<br><sub>Each flow represents data points moving between clusters as component count increases</sub>".format(layer_name),
            font_size=10,
            width=1400,
            height=800
        )
        
        # Save as HTML
        sankey_path = self.output_dir / 'sankey_component_flow_{}.html'.format(layer_name)
        fig.write_html(str(sankey_path))
        
        # Also save as static image if possible
        try:
            sankey_png_path = self.output_dir / 'sankey_component_flow_{}.png'.format(layer_name)
            fig.write_image(str(sankey_png_path), width=1400, height=800, scale=2)
            print("    Sankey diagram saved as PNG: {}".format(sankey_png_path))
        except Exception as e:
            print("    Could not save PNG version: {}".format(e))
        
        print("    Sankey diagram saved as HTML: {}".format(sankey_path))
    
    def _create_transition_matrices(self, layer_name, gmm_results):
        """Create transition matrices between consecutive component counts"""
        print("  Creating transition matrices...")
        
        transition_data = []
        
        for i in range(len(self.component_counts) - 1):
            current_n_components = self.component_counts[i]
            next_n_components = self.component_counts[i + 1]
            
            current_labels = gmm_results[current_n_components]['labels']
            next_labels = gmm_results[next_n_components]['labels']
            
            # Create transition matrix
            transition_matrix = np.zeros((current_n_components, next_n_components))
            for prev_cluster in range(current_n_components):
                prev_mask = current_labels == prev_cluster
                if np.sum(prev_mask) > 0:
                    next_clusters = next_labels[prev_mask]
                    for next_cluster in range(next_n_components):
                        transition_matrix[prev_cluster, next_cluster] = np.sum(next_clusters == next_cluster)
            
            # Normalize by row sums to get transition probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix_norm = transition_matrix / (row_sums + 1e-10)
            
            # Export as long format
            for prev_cluster in range(current_n_components):
                for next_cluster in range(next_n_components):
                    transition_data.append({
                        'layer': layer_name,
                        'from_components': current_n_components,
                        'to_components': next_n_components,
                        'from_cluster': prev_cluster,
                        'to_cluster': next_cluster,
                        'count': transition_matrix[prev_cluster, next_cluster],
                        'probability': transition_matrix_norm[prev_cluster, next_cluster]
                    })
        
        # Save transition data
        df_transitions = pd.DataFrame(transition_data)
        transitions_csv_path = self.output_dir / 'transitions_{}.csv'.format(layer_name)
        df_transitions.to_csv(transitions_csv_path, index=False)
        print("    Transition matrices saved to: {}".format(transitions_csv_path))
    
    def run_analysis_for_all_layers(self):
        """Run component flow analysis for all layers"""
        print("Starting component flow analysis for all layers...")
        
        for layer_name, embeddings in self.layer_embeddings.items():
            print("\n" + "="*60)
            print("Processing layer: {}".format(layer_name))
            print("="*60)
            
            try:
                self.create_sankey_for_layer(layer_name, embeddings)
                print("✅ {} completed successfully".format(layer_name))
                
            except Exception as e:
                print("❌ {} failed: {}".format(layer_name, e))
                import traceback
                traceback.print_exc()
        
        # Create summary
        self._create_summary_report()
        
        print("\n" + "="*60)
        print("All layer analyses completed!")
        print("Results saved in: {}".format(self.output_dir))
        print("="*60)
    
    def extract_gmm_centers_and_neighbors(self, layer_name, embeddings, n_components):
        """Extract GMM center data points and their nearest neighbors with no overlap across components"""
        print("  Extracting GMM centers and neighbors for {} components...".format(n_components))
        
        # Fit GMM
        labels, gmm, scaler, zca_transform, embeddings_processed = self.fit_gmm_with_components(embeddings, n_components)
        
        # Get GMM means (centers)
        gmm_means = gmm.means_
        
        # Transform means back to original space for nearest neighbor search
        means_scaled = gmm_means @ zca_transform
        means_original = scaler.inverse_transform(means_scaled)
        
        center_data = []
        used_indices = set()  # Track used data points to avoid overlap across components
        
        for comp_idx in range(n_components):
            # Get all data points assigned to this component
            component_mask = (labels == comp_idx)
            component_indices = np.where(component_mask)[0]
            component_embeddings = embeddings[component_indices]
            
            if len(component_indices) == 0:
                # Skip empty components
                continue
            
            # Find the data point closest to the GMM center (this is our "center data point")
            distances_to_center = np.linalg.norm(component_embeddings - means_original[comp_idx], axis=1)
            closest_member_idx = np.argmin(distances_to_center)
            center_data_point_idx = component_indices[closest_member_idx]
            
            # Store the center data point
            center_data.append({
                'layer': layer_name,
                'component': comp_idx,
                'component_weight': gmm.weights_[comp_idx],
                'center_coordinates': means_original[comp_idx].tolist(),
                'neighbor_rank': 0,  # Rank 0 for the center itself
                'neighbor_index': int(center_data_point_idx),
                'neighbor_coordinates': embeddings[center_data_point_idx].tolist(),
                'distance_to_center': float(distances_to_center[closest_member_idx]),
                'is_center': True,
                'component_size': len(component_indices),
                'n_components': n_components
            })
            
            # Mark this center as used
            used_indices.add(center_data_point_idx)
            
            # Find nearest neighbors within this component only
            # Sort component members by distance to center (excluding the center itself)
            member_distances = [(idx, dist) for idx, dist in zip(component_indices, distances_to_center) 
                              if idx != center_data_point_idx]
            member_distances.sort(key=lambda x: x[1])  # Sort by distance
            
            # Determine how many neighbors to select
            n_neighbors_needed = min(10, len(component_indices) - 1)  # -1 because we exclude the center
            
            # Select neighbors, ensuring no overlap with other components
            neighbors_selected = 0
            for member_idx, distance in member_distances:
                if neighbors_selected >= n_neighbors_needed:
                    break
                    
                # Only use this neighbor if it hasn't been used by another component
                if member_idx not in used_indices:
                    center_data.append({
                        'layer': layer_name,
                        'component': comp_idx,
                        'component_weight': gmm.weights_[comp_idx],
                        'center_coordinates': means_original[comp_idx].tolist(),
                        'neighbor_rank': neighbors_selected + 1,
                        'neighbor_index': int(member_idx),
                        'neighbor_coordinates': embeddings[member_idx].tolist(),
                        'distance_to_center': float(distance),
                        'is_center': False,
                        'component_size': len(component_indices),
                        'n_components': n_components
                    })
                    
                    used_indices.add(member_idx)
                    neighbors_selected += 1
            
            print("    Component {}: Selected {} points (1 center + {} neighbors)".format(
                comp_idx, neighbors_selected + 1, neighbors_selected))
        
        return center_data
    
    def run_centers_and_neighbors_analysis(self):
        """Run analysis to extract GMM centers and neighbors for different component counts"""
        print("Starting GMM centers and neighbors analysis...")
        
        all_center_data = []
        
        for layer_name, embeddings in self.layer_embeddings.items():
            print("\n" + "="*60)
            print("Processing layer: {}".format(layer_name))
            print("="*60)
            
            layer_center_data = []
            
            for n_components in self.component_counts:
                try:
                    center_data = self.extract_gmm_centers_and_neighbors(layer_name, embeddings, n_components)
                    layer_center_data.extend(center_data)
                    print("  ✅ {} components completed".format(n_components))
                    
                except Exception as e:
                    print("  ❌ {} components failed: {}".format(n_components, e))
                    continue
            
            all_center_data.extend(layer_center_data)
            
            # Save layer-specific results
            df_layer = pd.DataFrame(layer_center_data)
            layer_csv_path = self.output_dir / 'gmm_centers_and_neighbors_{}.csv'.format(layer_name)
            df_layer.to_csv(layer_csv_path, index=False)
            print("  Layer results saved to: {}".format(layer_csv_path))
        
        # Save all results
        df_all = pd.DataFrame(all_center_data)
        all_csv_path = self.output_dir / 'gmm_centers_and_neighbors_all_layers.csv'
        df_all.to_csv(all_csv_path, index=False)
        print("\nAll results saved to: {}".format(all_csv_path))
        
        # Create summary
        self._create_centers_summary(all_center_data)
        
        print("\n" + "="*60)
        print("GMM centers and neighbors analysis completed!")
        print("Results saved in: {}".format(self.output_dir))
        print("="*60)
    
    def _create_centers_summary(self, center_data):
        """Create summary of centers and neighbors analysis"""
        print("\nCreating centers and neighbors summary...")
        
        df = pd.DataFrame(center_data)
        
        summary_data = []
        
        for layer in df['layer'].unique():
            layer_data = df[df['layer'] == layer]
            
            for n_components in layer_data['n_components'].unique():
                comp_data = layer_data[layer_data['n_components'] == n_components]
                
                total_points = len(comp_data)
                unique_indices = len(set(comp_data['neighbor_index']))
                centers = len(comp_data[comp_data['is_center'] == True])
                
                summary_data.append({
                    'layer': layer,
                    'n_components': n_components,
                    'total_points': total_points,
                    'unique_indices': unique_indices,
                    'centers': centers,
                    'neighbors': total_points - centers,
                    'no_overlap': unique_indices == total_points
                })
        
        df_summary = pd.DataFrame(summary_data)
        summary_path = self.output_dir / 'centers_and_neighbors_summary.csv'
        df_summary.to_csv(summary_path, index=False)
        
        print("Summary saved to: {}".format(summary_path))
        
        # Print summary to console
        print("\n" + "="*60)
        print("CENTERS AND NEIGHBORS SUMMARY:")
        print("="*60)
        for _, row in df_summary.iterrows():
            print("{} ({} components): {} total points, {} unique indices, {} centers, {} neighbors".format(
                row['layer'], row['n_components'], row['total_points'], 
                row['unique_indices'], row['centers'], row['neighbors']))
            if row['no_overlap']:
                print("  ✅ No overlap detected")
            else:
                print("  ❌ Overlap detected: {} duplicate indices".format(row['total_points'] - row['unique_indices']))

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        embeddings_dir = sys.argv[1]
    else:
        embeddings_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings"
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzerResults/component_flow_analysis"
    
    # Check if user wants to run centers and neighbors analysis
    run_centers_analysis = len(sys.argv) > 3 and sys.argv[3] == "--centers"
    
    print("Starting Component Flow Analysis...")
    print("Embeddings directory: {}".format(embeddings_dir))
    print("Output directory: {}".format(output_dir))
    if run_centers_analysis:
        print("Mode: Centers and Neighbors Analysis")
    else:
        print("Mode: Sankey Diagram Analysis")
    
    # Create analyzer
    analyzer = ComponentFlowAnalyzer(
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        zca_threshold=1e-5
    )
    
    if run_centers_analysis:
        # Run centers and neighbors analysis
        analyzer.run_centers_and_neighbors_analysis()
        
        print("\n🎉 Centers and neighbors analysis completed!")
        print("📁 Results saved in: {}".format(output_dir))
        print("\n📊 Generated files:")
        print("  - gmm_centers_and_neighbors_[layer].csv (per layer)")
        print("  - gmm_centers_and_neighbors_all_layers.csv (combined)")
        print("  - centers_and_neighbors_summary.csv (summary)")
    else:
        # Run Sankey diagram analysis
        analyzer.run_analysis_for_all_layers()
        
        print("\n🎉 Component flow analysis completed!")
        print("📁 Results saved in: {}".format(output_dir))
        print("\n📊 Generated files for each layer:")
        print("  - sankey_component_flow_[layer].html (interactive Sankey diagram)")
        print("  - sankey_component_flow_[layer].png (static image)")
        print("  - transitions_[layer].csv (transition matrices)")
        print("  - component_flow_summary.csv (overall summary)")

if __name__ == "__main__":
    main()
