#!/usr/bin/env python3
"""
General Embedding Processing Pipeline

This pipeline accepts ANY embeddings and generates two output files:
1. Scatter plot data: id/index, x, y, color_value, layer
2. Sankey diagram data: real_id, from_layer, to_layer, from_cluster, to_cluster

Usage:
    pipeline = EmbeddingPipeline(
        embeddings={'layer1': array1, 'layer2': array2, ...},
        output_dir='./output',
        id_mapping={0: 100, 1: 101, ...},  # Optional: map embedding indices to real IDs
        n_components=13,  # Optional: number of GMM components
        color_values=array,  # Optional: values for coloring scatter plots
        metadata={'dataset': 'my_dataset', ...}  # Optional: additional metadata
    )
    pipeline.run()
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Optional, Union, List, Any

from embedding_pipeline_core import (
    load_embeddings_from_dict,
    fit_gmm,
    find_optimal_cluster_alignment,
    compute_mds_positions,
    map_embedding_indices_to_real_ids
)


class EmbeddingPipeline:
    """
    General pipeline for processing embeddings and generating visualization data.
    """
    
    def __init__(
        self,
        embeddings: Union[Dict[str, np.ndarray], Dict[str, Any]],
        output_dir: Union[str, Path],
        id_mapping: Optional[Dict[int, int]] = None,
        n_components: Optional[int] = None,
        zca_threshold: float = 1e-5,
        random_state: int = 42,
        color_values: Optional[np.ndarray] = None,
        layer_order: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_detect_components: bool = True
    ):
        """
        Initialize the embedding pipeline.
        
        Args:
            embeddings: Dictionary mapping layer names to embedding arrays/tensors
            output_dir: Directory to save output files
            id_mapping: Optional dictionary mapping embedding indices to real IDs
            n_components: Number of GMM components (auto-detect if None)
            zca_threshold: Threshold for ZCA whitening
            random_state: Random state for reproducibility
            color_values: Optional array of values for coloring scatter plots
            layer_order: Optional list specifying order of layers (default: sorted keys)
            metadata: Optional dictionary of additional metadata to save
            auto_detect_components: If True, automatically detect optimal n_components
        """
        # Process embeddings
        self.embeddings = load_embeddings_from_dict(embeddings)
        
        if not self.embeddings:
            raise ValueError("No valid embeddings provided")
        
        # Set layer order
        if layer_order is None:
            self.layer_order = sorted(self.embeddings.keys())
        else:
            # Validate layer order
            missing = set(layer_order) - set(self.embeddings.keys())
            if missing:
                raise ValueError(f"Layers in layer_order not found in embeddings: {missing}")
            self.layer_order = layer_order
        
        # Set output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store parameters
        self.id_mapping = id_mapping
        self.zca_threshold = zca_threshold
        self.random_state = random_state
        self.metadata = metadata or {}
        
        # Set color values
        n_samples = len(self.embeddings[self.layer_order[0]])
        if color_values is None:
            self.color_values = np.zeros(n_samples)
        else:
            if len(color_values) != n_samples:
                print(f"⚠ Warning: color_values length ({len(color_values)}) != n_samples ({n_samples})")
                print(f"  Using zeros for color values")
                self.color_values = np.zeros(n_samples)
            else:
                self.color_values = np.array(color_values)
        
        # Auto-detect or use provided n_components
        if n_components is None and auto_detect_components:
            print("Auto-detecting optimal number of components...")
            self.n_components = self._detect_optimal_components()
        elif n_components is None:
            raise ValueError("n_components must be provided if auto_detect_components=False")
        else:
            self.n_components = n_components
        
        print(f"Using {self.n_components} GMM components")
        
        # Storage for intermediate results
        self.gmm_results = {}
        self.mds_positions = {}
        self.scatter_data = None
        self.sankey_data = None
    
    def _detect_optimal_components(self, min_components=2, max_components=20):
        """Detect optimal number of components using AIC."""
        first_layer = self.layer_order[0]
        embeddings = self.embeddings[first_layer]
        
        best_aic = np.inf
        best_n = min_components
        
        for n in range(min_components, max_components + 1):
            try:
                gmm_result = fit_gmm(
                    embeddings,
                    n_components=n,
                    zca_threshold=self.zca_threshold,
                    random_state=self.random_state
                )
                
                # Calculate AIC
                log_likelihood = np.sum(gmm_result['gmm'].score_samples(gmm_result['embeddings_processed']))
                n_samples = embeddings.shape[0]
                n_features = embeddings.shape[1]
                n_params = n * (1 + n_features + n_features * (n_features + 1) / 2) - 1
                aic = 2 * n_params - 2 * log_likelihood
                
                if aic < best_aic:
                    best_aic = aic
                    best_n = n
            except Exception as e:
                print(f"  Warning: Could not fit {n} components: {e}")
                continue
        
        print(f"  Optimal components: {best_n} (AIC: {best_aic:.2f})")
        return best_n
    
    def fit_gmms(self):
        """Fit GMMs for all layers."""
        print("\nFitting GMMs for all layers...")
        self.gmm_results = {}
        
        for layer_name in self.layer_order:
            print(f"  Fitting GMM for {layer_name}...")
            embeddings = self.embeddings[layer_name]
            self.gmm_results[layer_name] = fit_gmm(
                embeddings,
                n_components=self.n_components,
                zca_threshold=self.zca_threshold,
                random_state=self.random_state
            )
        
        print(f"✓ Fitted GMMs for {len(self.gmm_results)} layers")
    
    def compute_mds(self):
        """Compute MDS positions for all layers."""
        print("\nComputing MDS positions...")
        self.mds_positions = compute_mds_positions(
            self.embeddings,
            n_components=2,
            random_state=self.random_state
        )
        print(f"✓ Computed MDS positions for {len(self.mds_positions)} layers")
    
    def generate_scatter_plot_data(self):
        """
        Generate scatter plot data: id/index, x, y, color_value, layer
        
        Returns:
            DataFrame with columns: id, x, y, color_value, layer
        """
        print("\nGenerating scatter plot data...")
        
        all_data = []
        n_samples = len(self.embeddings[self.layer_order[0]])
        
        for layer_name in self.layer_order:
            if layer_name not in self.mds_positions:
                print(f"  ⚠ Warning: No MDS positions for {layer_name}, skipping")
                continue
            
            pos = self.mds_positions[layer_name]
            
            # Map embedding indices to real IDs
            embedding_indices = np.arange(n_samples)
            real_ids = map_embedding_indices_to_real_ids(embedding_indices, self.id_mapping)
            
            for i in range(n_samples):
                all_data.append({
                    'id': int(real_ids[i]),
                    'x': float(pos[i, 0]),
                    'y': float(pos[i, 1]),
                    'color_value': float(self.color_values[i]),
                    'layer': layer_name
                })
        
        self.scatter_data = pd.DataFrame(all_data)
        print(f"✓ Generated scatter plot data: {len(self.scatter_data)} points")
        return self.scatter_data
    
    def generate_sankey_data(self):
        """
        Generate Sankey diagram data: real_id, from_layer, to_layer, from_cluster, to_cluster
        
        Returns:
            DataFrame with columns: real_id, from_layer, to_layer, from_cluster, to_cluster
        """
        print("\nGenerating Sankey diagram data...")
        
        all_data = []
        n_samples = len(self.embeddings[self.layer_order[0]])
        embedding_indices = np.arange(n_samples)
        real_ids = map_embedding_indices_to_real_ids(embedding_indices, self.id_mapping)
        
        # Generate transitions between consecutive layers
        for i in range(len(self.layer_order) - 1):
            from_layer = self.layer_order[i]
            to_layer = self.layer_order[i + 1]
            
            if from_layer not in self.gmm_results or to_layer not in self.gmm_results:
                print(f"  ⚠ Warning: Missing GMM results for {from_layer} -> {to_layer}, skipping")
                continue
            
            from_labels = self.gmm_results[from_layer]['labels']
            to_labels = self.gmm_results[to_layer]['labels']
            
            # Compute transition matrix for alignment
            transition_matrix = np.zeros((self.n_components, self.n_components))
            for prev_cluster in range(self.n_components):
                prev_mask = from_labels == prev_cluster
                if np.sum(prev_mask) > 0:
                    current_clusters = to_labels[prev_mask]
                    for curr_cluster in range(self.n_components):
                        transition_matrix[prev_cluster, curr_cluster] = np.sum(current_clusters == curr_cluster)
            
            # Align clusters
            permutation = find_optimal_cluster_alignment(transition_matrix, self.n_components)
            
            # Apply permutation to to_labels
            to_labels_aligned = permutation[to_labels]
            
            # Generate transition data for each point
            for embedding_idx in range(n_samples):
                all_data.append({
                    'real_id': int(real_ids[embedding_idx]),
                    'from_layer': from_layer,
                    'to_layer': to_layer,
                    'from_cluster': int(from_labels[embedding_idx]),
                    'to_cluster': int(to_labels_aligned[embedding_idx])
                })
        
        self.sankey_data = pd.DataFrame(all_data)
        print(f"✓ Generated Sankey diagram data: {len(self.sankey_data)} transitions")
        return self.sankey_data
    
    def save_outputs(self):
        """Save output files and metadata."""
        print("\nSaving output files...")
        
        # Save scatter plot data
        if self.scatter_data is not None:
            scatter_path = self.output_dir / 'scatter_plot_data.csv'
            self.scatter_data.to_csv(scatter_path, index=False)
            print(f"✓ Saved scatter plot data: {scatter_path}")
            print(f"  Columns: {list(self.scatter_data.columns)}")
            print(f"  Rows: {len(self.scatter_data)}")
        
        # Save Sankey diagram data
        if self.sankey_data is not None:
            sankey_path = self.output_dir / 'sankey_diagram_data.csv'
            self.sankey_data.to_csv(sankey_path, index=False)
            print(f"✓ Saved Sankey diagram data: {sankey_path}")
            print(f"  Columns: {list(self.sankey_data.columns)}")
            print(f"  Rows: {len(self.sankey_data)}")
        
        # Save metadata
        metadata = {
            'n_samples': len(self.embeddings[self.layer_order[0]]),
            'n_layers': len(self.layer_order),
            'layer_order': self.layer_order,
            'n_components': self.n_components,
            'zca_threshold': self.zca_threshold,
            'random_state': self.random_state,
            'has_id_mapping': self.id_mapping is not None,
            'id_mapping_size': len(self.id_mapping) if self.id_mapping else 0,
            **self.metadata
        }
        
        metadata_path = self.output_dir / 'pipeline_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")
        
        # Save ID mapping if provided
        if self.id_mapping is not None:
            mapping_path = self.output_dir / 'id_mapping.json'
            # Convert numpy types to native Python types for JSON
            mapping_dict = {int(k): int(v) for k, v in self.id_mapping.items()}
            with open(mapping_path, 'w') as f:
                json.dump(mapping_dict, f, indent=2)
            print(f"✓ Saved ID mapping: {mapping_path}")
    
    def run(self):
        """Run the complete pipeline."""
        print("="*60)
        print("Embedding Processing Pipeline")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Number of layers: {len(self.layer_order)}")
        print(f"Layers: {' -> '.join(self.layer_order)}")
        print(f"Number of components: {self.n_components}")
        print(f"Number of samples: {len(self.embeddings[self.layer_order[0]])}")
        print("="*60)
        
        # Step 1: Fit GMMs
        self.fit_gmms()
        
        # Step 2: Compute MDS positions
        self.compute_mds()
        
        # Step 3: Generate scatter plot data
        self.generate_scatter_plot_data()
        
        # Step 4: Generate Sankey diagram data
        self.generate_sankey_data()
        
        # Step 5: Save outputs
        self.save_outputs()
        
        print("\n" + "="*60)
        print("✅ Pipeline Complete")
        print("="*60)
        print(f"\nOutput files saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  1. scatter_plot_data.csv - For scatter plot visualization")
        print("     Columns: id, x, y, color_value, layer")
        print("  2. sankey_diagram_data.csv - For Sankey diagram visualization")
        print("     Columns: real_id, from_layer, to_layer, from_cluster, to_cluster")
        print("  3. pipeline_metadata.json - Pipeline configuration and metadata")
        if self.id_mapping:
            print("  4. id_mapping.json - Mapping from embedding indices to real IDs")

