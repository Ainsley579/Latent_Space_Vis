#!/usr/bin/env python3
"""
Generators for frontend data formats.

Converts analyzed embeddings to JSON formats expected by frontend:
1. ScatterGenerator: embeddings + clustering -> scatter JSON
2. SankeyGenerator: cluster transitions -> sankey edges
3. ClusterRepresentativesGenerator: cluster reps -> JSON
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from typing import List, Dict
except ImportError:
    pass


class ScatterGenerator(object):
    """Generate scatterplot JSON from embeddings, clustering, and 2D projections."""
    
    def __init__(self):
        pass
    
    def generate_csv(self, embeddings, layer_name, model_type, cluster_labels, 
                     coords_2d, output_path=None):
        """
        Generate scatter CSV row-by-row (for multiple layers).
        
        Args:
            embeddings: numpy array of embeddings
            layer_name: name of layer (emb, output_block_0, etc.)
            model_type: model name (dimenet10conf, dimenet_noaug)
            cluster_labels: cluster assignment per sample
            coords_2d: 2D coordinates (PCA/UMAP) of shape (n_samples, 2)
            output_path: optional path to save CSV
            
        Returns:
            pandas DataFrame with columns: real_id, layer_name, x, y, cluster_id
        """
        n_samples = len(embeddings)
        
        data = {
            'real_id': list(range(n_samples)),
            'layer_name': [layer_name] * n_samples,
            'model_type': [model_type] * n_samples,
            'x': coords_2d[:, 0].tolist(),
            'y': coords_2d[:, 1].tolist(),
            'cluster_id': cluster_labels.astype(int).tolist()
        }
        
        df = pd.DataFrame(data)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Saved scatter CSV to {output_path}")
        
        return df
    
    def generate_json(self, csv_rows, output_path):
        """
        Convert CSV rows to JSON format expected by frontend.
        
        Args:
            csv_rows: list of dicts from CSV
            output_path: path to save JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        json_data = []
        for row in csv_rows:
            json_data.append({
                'real_id': str(row['real_id']),
                'x': float(row['x']),
                'y': float(row['y']),
                'cluster_id': int(row['cluster_id'])
            })
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Saved JSON ({len(json_data)} points) to {output_path}")
        return json_data


class SankeyGenerator(object):
    """Generate Sankey transition data."""
    
    def __init__(self):
        pass
    
    def generate_transitions_csv(self, transitions, from_n_components, to_n_components,
                                  layer_name, output_path=None):
        """
        Generate transitions CSV for Sankey.
        
        Args:
            transitions: list of transition dicts from SankeyFlowAnalyzer
            from_n_components: number of components in source
            to_n_components: number of components in target
            layer_name: layer name for reference
            output_path: optional path to save CSV
            
        Returns:
            pandas DataFrame
        """
        # Update component counts in transitions
        for t in transitions:
            t['from_components'] = from_n_components
            t['to_components'] = to_n_components
        
        df = pd.DataFrame(transitions)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Saved transitions CSV ({len(df)} rows) to {output_path}")
        
        return df


class ClusterRepresentativesGenerator(object):
    """Generate cluster representatives JSON."""
    
    def __init__(self):
        pass
    
    def generate_json(self, representatives_dict, layer_name, model_type, output_path=None):
        """
        Generate cluster representatives JSON.
        
        Args:
            representatives_dict: output from ClusterRepresentativesExtractor.get_top_k()
            layer_name: layer name
            model_type: model name
            output_path: optional path to save JSON
            
        Returns:
            list of dicts
        """
        json_data = []
        
        for cluster_id, reps_info in representatives_dict.items():
            json_data.append({
                'layer_name': layer_name,
                'model_type': model_type,
                'cluster_id': int(cluster_id),
                'top_k_closest_real_ids': reps_info['top_k_closest'],
                'top_k_farthest_real_ids': reps_info['top_k_farthest'],
                'k': reps_info['k']
            })
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Saved cluster representatives JSON ({len(json_data)} clusters) to {output_path}")
        
        return json_data
