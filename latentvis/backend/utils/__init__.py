#!/usr/bin/env python3
"""
Utility functions for data standardization and I/O.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict


class EmbeddingLoader(object):
    """Load embeddings from various formats."""
    
    @staticmethod
    def load_numpy(path):
        """Load from .npy or .npz file."""
        path = Path(path)
        if path.suffix == '.npy':
            return np.load(path)
        elif path.suffix == '.npz':
            data = np.load(path)
            # Return first array if multiple exist
            if isinstance(data, np.lib.npyio.NpzFile):
                key = list(data.keys())[0]
                return data[key]
            return data
        else:
            raise ValueError(f"Unknown format: {path.suffix}")
    
    @staticmethod
    def load_pickle(path):
        """Load from pickle file (dict of layer->embeddings)."""
        import pickle
        path = Path(path)
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_embedding_dict(source_paths):
        """
        Load multiple embeddings files.
        
        Args:
            source_paths: dict mapping layer_name -> file_path
            
        Returns:
            dict mapping layer_name -> embeddings
        """
        embeddings = {}
        for layer_name, path in source_paths.items():
            embeddings[layer_name] = EmbeddingLoader.load_numpy(path)
        return embeddings


class DataValidator(object):
    """Validate data consistency across outputs."""
    
    LAYERS = [
        'emb', 'output_block_0', 'output_block_1', 'output_block_2',
        'output_block_3', 'output_block_4', 'u_mlp_input', 'u_mlp_l1', 'u_mlp_l2'
    ]
    
    @staticmethod
    def validate_embeddings(embeddings_dict):
        """Check that all embeddings have same number of samples."""
        if not embeddings_dict:
            return False, "No embeddings provided"
        
        n_samples_list = []
        for layer_name, emb in embeddings_dict.items():
            if len(emb.shape) != 2:
                return False, f"{layer_name}: expected 2D array, got shape {emb.shape}"
            n_samples_list.append((layer_name, emb.shape[0]))
        
        # Check consistency
        n_samples = n_samples_list[0][1]
        for layer_name, n in n_samples_list:
            if n != n_samples:
                return False, f"Inconsistent sample count: {layer_name} has {n}, expected {n_samples}"
        
        return True, f"Valid: {len(embeddings_dict)} layers, {n_samples} samples"
    
    @staticmethod
    def validate_scatter_json(json_path):
        """Validate scatterplot JSON structure."""
        json_path = Path(json_path)
        
        if not json_path.exists():
            return False, f"File not found: {json_path}"
        
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                return False, "Expected list"
            
            if len(data) == 0:
                return False, "Empty list"
            
            required_keys = {'real_id', 'x', 'y', 'cluster_id'}
            first = data[0]
            if not required_keys.issubset(first.keys()):
                return False, f"Missing keys: {required_keys - set(first.keys())}"
            
            # Check real_ids are sequential
            real_ids = sorted([int(p['real_id']) for p in data])
            expected = list(range(len(real_ids)))
            if real_ids != expected:
                return False, f"Non-sequential real_ids"
            
            return True, f"Valid: {len(data)} points with real_ids 0-{len(data)-1}"
        
        except Exception as e:
            return False, f"Error: {str(e)}"


class ConfigManager(object):
    """Manage configuration for data generation pipeline."""
    
    DEFAULT_CONFIG = {
        'n_components_range': [2, 10],
        'reduction_method': 'pca',
        'random_state': 42,
        'scatter_k_samples': 152,  # total expected samples
        'cluster_k': 5  # top-k closest/farthest
    }
    
    @staticmethod
    def load_config(config_path):
        """Load config from JSON or YAML."""
        config_path = Path(config_path)
        
        if config_path.suffix == '.json':
            with open(config_path) as f:
                return json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path) as f:
                    return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required to load .yaml config")
        else:
            raise ValueError(f"Unknown config format: {config_path.suffix}")
    
    @staticmethod
    def merge_with_defaults(user_config):
        """Merge user config with defaults."""
        config = ConfigManager.DEFAULT_CONFIG.copy()
        config.update(user_config or {})
        return config
