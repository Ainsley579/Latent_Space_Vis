#!/usr/bin/env python3
"""
Tests for the embedding processing pipeline.

Run with: pytest test_embedding_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil
import json

# Import modules to test
from embedding_pipeline_core import (
    load_embeddings_from_dict,
    load_embeddings_from_files,
    adaptive_zca_whitening,
    fit_gmm,
    find_optimal_cluster_alignment,
    compute_mds_positions,
    map_embedding_indices_to_real_ids
)
from embedding_pipeline import EmbeddingPipeline


class TestCoreFunctions:
    """Test core utility functions"""
    
    def test_load_embeddings_from_dict_numpy(self):
        """Test loading embeddings from numpy arrays"""
        embeddings = {
            'layer1': np.random.randn(100, 50),
            'layer2': np.random.randn(100, 30)
        }
        result = load_embeddings_from_dict(embeddings)
        
        assert len(result) == 2
        assert 'layer1' in result
        assert 'layer2' in result
        assert result['layer1'].shape == (100, 50)
        assert result['layer2'].shape == (100, 30)
    
    def test_load_embeddings_from_dict_torch(self):
        """Test loading embeddings from torch tensors"""
        embeddings = {
            'layer1': torch.randn(100, 50),
            'layer2': torch.randn(100, 30)
        }
        result = load_embeddings_from_dict(embeddings)
        
        assert len(result) == 2
        assert isinstance(result['layer1'], np.ndarray)
        assert result['layer1'].shape == (100, 50)
    
    def test_load_embeddings_from_dict_reshape(self):
        """Test reshaping multi-dimensional arrays"""
        embeddings = {
            'layer1': np.random.randn(100, 10, 5)  # 3D array
        }
        result = load_embeddings_from_dict(embeddings)
        
        assert result['layer1'].shape == (100, 50)  # Flattened to 2D
    
    def test_adaptive_zca_whitening(self):
        """Test ZCA whitening"""
        X = np.random.randn(100, 50)
        X_whitened, zca_transform = adaptive_zca_whitening(X, threshold=1e-5)
        
        assert X_whitened.shape == X.shape
        assert zca_transform.shape[1] <= X.shape[1]  # May reduce dimensions
        
        # Check that whitening decorrelates (covariance should be closer to identity)
        cov = np.cov(X_whitened, rowvar=False)
        # Diagonal should be close to 1, off-diagonal should be small
        np.fill_diagonal(cov, 0)
        assert np.abs(cov).max() < 1.0  # Off-diagonal should be small
    
    def test_fit_gmm(self):
        """Test GMM fitting"""
        # Create data with clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(50, 10) + np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        cluster2 = np.random.randn(50, 10) - np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        embeddings = np.vstack([cluster1, cluster2])
        
        result = fit_gmm(embeddings, n_components=2, random_state=42)
        
        assert 'gmm' in result
        assert 'labels' in result
        assert 'probabilities' in result
        assert len(result['labels']) == 100
        assert result['probabilities'].shape == (100, 2)
        assert len(result['labels']) == embeddings.shape[0]
        
        # Check that labels are reasonable (0 or 1)
        assert set(result['labels']) == {0, 1}
    
    def test_find_optimal_cluster_alignment(self):
        """Test cluster alignment using Hungarian algorithm"""
        n_components = 3
        
        # Create a transition matrix where clusters are shifted
        transition_matrix = np.array([
            [0.1, 0.8, 0.1],  # Cluster 0 mostly goes to cluster 1
            [0.1, 0.1, 0.8],  # Cluster 1 mostly goes to cluster 2
            [0.8, 0.1, 0.1]   # Cluster 2 mostly goes to cluster 0
        ])
        
        permutation = find_optimal_cluster_alignment(transition_matrix, n_components)
        
        assert len(permutation) == n_components
        assert set(permutation) == set(range(n_components))  # Should be a valid permutation
    
    def test_compute_mds_positions(self):
        """Test MDS position computation"""
        embeddings = {
            'layer1': np.random.randn(50, 20),
            'layer2': np.random.randn(50, 20)
        }
        
        positions = compute_mds_positions(embeddings, n_components=2, random_state=42)
        
        assert len(positions) == 2
        assert 'layer1' in positions
        assert 'layer2' in positions
        assert positions['layer1'].shape == (50, 2)
        assert positions['layer2'].shape == (50, 2)
    
    def test_map_embedding_indices_to_real_ids(self):
        """Test ID mapping"""
        embedding_indices = np.array([0, 1, 2, 3, 4])
        id_mapping = {0: 100, 1: 101, 2: 102, 3: 103, 4: 104}
        
        real_ids = map_embedding_indices_to_real_ids(embedding_indices, id_mapping)
        
        assert len(real_ids) == len(embedding_indices)
        assert real_ids[0] == 100
        assert real_ids[1] == 101
        assert real_ids[4] == 104
    
    def test_map_embedding_indices_no_mapping(self):
        """Test ID mapping when no mapping is provided"""
        embedding_indices = np.array([0, 1, 2, 3, 4])
        
        real_ids = map_embedding_indices_to_real_ids(embedding_indices, None)
        
        assert np.array_equal(real_ids, embedding_indices)


class TestEmbeddingPipeline:
    """Test the main EmbeddingPipeline class"""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing"""
        np.random.seed(42)
        return {
            'emb': np.random.randn(100, 50),
            'output_block_0': np.random.randn(100, 40),
            'output_block_1': np.random.randn(100, 30)
        }
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_pipeline_initialization(self, sample_embeddings, temp_output_dir):
        """Test pipeline initialization"""
        pipeline = EmbeddingPipeline(
            embeddings=sample_embeddings,
            output_dir=temp_output_dir,
            n_components=3
        )
        
        assert len(pipeline.embeddings) == 3
        assert len(pipeline.layer_order) == 3
        assert pipeline.n_components == 3
        assert pipeline.output_dir == temp_output_dir
    
    def test_pipeline_auto_detect_components(self, sample_embeddings, temp_output_dir):
        """Test automatic component detection"""
        pipeline = EmbeddingPipeline(
            embeddings=sample_embeddings,
            output_dir=temp_output_dir,
            n_components=None,
            auto_detect_components=True
        )
        
        assert pipeline.n_components is not None
        assert pipeline.n_components >= 2
    
    def test_pipeline_with_id_mapping(self, sample_embeddings, temp_output_dir):
        """Test pipeline with ID mapping"""
        id_mapping = {i: i + 1000 for i in range(100)}
        
        pipeline = EmbeddingPipeline(
            embeddings=sample_embeddings,
            output_dir=temp_output_dir,
            n_components=3,
            id_mapping=id_mapping
        )
        
        assert pipeline.id_mapping == id_mapping
    
    def test_pipeline_with_color_values(self, sample_embeddings, temp_output_dir):
        """Test pipeline with color values"""
        color_values = np.random.randn(100)
        
        pipeline = EmbeddingPipeline(
            embeddings=sample_embeddings,
            output_dir=temp_output_dir,
            n_components=3,
            color_values=color_values
        )
        
        assert len(pipeline.color_values) == 100
        assert np.array_equal(pipeline.color_values, color_values)
    
    def test_pipeline_fit_gmms(self, sample_embeddings, temp_output_dir):
        """Test GMM fitting"""
        pipeline = EmbeddingPipeline(
            embeddings=sample_embeddings,
            output_dir=temp_output_dir,
            n_components=3
        )
        
        pipeline.fit_gmms()
        
        assert len(pipeline.gmm_results) == 3
        for layer in sample_embeddings.keys():
            assert layer in pipeline.gmm_results
            assert 'labels' in pipeline.gmm_results[layer]
            assert len(pipeline.gmm_results[layer]['labels']) == 100
    
    def test_pipeline_compute_mds(self, sample_embeddings, temp_output_dir):
        """Test MDS computation"""
        pipeline = EmbeddingPipeline(
            embeddings=sample_embeddings,
            output_dir=temp_output_dir,
            n_components=3
        )
        
        pipeline.compute_mds()
        
        assert len(pipeline.mds_positions) == 3
        for layer in sample_embeddings.keys():
            assert layer in pipeline.mds_positions
            assert pipeline.mds_positions[layer].shape == (100, 2)
    
    def test_pipeline_generate_scatter_plot_data(self, sample_embeddings, temp_output_dir):
        """Test scatter plot data generation"""
        pipeline = EmbeddingPipeline(
            embeddings=sample_embeddings,
            output_dir=temp_output_dir,
            n_components=3
        )
        
        pipeline.compute_mds()
        scatter_data = pipeline.generate_scatter_plot_data()
        
        assert scatter_data is not None
        assert isinstance(scatter_data, pd.DataFrame)
        assert 'id' in scatter_data.columns
        assert 'x' in scatter_data.columns
        assert 'y' in scatter_data.columns
        assert 'color_value' in scatter_data.columns
        assert 'layer' in scatter_data.columns
        assert len(scatter_data) == 100 * 3  # 100 samples * 3 layers
    
    def test_pipeline_generate_sankey_data(self, sample_embeddings, temp_output_dir):
        """Test Sankey diagram data generation"""
        pipeline = EmbeddingPipeline(
            embeddings=sample_embeddings,
            output_dir=temp_output_dir,
            n_components=3
        )
        
        pipeline.fit_gmms()
        sankey_data = pipeline.generate_sankey_data()
        
        assert sankey_data is not None
        assert isinstance(sankey_data, pd.DataFrame)
        assert 'real_id' in sankey_data.columns
        assert 'from_layer' in sankey_data.columns
        assert 'to_layer' in sankey_data.columns
        assert 'from_cluster' in sankey_data.columns
        assert 'to_cluster' in sankey_data.columns
        # Should have transitions between 2 pairs of layers
        assert len(sankey_data) == 100 * 2  # 100 samples * 2 transitions
    
    def test_pipeline_full_run(self, sample_embeddings, temp_output_dir):
        """Test complete pipeline run"""
        id_mapping = {i: i + 1000 for i in range(100)}
        color_values = np.random.randn(100)
        metadata = {'test': 'value', 'dataset': 'test_dataset'}
        
        pipeline = EmbeddingPipeline(
            embeddings=sample_embeddings,
            output_dir=temp_output_dir,
            n_components=3,
            id_mapping=id_mapping,
            color_values=color_values,
            metadata=metadata
        )
        
        pipeline.run()
        
        # Check output files exist
        scatter_path = temp_output_dir / 'scatter_plot_data.csv'
        sankey_path = temp_output_dir / 'sankey_diagram_data.csv'
        metadata_path = temp_output_dir / 'pipeline_metadata.json'
        mapping_path = temp_output_dir / 'id_mapping.json'
        
        assert scatter_path.exists()
        assert sankey_path.exists()
        assert metadata_path.exists()
        assert mapping_path.exists()
        
        # Check scatter plot data
        scatter_df = pd.read_csv(scatter_path)
        assert len(scatter_df) > 0
        assert 'id' in scatter_df.columns
        assert scatter_df['id'].min() >= 1000  # Should use mapped IDs
        
        # Check Sankey data
        sankey_df = pd.read_csv(sankey_path)
        assert len(sankey_df) > 0
        assert 'real_id' in sankey_df.columns
        assert sankey_df['real_id'].min() >= 1000  # Should use mapped IDs
        
        # Check metadata
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
        assert saved_metadata['n_samples'] == 100
        assert saved_metadata['n_layers'] == 3
        assert saved_metadata['test'] == 'value'
        assert saved_metadata['dataset'] == 'test_dataset'
        
        # Check ID mapping
        with open(mapping_path, 'r') as f:
            saved_mapping = json.load(f)
        assert saved_mapping['0'] == 1000
        assert saved_mapping['99'] == 1099
    
    def test_pipeline_with_custom_layer_order(self, sample_embeddings, temp_output_dir):
        """Test pipeline with custom layer order"""
        custom_order = ['output_block_1', 'emb', 'output_block_0']
        
        pipeline = EmbeddingPipeline(
            embeddings=sample_embeddings,
            output_dir=temp_output_dir,
            n_components=3,
            layer_order=custom_order
        )
        
        assert pipeline.layer_order == custom_order
    
    def test_pipeline_invalid_layer_order(self, sample_embeddings, temp_output_dir):
        """Test pipeline with invalid layer order"""
        invalid_order = ['nonexistent_layer']
        
        with pytest.raises(ValueError):
            EmbeddingPipeline(
                embeddings=sample_embeddings,
                output_dir=temp_output_dir,
                n_components=3,
                layer_order=invalid_order
            )
    
    def test_pipeline_empty_embeddings(self, temp_output_dir):
        """Test pipeline with empty embeddings"""
        with pytest.raises(ValueError):
            EmbeddingPipeline(
                embeddings={},
                output_dir=temp_output_dir,
                n_components=3
            )
    
    def test_pipeline_mismatched_color_values(self, sample_embeddings, temp_output_dir):
        """Test pipeline with mismatched color values length"""
        color_values = np.random.randn(50)  # Wrong length
        
        pipeline = EmbeddingPipeline(
            embeddings=sample_embeddings,
            output_dir=temp_output_dir,
            n_components=3,
            color_values=color_values
        )
        
        # Should use zeros instead
        assert len(pipeline.color_values) == 100
        assert np.all(pipeline.color_values == 0)


class TestLoadEmbeddingsFromFiles:
    """Test loading embeddings from files"""
    
    @pytest.fixture
    def temp_embeddings_dir(self):
        """Create temporary directory with embedding files"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create sample embedding files
        embeddings = {
            'layer1': np.random.randn(100, 50),
            'layer2': np.random.randn(100, 30)
        }
        
        for layer_name, data in embeddings.items():
            filepath = temp_path / f'{layer_name}.pt'
            torch.save(torch.from_numpy(data), filepath)
        
        yield temp_path
        
        shutil.rmtree(temp_dir)
    
    def test_load_embeddings_from_files(self, temp_embeddings_dir):
        """Test loading embeddings from files"""
        layer_files = {
            'layer1': 'layer1.pt',
            'layer2': 'layer2.pt'
        }
        
        embeddings = load_embeddings_from_files(temp_embeddings_dir, layer_files)
        
        assert len(embeddings) == 2
        assert 'layer1' in embeddings
        assert 'layer2' in embeddings
        assert embeddings['layer1'].shape == (100, 50)
        assert embeddings['layer2'].shape == (100, 30)
    
    def test_load_embeddings_from_files_missing(self, temp_embeddings_dir):
        """Test loading embeddings when some files are missing"""
        layer_files = {
            'layer1': 'layer1.pt',
            'layer2': 'layer2.pt',
            'layer3': 'nonexistent.pt'  # This doesn't exist
        }
        
        embeddings = load_embeddings_from_files(temp_embeddings_dir, layer_files)
        
        # Should only load existing files
        assert len(embeddings) == 2
        assert 'layer1' in embeddings
        assert 'layer2' in embeddings
        assert 'layer3' not in embeddings


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zca_whitening_small_data(self):
        """Test ZCA whitening with very small dataset"""
        X = np.random.randn(10, 5)
        X_whitened, _ = adaptive_zca_whitening(X, threshold=1e-5)
        
        assert X_whitened.shape == X.shape
    
    def test_fit_gmm_single_component(self):
        """Test GMM with single component"""
        embeddings = np.random.randn(50, 10)
        result = fit_gmm(embeddings, n_components=1, random_state=42)
        
        assert len(result['labels']) == 50
        assert np.all(result['labels'] == 0)  # All should be in cluster 0
    
    def test_fit_gmm_many_components(self):
        """Test GMM with many components"""
        embeddings = np.random.randn(20, 10)
        result = fit_gmm(embeddings, n_components=10, random_state=42)
        
        assert len(result['labels']) == 20
        assert len(set(result['labels'])) <= 10  # At most 10 unique clusters
    
    def test_mds_small_data(self):
        """Test MDS with small dataset"""
        embeddings = {
            'layer1': np.random.randn(5, 10)
        }
        
        positions = compute_mds_positions(embeddings, n_components=2, random_state=42)
        
        assert positions['layer1'].shape == (5, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

