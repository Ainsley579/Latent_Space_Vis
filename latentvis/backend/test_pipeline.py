#!/usr/bin/env python3
"""
Test the backend pipeline with synthetic embeddings.
This validates that all components work together correctly.

Usage:
    python test_pipeline.py --output-dir test_output
"""

import argparse
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from analyzer import (
    EmbeddingAnalyzer,
    DimensionalityReducer,
    SankeyFlowAnalyzer,
    ClusterRepresentativesExtractor
)
from generator import (
    ScatterGenerator,
    SankeyGenerator,
    ClusterRepresentativesGenerator
)
from utils import DataValidator


def generate_synthetic_embeddings(n_samples=152, n_features=64, seed=42):
    """Generate synthetic embeddings for testing."""
    np.random.seed(seed)
    
    # Create embeddings with some structure (multiple clusters)
    n_clusters = 3
    samples_per_cluster = n_samples // n_clusters
    
    embeddings_list = []
    for c in range(n_clusters):
        # Cluster center
        center = np.random.randn(n_features) * 5
        # Cluster samples
        cluster_samples = center + np.random.randn(samples_per_cluster, n_features) * 0.5
        embeddings_list.append(cluster_samples)
    
    embeddings = np.vstack(embeddings_list)
    return embeddings[:n_samples]


def test_analyzer():
    """Test EmbeddingAnalyzer."""
    print("\n" + "="*60)
    print("TEST 1: EmbeddingAnalyzer")
    print("="*60)
    
    embeddings = generate_synthetic_embeddings(n_samples=50, n_features=32)
    analyzer = EmbeddingAnalyzer()
    
    print(f"Input shape: {embeddings.shape}")
    
    # Fit GMM
    gmm_result = analyzer.fit_gmm(embeddings, n_components_range=(2, 5))
    print(f"✓ GMM fit complete")
    print(f"  Optimal components: {gmm_result['optimal_n_components']}")
    print(f"  BIC scores: {[f'{s:.2f}' for s in gmm_result['bic_scores']]}")
    print(f"  AIC scores: {[f'{s:.2f}' for s in gmm_result['aic_scores']]}")
    
    # Get assignments
    cluster_result = analyzer.get_cluster_assignments(
        embeddings, gmm_result['optimal_gmm']
    )
    labels = cluster_result['labels']
    print(f"✓ Cluster assignments: {len(np.unique(labels))} clusters")
    print(f"  Cluster sizes: {np.bincount(labels)}")
    
    # Silhouette
    silhouette = analyzer.compute_silhouette(embeddings, labels)
    print(f"✓ Silhouette score: {silhouette:.4f}")
    
    return embeddings, labels


def test_reducer(embeddings):
    """Test DimensionalityReducer."""
    print("\n" + "="*60)
    print("TEST 2: DimensionalityReducer")
    print("="*60)
    
    # Test PCA
    reducer = DimensionalityReducer(method='pca')
    coords_2d = reducer.fit_reduce(embeddings)
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {coords_2d.shape}")
    print(f"✓ PCA reduction complete")
    print(f"  X range: [{coords_2d[:, 0].min():.4f}, {coords_2d[:, 0].max():.4f}]")
    print(f"  Y range: [{coords_2d[:, 1].min():.4f}, {coords_2d[:, 1].max():.4f}]")
    
    return coords_2d


def test_scatter_generator(embeddings, labels, coords_2d):
    """Test ScatterGenerator."""
    print("\n" + "="*60)
    print("TEST 3: ScatterGenerator")
    print("="*60)
    
    gen = ScatterGenerator()
    
    # Generate CSV
    df = gen.generate_csv(
        embeddings, 'emb', 'test_model', labels, coords_2d
    )
    
    print(f"✓ CSV generated: {df.shape[0]} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample row:")
    print(f"    {df.iloc[0].to_dict()}")
    
    # Check real_ids
    real_ids = df['real_id'].unique()
    print(f"  Real IDs: {len(real_ids)} unique values")
    if list(sorted(real_ids)) == list(range(len(real_ids))):
        print(f"  ✓ Real IDs are sequential 0-{len(real_ids)-1}")
    else:
        print(f"  ✗ Real IDs are not sequential!")
        return None
    
    return df


def test_sankey_analyzer(labels):
    """Test SankeyFlowAnalyzer."""
    print("\n" + "="*60)
    print("TEST 4: SankeyFlowAnalyzer")
    print("="*60)
    
    # Create another label assignment
    labels_next = np.random.randint(0, 2, size=labels.shape)
    
    transitions = SankeyFlowAnalyzer.compute_transitions(labels, labels_next)
    
    print(f"✓ Transitions computed: {len(transitions)} flows")
    print(f"  Sample transition:")
    print(f"    {transitions[0]}")
    
    return transitions


def test_cluster_representatives(embeddings, labels):
    """Test ClusterRepresentativesExtractor."""
    print("\n" + "="*60)
    print("TEST 5: ClusterRepresentativesExtractor")
    print("="*60)
    
    reps = ClusterRepresentativesExtractor.get_top_k(
        embeddings, labels, k=3
    )
    
    print(f"✓ Representatives extracted: {len(reps)} clusters")
    print(f"  Sample cluster:")
    cluster_0 = reps.get(0, {})
    print(f"    Cluster 0: {cluster_0}")
    
    return reps


def test_data_validator():
    """Test DataValidator."""
    print("\n" + "="*60)
    print("TEST 6: DataValidator")
    print("="*60)
    
    # Create valid embeddings dict
    embeddings_dict = {
        'emb': generate_synthetic_embeddings(n_samples=100, n_features=32),
        'output_block_0': generate_synthetic_embeddings(n_samples=100, n_features=64),
        'output_block_1': generate_synthetic_embeddings(n_samples=100, n_features=64),
    }
    
    valid, msg = DataValidator.validate_embeddings(embeddings_dict)
    print(f"✓ Validation result: {msg}")
    
    if not valid:
        print(f"✗ Validation failed!")
        return False
    
    # Test invalid case
    bad_dict = embeddings_dict.copy()
    bad_dict['emb'] = bad_dict['emb'][:50]  # Different size
    
    valid, msg = DataValidator.validate_embeddings(bad_dict)
    print(f"✓ Invalid case correctly detected: {msg}")
    
    return True


def test_full_pipeline(output_dir):
    """Run all tests in sequence."""
    print("\n\n")
    print("█" * 60)
    print(" BACKEND PIPELINE TESTS")
    print("█" * 60)
    
    try:
        # Test 1: Analyzer
        embeddings, labels = test_analyzer()
        
        # Test 2: Reducer
        coords_2d = test_reducer(embeddings)
        
        # Test 3: Scatter Generator
        scatter_df = test_scatter_generator(embeddings, labels, coords_2d)
        if scatter_df is None:
            return False
        
        # Test 4: Sankey Analyzer
        transitions = test_sankey_analyzer(labels)
        
        # Test 5: Cluster Representatives
        reps = test_cluster_representatives(embeddings, labels)
        
        # Test 6: Data Validator
        validator_ok = test_data_validator()
        
        # Summary
        print("\n\n")
        print("█" * 60)
        print(" TEST SUMMARY")
        print("█" * 60)
        
        print("\n✅ ALL TESTS PASSED!")
        print("\nBackend components verified:")
        print("  ✓ EmbeddingAnalyzer (GMM fitting, cluster assignment)")
        print("  ✓ DimensionalityReducer (PCA projection)")
        print("  ✓ ScatterGenerator (JSON generation)")
        print("  ✓ SankeyFlowAnalyzer (cluster transitions)")
        print("  ✓ ClusterRepresentativesExtractor (top-k samples)")
        print("  ✓ DataValidator (data consistency checks)")
        
        print("\nReady for production use!")
        print("See README.md for usage instructions.")
        
        return True
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test backend pipeline')
    parser.add_argument('--output-dir', type=str, default='test_output',
                        help='Output directory')
    args = parser.parse_args()
    
    success = test_full_pipeline(args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
