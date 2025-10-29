#!/usr/bin/env python3
"""
Test script for GNN GMM analysis

This script tests the basic functionality without running the full analysis.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Go up to latent-spaces directory
sys.path.append(parent_dir)

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        import pandas
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import seaborn
        print("✓ seaborn imported successfully")
    except ImportError as e:
        print(f"✗ seaborn import failed: {e}")
        return False
    
    return True

def test_model_setup():
    """Test model setup functionality"""
    print("\nTesting model setup...")
    
    try:
        from model_setup import dimenet10conf
        print("✓ model_setup imported successfully")
        
        # Test if we can create the function (don't actually load model)
        print("✓ dimenet10conf function available")
        
    except ImportError as e:
        print(f"✗ model_setup import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ model_setup test failed: {e}")
        return False
    
    return True

def test_gmm_analysis_class():
    """Test GMM analysis class instantiation"""
    print("\nTesting GMM analysis class...")
    
    try:
        from gnn_gmm_analysis import GNNGMMAnalyzer
        print("✓ GNNGMMAnalyzer imported successfully")
        
        # Test instantiation
        analyzer = GNNGMMAnalyzer(
            model_type="dimenet-10conf",
            checkpoint_index=0,
            output_dir="../GNNGMMAnalyzerResults/test_output",
            sample_frac=0.1
        )
        print("✓ GNNGMMAnalyzer instantiated successfully")
        
        # Test output directory creation
        if analyzer.output_dir.exists():
            print("✓ Output directory created successfully")
        else:
            print("✗ Output directory not created")
            return False
        
    except ImportError as e:
        print(f"✗ GNNGMMAnalyzer import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ GNNGMMAnalyzer instantiation failed: {e}")
        return False
    
    return True

def test_optimized_extraction():
    """Test optimized extraction class"""
    print("\nTesting optimized extraction...")
    
    try:
        from optimized_embedding_extraction import OptimizedEmbeddingExtractor
        print("✓ OptimizedEmbeddingExtractor imported successfully")
        
        # Test instantiation
        extractor = OptimizedEmbeddingExtractor(
            model_type="dimenet-10conf",
            checkpoint_index=0,
            output_dir="../GNNGMMAnalyzerResults/test_extraction",
            sample_frac=0.1
        )
        print("✓ OptimizedEmbeddingExtractor instantiated successfully")
        
    except ImportError as e:
        print(f"✗ OptimizedEmbeddingExtractor import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ OptimizedEmbeddingExtractor instantiation failed: {e}")
        return False
    
    return True

def create_sample_data():
    """Create sample data for testing"""
    print("\nCreating sample data for testing...")
    
    # Create sample embeddings
    n_samples = 100
    n_features = 64
    
    sample_embeddings = {
        'emb': np.random.randn(n_samples, n_features),
        'output_block_0': np.random.randn(n_samples, n_features),
        'output_block_1': np.random.randn(n_samples, n_features),
        'output_block_2': np.random.randn(n_samples, n_features)
    }
    
    # Save sample data
    test_dir = Path("test_sample_data")
    test_dir.mkdir(exist_ok=True)
    
    for layer_name, embeddings in sample_embeddings.items():
        np.save(test_dir / f"{layer_name}.npy", embeddings)
    
    print(f"✓ Sample data created in {test_dir}")
    return sample_embeddings

def test_gmm_fitting():
    """Test GMM fitting on sample data"""
    print("\nTesting GMM fitting...")
    
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        
        # Create sample data
        sample_data = create_sample_data()
        
        # Test GMM fitting
        embeddings = sample_data['emb']
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(embeddings_scaled)
        
        labels = gmm.predict(embeddings_scaled)
        probabilities = gmm.predict_proba(embeddings_scaled)
        
        print(f"✓ GMM fitted successfully: {len(np.unique(labels))} clusters")
        print(f"✓ Probabilities shape: {probabilities.shape}")
        
    except Exception as e:
        print(f"✗ GMM fitting failed: {e}")
        return False
    
    return True

def cleanup_test_files():
    """Clean up test files"""
    print("\nCleaning up test files...")
    
    test_dirs = ["../GNNGMMAnalyzerResults/test_output", "../GNNGMMAnalyzerResults/test_extraction", "test_sample_data"]
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            import shutil
            shutil.rmtree(test_dir)
            print(f"✓ Removed {test_dir}")

def main():
    """Run all tests"""
    print("Running GNN GMM Analysis Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_model_setup,
        test_gmm_analysis_class,
        test_optimized_extraction,
        test_gmm_fitting
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ Test failed: {test.__name__}")
        except Exception as e:
            print(f"✗ Test error in {test.__name__}: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The GMM analysis should work correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    # Clean up
    cleanup_test_files()

if __name__ == "__main__":
    main()
