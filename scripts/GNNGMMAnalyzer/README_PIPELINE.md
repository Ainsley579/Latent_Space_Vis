# Embedding Processing Pipeline

This directory contains a general-purpose pipeline for processing embeddings and generating visualization data.

## Overview

The pipeline has been reorganized into a modular structure:

1. **Core Functions** (`embedding_pipeline_core.py`): General utilities that work with any embeddings
2. **Unified Pipeline** (`embedding_pipeline.py`): Main pipeline class that generates the two required output files
3. **Entry Point** (`run_embedding_pipeline.py`): Command-line interface for the general pipeline
4. **Legacy Scripts**: Updated to use core functions while maintaining backward compatibility

## Output Files

The pipeline generates two main output files:

1. **`scatter_plot_data.csv`**: For scatter plot visualization
   - Columns: `id`, `x`, `y`, `color_value`, `layer`
   - One row per data point per layer

2. **`sankey_diagram_data.csv`**: For Sankey diagram visualization
   - Columns: `real_id`, `from_layer`, `to_layer`, `from_cluster`, `to_cluster`
   - One row per transition between consecutive layers

## Usage

### General Pipeline

The general pipeline accepts ANY embeddings:

```python
from embedding_pipeline import EmbeddingPipeline
import numpy as np

# Prepare embeddings (dict of layer_name -> numpy array)
embeddings = {
    'layer1': np.array([...]),  # shape: (n_samples, n_features)
    'layer2': np.array([...]),
    ...
}

# Optional: ID mapping (embedding index -> real dataset index)
id_mapping = {0: 100, 1: 101, ...}

# Optional: Color values for scatter plots
color_values = np.array([...])

# Create and run pipeline
pipeline = EmbeddingPipeline(
    embeddings=embeddings,
    output_dir='./output',
    id_mapping=id_mapping,
    n_components=13,  # or None for auto-detection
    color_values=color_values,
    metadata={'dataset': 'my_dataset', ...}  # Optional metadata
)
pipeline.run()
```

### Command-Line Interface

```bash
# Basic usage with auto-detection
python run_embedding_pipeline.py \
    --embeddings_dir ./embeddings \
    --output_dir ./output

# Specify layer files explicitly
python run_embedding_pipeline.py \
    --embeddings_dir ./embeddings \
    --layer_files emb.pt,output0.pt,output1.pt \
    --layer_names emb,output0,output1 \
    --output_dir ./output

# With ID mapping
python run_embedding_pipeline.py \
    --embeddings_dir ./embeddings \
    --id_mapping mapping.json \
    --output_dir ./output

# With training/test data (saved as metadata)
python run_embedding_pipeline.py \
    --embeddings_dir ./embeddings \
    --train_data train.json \
    --test_data test.json \
    --output_dir ./output
```

### Legacy Scripts (Backward Compatible)

All existing scripts continue to work as before:

```bash
# Generate cluster transition matrices
python generate_cluster_transition_matrices_with_real_id.py \
    --embeddings_dir ./embeddings \
    --output_dir ./output \
    --version 1

# Generate scatter plot data
python generate_scatter_plot_data.py \
    --embeddings_dir ./embeddings \
    --output_dir ./output \
    --version 1

# Generate real ID tracking
python generate_real_id_tracking_csv.py \
    --embeddings_dir ./embeddings \
    --output_dir ./output \
    --version 1
```

## Architecture

### Core Functions (`embedding_pipeline_core.py`)

General-purpose functions that work with any embeddings:

- `load_embeddings_from_dict()`: Load embeddings from a dictionary
- `load_embeddings_from_files()`: Load embeddings from files
- `adaptive_zca_whitening()`: Apply ZCA whitening
- `fit_gmm()`: Fit Gaussian Mixture Model
- `find_optimal_cluster_alignment()`: Align clusters using Hungarian algorithm
- `compute_mds_positions()`: Compute MDS positions for visualization
- `map_embedding_indices_to_real_ids()`: Map embedding indices to real IDs

### Unified Pipeline (`embedding_pipeline.py`)

Main pipeline class that:

1. Accepts embeddings in any format (dict, files, etc.)
2. Optionally accepts ID mappings, training/test data, metadata
3. Automatically detects optimal number of components (optional)
4. Generates the two required output files
5. Saves all metadata and configuration

### Backward Compatibility

All existing scripts have been updated to use the core functions internally while maintaining their original interfaces. This ensures:

- All previous code continues to work
- No breaking changes to existing scripts
- Improved code reusability and maintainability

## File Structure

```
GNNGMMAnalyzer/
├── embedding_pipeline_core.py      # Core utilities
├── embedding_pipeline.py            # Unified pipeline class
├── run_embedding_pipeline.py        # Entry point for general pipeline
├── generate_cluster_transition_matrices_with_real_id.py  # Legacy (updated)
├── generate_scatter_plot_data.py    # Legacy (updated)
├── generate_real_id_tracking_csv.py # Legacy (updated)
└── ... (other scripts)
```

## Parameters

### Required
- `embeddings`: Dictionary of layer_name -> numpy array
- `output_dir`: Directory to save output files

### Optional
- `id_mapping`: Dictionary mapping embedding indices to real IDs
- `n_components`: Number of GMM components (auto-detect if None)
- `zca_threshold`: Threshold for ZCA whitening (default: 1e-5)
- `random_state`: Random state for reproducibility (default: 42)
- `color_values`: Array of values for coloring scatter plots
- `layer_order`: List specifying order of layers (default: sorted)
- `metadata`: Dictionary of additional metadata to save

## Output Files

The pipeline generates:

1. **`scatter_plot_data.csv`**: Scatter plot visualization data
2. **`sankey_diagram_data.csv`**: Sankey diagram visualization data
3. **`pipeline_metadata.json`**: Pipeline configuration and metadata
4. **`id_mapping.json`**: ID mapping (if provided)

## Testing

Comprehensive tests are provided in `test_embedding_pipeline.py`. Run tests with:

```bash
# Using pytest directly
pytest test_embedding_pipeline.py -v

# Or use the test runner script
./run_tests.sh
```

The test suite includes:
- **Core Functions Tests**: Test all utility functions (ZCA, GMM, MDS, etc.)
- **Pipeline Tests**: Test the main EmbeddingPipeline class
- **File Loading Tests**: Test loading embeddings from files
- **Edge Cases**: Test error handling and edge cases

All 28 tests should pass. These tests ensure:
- Core functions work correctly with various input types
- Pipeline generates correct output files
- ID mappings are applied correctly
- Metadata is saved properly
- Error handling works for invalid inputs

## Examples

See the docstrings in `embedding_pipeline.py` and `run_embedding_pipeline.py` for more detailed examples.

