# GNN Layer-wise GMM Analysis

This folder contains comprehensive tools for analyzing Gaussian Mixture Models (GMM) on Graph Neural Network (GNN) embeddings across different layers.

## Folder Structure

```
latent-spaces/
├── GNNGMMAnalyzer/                    # Analysis scripts
│   ├── gnn_gmm_analysis.py           # Main analysis script
│   ├── optimized_embedding_extraction.py  # Optimized extraction
│   ├── run_gmm_analysis_example.py   # Example usage
│   ├── test_gmm_analysis.py          # Test script
│   └── README_GMM_Analysis.md         # This documentation
└── GNNGMMAnalyzerResults/             # Results directory
    ├── gnn_gmm_results/              # Main analysis results
    ├── optimized_embeddings/          # Extracted embeddings
    ├── example_gmm_results/          # Example analysis results
    └── full_gmm_results/             # Full analysis results
```

## Overview

The analysis includes:
1. **Baseline GMM fitting** with BIC/AIC for optimal components
2. **Layer-wise trajectory analysis** showing how clusters evolve
3. **Visualization of evolution** across layers with 2D projections
4. **Quantification of cluster sharpening** using covariance metrics
5. **Boundary analysis** for ambiguous molecules
6. **Fuzzy clustering** with probability assignments
7. **Parallel coordinates plots** for trajectory tracking

## Files

- `gnn_gmm_analysis.py` - Main analysis script with comprehensive GMM analysis
- `optimized_embedding_extraction.py` - Optimized embedding extraction (addresses bottleneck)
- `run_gmm_analysis_example.py` - Example script showing how to use the analysis
- `README_GMM_Analysis.md` - This documentation

## Usage

### Basic Usage

```bash
# Navigate to the GNNGMMAnalyzer directory
cd GNNGMMAnalyzer

# Run analysis on dimenet-10conf model
python gnn_gmm_analysis.py --model_type dimenet-10conf --checkpoint_index 0

# Run with custom output directory and sample fraction
python gnn_gmm_analysis.py --model_type dimenet-10conf --checkpoint_index 0 --output_dir ../GNNGMMAnalyzerResults/my_results --sample_frac 0.5
```

### Example Usage

```bash
# Run example analysis (50% of data)
python run_gmm_analysis_example.py --mode example

# Run full analysis (100% of data)
python run_gmm_analysis_example.py --mode full
```

### Optimized Embedding Extraction

```bash
# Extract embeddings using single pass method
python optimized_embedding_extraction.py --model_type dimenet-10conf --method single_pass

# Benchmark different extraction methods
python optimized_embedding_extraction.py --model_type dimenet-10conf --method benchmark
```

## Command Line Arguments

### gnn_gmm_analysis.py

- `--model_type`: Model type (`dimenet-10conf`, `dimenet-100conf`, `dimenet-noaug`)
- `--checkpoint_index`: Checkpoint index to load (default: 0)
- `--output_dir`: Output directory for results (default: `gnn_gmm_results`)
- `--sample_frac`: Fraction of data to sample (default: 1.0)
- `--max_components`: Maximum components for BIC/AIC testing (default: 20)

### optimized_embedding_extraction.py

- `--model_type`: Model type
- `--checkpoint_index`: Checkpoint index
- `--output_dir`: Output directory for embeddings
- `--sample_frac`: Fraction of data to sample
- `--method`: Extraction method (`single_pass`, `hooks`, `benchmark`)

## Output Files

The analysis generates several output files:

### Visualizations
- `component_selection.png` - BIC/AIC/Silhouette scores vs components
- `layer_evolution.png` - 2D projections showing evolution across layers
- `cluster_sharpening.png` - Determinant/trace trends across layers
- `boundary_analysis.png` - Analysis of ambiguous molecules
- `parallel_coordinates.png` - Parallel coordinates plot
- `fuzzy_clustering_analysis.png` - Fuzzy clustering metrics

### Data Files
- `embeddings/` - Directory containing layer embeddings (.pt and .npy files)
- `boundary_analysis.npy` - Boundary molecule analysis results
- `fuzzy_clustering_results.npy` - Fuzzy clustering results
- `analysis_summary.txt` - Summary report

## Key Features

### 1. Optimal Component Selection
- Uses BIC, AIC, and Silhouette scores to find optimal number of components
- Plots all three metrics for comparison
- Defaults to AIC recommendation

### 2. Layer-wise Analysis
- Extracts embeddings from all GNN layers (embedding + output blocks)
- Fits GMM to each layer with same number of components
- Tracks evolution of cluster structure

### 3. Visualization
- 2D projections using PCA for consistent visualization
- GMM ellipses overlaid on scatter plots
- Points colored by molecule class (if available)
- Shows "movie" of cluster formation

### 4. Cluster Sharpening
- Tracks average determinant and trace of covariance matrices
- Decreasing trends indicate tighter, more distinct clusters
- Quantifies how well the GNN separates different molecule types

### 5. Boundary Analysis
- Identifies molecules with highest assignment entropy
- Shows which molecules remain ambiguous even at final layer
- Analyzes SMILES strings of boundary cases

### 6. Fuzzy Clustering
- Provides probability assignments for each point
- Calculates confidence and ambiguity metrics
- Shows how uncertainty changes across layers

### 7. Parallel Coordinates
- Visualizes trajectory of embeddings across layers
- Shows how individual molecules move through embedding space
- Helps identify common patterns

## Performance Optimization

The scripts include several optimizations:

1. **Efficient Embedding Extraction**: Single-pass extraction for multiple layers
2. **Memory Management**: Processes data in batches
3. **Caching**: Saves intermediate results to avoid recomputation
4. **Sampling**: Option to use subset of data for faster analysis

## Environment Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy
- scipy

## Conda Environment

Make sure to use the `losslandscapestda` conda environment:

```bash
conda activate losslandscapestda
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `sample_frac` to use less data
2. **Slow Extraction**: Use `optimized_embedding_extraction.py` first
3. **Missing Dependencies**: Ensure all required packages are installed
4. **Model Loading**: Check that model paths are correct

### Performance Tips

1. Start with `sample_frac=0.1` for quick testing
2. Use `--max_components=10` for faster BIC/AIC testing
3. Run embedding extraction separately if needed
4. Monitor memory usage during analysis

## Example Workflow

1. **Quick Test**:
   ```bash
   cd GNNGMMAnalyzer
   python gnn_gmm_analysis.py --sample_frac 0.1 --max_components 10
   ```

2. **Full Analysis**:
   ```bash
   cd GNNGMMAnalyzer
   python gnn_gmm_analysis.py --sample_frac 1.0 --max_components 20
   ```

3. **Extract Embeddings Separately**:
   ```bash
   cd GNNGMMAnalyzer
   python optimized_embedding_extraction.py --method single_pass
   ```

## Results Interpretation

### Component Selection
- Lower BIC/AIC scores indicate better fit
- Higher Silhouette scores indicate better separation
- Choose components that balance fit quality and interpretability

### Layer Evolution
- Look for increasing separation between different molecule types
- Ellipses should become more distinct and tighter
- Points should cluster more clearly by molecule class

### Cluster Sharpening
- Decreasing determinant/trace indicates tighter clusters
- Sharp drops suggest significant learning at specific layers
- Plateaus may indicate convergence

### Boundary Analysis
- High entropy molecules are ambiguous to the model
- These may represent edge cases or difficult-to-classify molecules
- SMILES analysis can reveal structural patterns
