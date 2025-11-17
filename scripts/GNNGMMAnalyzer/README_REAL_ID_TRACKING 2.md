# Real ID Tracking CSV Generator

This script generates CSV files that track individual point IDs (real IDs) through:
1. **Layer transitions** - How points move through clusters as they pass through different GNN layers
2. **Component transitions** - How points move through clusters as the number of GMM components changes

## Quick Start

### Option 1: Using the Helper Script (Recommended)

```bash
cd scripts/GNNGMMAnalyzer
bash run_real_id_tracking.sh
```

This will use default paths:
- Embeddings: `../../scripts/dimenet10conf_embeddings`
- Output: `../../frontend/public/data`

### Option 2: Using the Helper Script with Custom Paths

```bash
cd scripts/GNNGMMAnalyzer
bash run_real_id_tracking.sh [embeddings_dir] [output_dir] [n_components] [layer_name]
```

Example:
```bash
bash run_real_id_tracking.sh \
    ../../scripts/dimenet10conf_embeddings \
    ../../frontend/public/data \
    13 \
    output_block_0
```

### Option 3: Direct Python Command

```bash
cd scripts/GNNGMMAnalyzer
python3 generate_real_id_tracking_csv.py \
    --embeddings_dir ../../scripts/dimenet10conf_embeddings \
    --output_dir ../../frontend/public/data \
    --n_components 13 \
    --layer_name output_block_0 \
    --zca_threshold 1e-5
```

## Environment Setup

### Check Your Environment

First, check if all required packages are installed:

```bash
cd scripts/GNNGMMAnalyzer
python3 check_environment.py
```

### Install Required Packages

If packages are missing, install them:

**Using pip:**
```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn scipy
```

**Using conda (recommended for this project):**
```bash
conda activate losslandscapestda
conda install pytorch scikit-learn pandas numpy matplotlib seaborn scipy -c conda-forge
```

## Required Dependencies

- Python 3.7+
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Command Line Arguments

- `--embeddings_dir` (required): Path to directory containing embedding `.pt` files
- `--output_dir` (optional): Output directory for CSV files (default: `../../frontend/public/data`)
- `--n_components` (optional): Number of GMM components for layer analysis (default: auto-detect using AIC)
- `--layer_name` (optional): Layer name for component transition analysis (default: first layer)
- `--zca_threshold` (optional): Threshold for ZCA whitening (default: 1e-5)

## Output Files

The script creates two folders in the output directory:

### 1. `real_id_layer_transitions/`
Contains: `real_id_layer_transitions.csv`

**Columns:**
- `real_id`: The point/molecule index (0-based)
- `from_layer`: Source layer name (e.g., "emb", "output_block_0")
- `to_layer`: Target layer name
- `from_cluster`: Cluster ID in source layer
- `to_cluster`: Cluster ID in target layer

**Use case:** Track how individual points move through clusters as they pass through GNN layers (for Sankey diagrams).

### 2. `real_id_component_transitions/`
Contains: `real_id_component_transitions.csv`

**Columns:**
- `real_id`: The point/molecule index (0-based)
- `layer`: Layer name (e.g., "output_block_0")
- `from_components`: Number of components before change
- `to_components`: Number of components after change
- `from_cluster`: Cluster ID with `from_components` components
- `to_cluster`: Cluster ID with `to_components` components

**Use case:** Track how individual points move through clusters as the number of GMM components changes for a specific layer.

## Example Usage

### Basic Usage (Auto-detect components)
```bash
python3 generate_real_id_tracking_csv.py \
    --embeddings_dir ../../scripts/dimenet10conf_embeddings \
    --output_dir ../../frontend/public/data
```

### Specify Number of Components
```bash
python3 generate_real_id_tracking_csv.py \
    --embeddings_dir ../../scripts/dimenet10conf_embeddings \
    --output_dir ../../frontend/public/data \
    --n_components 13
```

### Specify Layer for Component Analysis
```bash
python3 generate_real_id_tracking_csv.py \
    --embeddings_dir ../../scripts/dimenet10conf_embeddings \
    --output_dir ../../frontend/public/data \
    --layer_name output_block_2
```

## Troubleshooting

### Error: "No embeddings loaded"
- Check that the embeddings directory path is correct
- Verify that `.pt` files exist in the directory
- The script looks for files with patterns: `*_cas.pt`, `*_unique.pt`, or `*.pt`

### Error: "Required Python packages not found"
- Run `python3 check_environment.py` to see which packages are missing
- Install missing packages using pip or conda

### Error: "CUDA out of memory" or similar
- The script automatically sets `CUDA_VISIBLE_DEVICES=""` to use CPU
- If you still see GPU-related errors, ensure PyTorch CPU version is installed

### Slow Performance
- The script uses CPU by default for compatibility
- Component detection can take a few minutes - this is normal
- Consider specifying `--n_components` to skip auto-detection

## Notes

- The script does **not** modify existing CSV files
- It creates new folders and files in the output directory
- The script prioritizes `*_cas.pt` files if available, then `*_unique.pt`, then `*.pt`
- All embeddings are processed with ZCA whitening and standardization (same as other analysis scripts)

