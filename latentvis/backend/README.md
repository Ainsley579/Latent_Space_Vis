# Backend Documentation

## Overview

The Latent Space Vis backend provides a unified pipeline for generating frontend-ready visualizations from neural network embeddings.

### Features

- **GMM Clustering**: Automatic Gaussian Mixture Model clustering with BIC/AIC optimization
- **Dimensionality Reduction**: PCA or UMAP 2D projections
- **Cluster Analysis**: Compute transitions across layers (for Sankey diagrams)
- **Representative Extraction**: Find top-k closest/farthest samples per cluster
- **Real-ID Tracking**: All data points tracked with real_id for consistency

## Directory Structure

```
latentvis/backend/
├── analyzer/
│   ├── __init__.py
│   ├── EmbeddingAnalyzer      # GMM clustering
│   ├── DimensionalityReducer  # PCA/UMAP
│   ├── SankeyFlowAnalyzer     # Layer transitions
│   └── ClusterRepresentativesExtractor
├── generator/
│   ├── __init__.py
│   ├── ScatterGenerator       # 2D scatter JSON
│   ├── SankeyGenerator        # Sankey transitions
│   └── ClusterRepresentativesGenerator
├── utils/
│   ├── __init__.py
│   ├── EmbeddingLoader        # Load .npy, .npz, pickle
│   ├── DataValidator          # Check data consistency
│   └── ConfigManager          # Load config files
├── scripts/
│   ├── build_scatter_by_layer.py     # CSV → JSON conversion
│   └── validate_scatterplot_data.py  # Verify data accuracy
├── data/
│   ├── scatter_plots/         # Intermediate CSV files
│   ├── sankey/                # Transition data
│   └── cluster_representatives/
├── generate_frontend_data.py  # Main entry point
└── config.yaml                # Configuration template
```

## Usage

### 1. Basic Usage (Most Common)

```bash
# Generate all frontend data from embedding directory
cd /path/to/Latent_Space_Vis

python latentvis/backend/generate_frontend_data.py \
    --embeddings-dir /path/to/embeddings/dimenet10conf \
    --model-name dimenet10conf \
    --output-dir latentvis/frontend/public/data

python latentvis/backend/generate_frontend_data.py \
    --embeddings-dir /path/to/embeddings/dimenet_noaug \
    --model-name dimenet_noaug \
    --output-dir latentvis/frontend/public/data
```

### 2. With Custom Configuration

```bash
python latentvis/backend/generate_frontend_data.py \
    --embeddings-dir /path/to/embeddings \
    --model-name mymodel \
    --config latentvis/backend/config.yaml \
    --reduction-method umap \
    --output-dir latentvis/frontend/public/data
```

### 3. Python API

```python
from pathlib import Path
from latentvis.backend.analyzer import EmbeddingAnalyzer, DimensionalityReducer
from latentvis.backend.generator import ScatterGenerator
from latentvis.backend.utils import EmbeddingLoader

# Load embeddings
embeddings_dir = Path('/path/to/embeddings')
embeddings_dict = {}
for npy_file in embeddings_dir.glob('*.npy'):
    layer_name = npy_file.stem
    embeddings_dict[layer_name] = EmbeddingLoader.load_numpy(npy_file)

# Analyze
analyzer = EmbeddingAnalyzer()
gmm_result = analyzer.fit_gmm(embeddings_dict['emb'])
cluster_result = analyzer.get_cluster_assignments(
    embeddings_dict['emb'], gmm_result['optimal_gmm']
)

# Reduce
reducer = DimensionalityReducer('pca')
coords_2d = reducer.fit_reduce(embeddings_dict['emb'])

# Generate
scatter_gen = ScatterGenerator()
scatter_df = scatter_gen.generate_csv(
    embeddings_dict['emb'], 'emb', 'dimenet10conf',
    cluster_result['labels'], coords_2d
)
```

## Input Formats

### Embeddings

Expected: NumPy arrays in one of these formats:
- `.npy` files (one per layer): `emb.npy`, `output_block_0.npy`, etc.
- `.npz` file: single archive with multiple arrays
- `.pkl` pickle file: dict mapping layer_name → array

Shape: `(n_samples, n_features)` where `n_samples` should be the same across all layers

### Names (Expected DimeNet Layers)

```
emb, output_block_0, output_block_1, output_block_2, output_block_3,
output_block_4, u_mlp_input, u_mlp_l1, u_mlp_l2
```

## Output Formats

### Scatterplot JSON

Location: `frontend/public/data/scatterplot/{model_name}_{layer_name}.json`

Format:
```json
[
  {
    "real_id": "0",
    "x": 18.023569107055664,
    "y": 2446.088134765625,
    "cluster_id": 1
  },
  ...
]
```

**Key Points**:
- `real_id`: Sample index 0 to n_samples-1 (string)
- `x`, `y`: 2D coordinates from PCA/UMAP (float)
- `cluster_id`: GMM cluster assignment (int)
- **All samples must have consecutive real_id 0-(n-1)**

### Sankey Transitions

Location: `backend/data/sankey/{model_name}/*.csv` → generated as needed

Format:
```csv
from_components,to_components,from_cluster,to_cluster,count,probability
2,3,0,0,50,0.333
2,3,0,1,75,0.500
2,3,0,2,25,0.167
```

### Cluster Representatives

Location: `frontend/public/data/cluster_representatives/cluster_representatives.json`

Format:
```json
[
  {
    "layer_name": "emb",
    "model_type": "dimenet10conf",
    "cluster_id": 0,
    "top_k_closest_real_ids": [5, 12, 23, 45, 67],
    "top_k_farthest_real_ids": [150, 149, 148, 147, 146],
    "k": 5
  },
  ...
]
```

## Validation

Verify generated data:

```bash
python latentvis/backend/scripts/validate_scatterplot_data.py
```

Checks:
- ✓ All 18 JSON files exist (9 layers × 2 models)
- ✓ Each file has 152 points
- ✓ real_id are consecutive 0-151
- ✓ Coordinate values match source CSVs
- ✓ Cluster IDs are valid

## Data Pipeline Flow

```
User Embeddings (.npy)
         ↓
[EmbeddingAnalyzer]
  - Load & scale
  - Fit GMM (2-10 components)
  - Predict clusters
         ↓
[DimensionalityReducer]
  - PCA/UMAP to 2D
         ↓
[ScatterGenerator]
  - Create {model}_{layer}.json
  - Each: 152 points with real_id
         ↓
[SankeyFlowAnalyzer]
  - Compute layer transitions
  - Create {model}_transitions.csv
         ↓
[ClusterRepresentativesExtractor]
  - Find top-k closest/farthest
  - Create cluster_representatives.json
         ↓
Frontend JSON files
  latentvis/frontend/public/data/
    ├── scatterplot/*.json
    └── cluster_representatives.json
```

## Configuration Options

See `config.yaml` for defaults:

```yaml
n_components_range: [2, 10]     # Range to test for optimal GMM
reduction_method: pca            # 'pca' or 'umap'
random_state: 42                 # Reproducible results
scatter_k_samples: 152           # Expected sample count
cluster_k: 5                      # Top-k best/worst samples per cluster
```

## Troubleshooting

### "No embeddings found in directory"
- Ensure embeddings are saved as `.npy` files
- File names should match layer names (emb.npy, output_block_0.npy, etc.)

### "Inconsistent sample count"
- All embedding files must have same first dimension (number of samples)
- Check that all layers were extracted from same model/checkpoint

### JSON validation fails
- Check `validate_scatterplot_data.py` output
- Ensure all 18 layer × model combinations are present

### Memory issues
- Reduce sample count in config
- Use PCA instead of UMAP for faster reduction

## Development

To add new functionality:

1. **New Analyzer**: Add class to `analyzer/__init__.py`
2. **New Generator**: Add class to `generator/__init__.py`
3. **New Utility**: Add to `utils/__init__.py`
4. **Update Pipeline**: Modify `generate_frontend_data.py` to use new components

Example:
```python
# In analyzer/__init__.py
class MyNewAnalyzer(object):
    def analyze(self, embeddings):
        ...
        return results

# In generate_frontend_data.py line ~140
my_analyzer = MyNewAnalyzer()
results = my_analyzer.analyze(embeddings)
```

## Performance Notes

For 152 samples × 9 layers with 64-128 dimensional embeddings:
- GMM fitting: ~100ms per layer
- PCA reduction: ~50ms per layer
- Total: ~1.5 seconds for all layers

## References

- GMM: scikit-learn GaussianMixture
- Reduction: scikit-learn PCA / umap-learn
- Project: Latent Space Visualization for Neural Networks
