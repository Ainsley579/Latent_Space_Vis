# DimeNet Benchmark: Standardized Data Summary

Generated for system testing. All data includes **both dimenet10conf and dimenet_noaug** models.

## Model Coverage

| Data | dimenet10conf | dimenet_noaug |
|------|---------------|---------------|
| global_index | ✓ (152 samples) | ✓ (156 samples) |
| clustering_metrics | ✓ FMI, BIC/AIC | ✓ optimal_n_components |
| cluster_assignments | ✓ | ✓ |
| sankey transitions | ✓ | ✓ |
| scatter_plots | ✓ | ✓ |
| cluster_representatives | ✓ | ✓ |

## Directory Structure

```
DimenetBenchmark/
├── data/
│   ├── global_index.csv              # real_id, dataset, model_type, sample_index (both models)
│   ├── clustering_metrics/
│   │   ├── clustering_metrics.csv    # FMI, BIC/AIC, optimal_n per model/layer
│   │   ├── cluster_assignments_with_real_id.csv           # 10-conf (152 rows)
│   │   └── cluster_assignments_with_real_id_dimenet_noaug.csv  # no-aug (156 rows)
│   ├── sankey/
│   │   ├── sankey_index.json         # Metadata for 2D Sankey (X:layers, Y:models)
│   │   ├── dimenet10conf/            # 10-conf model transition flows
│   │   │   └── transitions_*.csv     # Per-layer (emb, output_block_0..4, u_mlp_*)
│   │   └── dimenet_noaug/            # No-aug model transition flows
│   │       └── transitions_*.csv     # Per-layer
│   ├── scatter_plots/
│   │   ├── scatter_by_layer.csv      # Combined: real_id, layer, model_type, x, y, cluster_id
│   │   └── scatter_noaug_by_layer.csv  # No-aug only (also in combined)
│   └── cluster_representatives/      # Top-K closest/farthest per cluster
│       └── cluster_representatives.json  # Both 10-conf and no-aug
├── scripts/
│   ├── build_standardized_data.py
│   ├── generate_cluster_representatives.py
│   └── add_noaug_data.py
└── DIMENET_BENCHMARK_SUMMARY.md
```

## Data Schemas

### global_index.csv
| Column | Description |
|--------|-------------|
| real_id | Sample index within model (0-151 for 10-conf, 0-155 for no-aug) |
| dataset | dimenet_cas |
| model_type | dimenet10conf or dimenet_noaug |
| sample_index | Row index in embeddings |

**Note:** Primary key is (model_type, real_id). real_id is scoped per model.

### clustering_metrics.csv
| Column | Description |
|--------|-------------|
| model_type | dimenet10conf, dimenet_noaug, or dimenet10conf_vs_noaug |
| layer_name | emb, output_block_0..4, u_mlp_input, u_mlp_l1, u_mlp_l2 |
| fmi_score | Fowlkes-Mallows Index (10-conf vs no-aug) |
| bic, aic | Model selection metrics |
| optimal_n_components | PG-Means optimal cluster count |

### Cluster Assignments
- **cluster_assignments_with_real_id.csv**: 10-conf, 152 samples
- **cluster_assignments_with_real_id_dimenet_noaug.csv**: no-aug, 156 samples

### Sankey (transitions_*.csv)
- **X-axis**: Layers (emb → output_block_0..4 → u_mlp_input → u_mlp_l1 → u_mlp_l2)
- **Y-axis**: Models (dimenet10conf, dimenet_noaug)
- Flow: from_components, to_components, from_cluster, to_cluster, count, probability

### scatter_by_layer.csv
| Column | Description |
|--------|-------------|
| real_id | Sample index within model |
| layer_name | Which layer |
| model_type | dimenet10conf or dimenet_noaug |
| x, y | 2D projection (PCA) |
| cluster_id | Cluster assignment |

## Source Migration

- **GNNGMMAnalyzerResults**: FMI/CSPA, BIC/AIC, component flow transitions, cluster assignments (both models)
- **GNNGMMAnalyzerExperimental**: experiment1_parameter_scan (BIC/AIC vs reg_covar)

## Usage for System Testing

```python
import pandas as pd
import json

# Load with model_type filter
global_idx = pd.read_csv("data/global_index.csv")
clusters_10conf = pd.read_csv("data/clustering_metrics/cluster_assignments_with_real_id.csv")
clusters_noaug = pd.read_csv("data/clustering_metrics/cluster_assignments_with_real_id_dimenet_noaug.csv")
scatter = pd.read_csv("data/scatter_plots/scatter_by_layer.csv")

# Filter by model
scatter_10conf = scatter[scatter["model_type"] == "dimenet10conf"]
scatter_noaug = scatter[scatter["model_type"] == "dimenet_noaug"]

# Sankey: load transitions by model and layer
sankey_meta = json.load(open("data/sankey/sankey_index.json"))
# e.g. dimenet10conf: "data/sankey/dimenet10conf/transitions_emb.csv"
# e.g. dimenet_noaug: "data/sankey/dimenet_noaug/transitions_emb.csv"
```
