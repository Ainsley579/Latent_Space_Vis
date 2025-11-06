# Quick Start Guide - Real ID Tracking

## ✅ Your Environment is Ready!

All required packages are installed. You can now generate the CSV files.

## 🚀 Run the Script (Easiest Way)

```bash
cd scripts/GNNGMMAnalyzer
bash run_real_id_tracking.sh
```

This will:
- Use embeddings from: `scripts/dimenet10conf_embeddings/`
- Save output to: `frontend/public/data/`
- Auto-detect optimal number of components
- Generate both CSV files

## 📁 Output Files

After running, you'll find:

1. **`frontend/public/data/real_id_layer_transitions/real_id_layer_transitions.csv`**
   - Tracks real IDs through layer transitions (Sankey through layers)

2. **`frontend/public/data/real_id_component_transitions/real_id_component_transitions.csv`**
   - Tracks real IDs through component count changes

## 🎯 Custom Options

If you want to specify parameters:

```bash
bash run_real_id_tracking.sh \
    ../../scripts/dimenet10conf_embeddings \
    ../../frontend/public/data \
    13 \
    output_block_0
```

Parameters:
1. Embeddings directory (default: `../../scripts/dimenet10conf_embeddings`)
2. Output directory (default: `../../frontend/public/data`)
3. Number of components (optional, auto-detects if not provided)
4. Layer name for component analysis (optional, uses first layer if not provided)

## 📊 What the CSV Files Contain

### Layer Transitions CSV
- `real_id`: Point index (0, 1, 2, ...)
- `from_layer`: Source layer (e.g., "emb")
- `to_layer`: Target layer (e.g., "output_block_0")
- `from_cluster`: Cluster ID in source layer
- `to_cluster`: Cluster ID in target layer

### Component Transitions CSV
- `real_id`: Point index (0, 1, 2, ...)
- `layer`: Layer name (e.g., "output_block_0")
- `from_components`: Number of components before change
- `to_components`: Number of components after change
- `from_cluster`: Cluster ID with `from_components` components
- `to_cluster`: Cluster ID with `to_components` components

## ❓ Need Help?

- Check environment: `python3 check_environment.py`
- See full documentation: `README_REAL_ID_TRACKING.md`
- Check script help: `python3 generate_real_id_tracking_csv.py --help`

