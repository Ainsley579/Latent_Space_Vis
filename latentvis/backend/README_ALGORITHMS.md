# Backend Algorithms (Code-Accurate)

This document describes how the backend algorithms are implemented in code today.
It is aligned with:

- `run_pipeline_from_embeddings.py`
- `embedding_loader.py`
- `clustering.py`
- `dimensionality_reduction.py`

## 1. End-to-End Pipeline

Entry point: `run_pipeline_from_embeddings.py::run_pipeline(...)`

Per run, the backend:

1. Loads embeddings and normalizes them into `(epoch, layer) -> ndarray`.
2. Reorders layers for traversal.
3. For each `(epoch, layer)` cell:
   - Flattens embedding to 2D `(n_samples, n_features)`.
   - Computes 2D scatter coordinates (PCA/MDS/UMAP/t-SNE).
   - Runs one or more clustering methods.
   - Builds cluster representatives (closest/farthest + full sorted members).
4. Writes frontend artifacts:
   - scatterplot JSONs (default + per-method suffix files)
   - BIC/AIC JSONs
   - Sankey edges and per-sample Sankey rows
   - top-k and sorted-members JSONs
   - cluster assignment CSV
   - optional substructure JSON (RDKit)
5. Optionally writes FMI/CSPA summary across consecutive epochs.

## 2. How Layers/Epochs Are Split

Source: `embedding_loader.py`

### Input formats

- Single `.pt` file.
- Directory of `.pt` / `.npy` files.

### `.pt` parsing behavior

`load_single_pt(...)` supports these structures:

- Nested: `dict[epoch][layer] = tensor`
- Flat layers: `dict[layer] = tensor` (epoch becomes `"default"`)
- Explicit fields: `{"epochs", "layers", "embeddings"}`
- Deeply nested structures via recursive tensor discovery (`_find_tensors_nested`)

Special handling for ESOL-like structure:

- Expected shape: `epoch -> split(train/val/test) -> embeddings -> layer -> tensor`
- Split selection default order: `test -> val -> train`
- `--data-split` can force one split
- `real_ids` come from `split_data["id"]`
- Optional `id_to_smiles` map is built from aligned `id` + `smiles`

### Directory parsing behavior

`load_from_directory(...)`:

- If `--layer-file-map` is provided, layer order follows that map.
- Otherwise loads all `.pt`/`.npy` files and uses filename stem as layer.
- Single epoch key = `--model-name` (or directory name).

### Layer order used by pipeline

After loading, `run_pipeline` applies `_preferred_layer_order(...)`:

1. `fingerprint` first (if present)
2. `gnn_layer_1..N`
3. `ffn_layer_1..N` / `fnn_layer_1..N`
4. remaining layers in original order

This order controls Sankey traversal and layer axis layout.

## 3. Per-Cell Preprocessing

Sources:

- `run_pipeline_from_embeddings.py::_flatten_embeddings`
- `clustering.py::_flatten`
- `dimensionality_reduction.py`

Rules:

- If embedding rank > 2, flatten to `(n_samples, -1)`.
- If rank == 1, convert to column vector.
- Clustering and DR both standardize features with `StandardScaler`.

## 4. Clustering Methods

Source: `clustering.py`

`fit_clustering(...)` dispatches to one of:

- `pgmeans`
- `hdbscan`
- `agglomerative`
- `knn`

Naming note:

- In pipeline code, the implemented method name is `pgmeans`.
- Flask/API paths also use `pgmeans` as the canonical method key.

When `--clustering-method all`, pipeline runs all four and writes:

- default method (first, usually `pgmeans`) to output root
- others under `clustering/<method>/...`

### 4.1 PGMeans

Function: `fit_pgmeans(...)`

- Implementation detail: this repo implements PGMeans as a GaussianMixture model sweep with BIC/AIC-based model selection.
- Standardize + cast to `float64` for numerical stability.
- Test `k = k_min..k_max`.
- For each `k`, attempts robust fitting:
  - covariance: `full`, then `diag`
  - `reg_covar`: `1e-6` to `1e-2`
- Uses BIC minimum to choose best `k`.
- Outputs `labels`, `k`, `k_values`, `bic`, `aic`.
- If all fits fail: fallback to one cluster (`k=1`, all label 0).

### 4.2 HDBSCAN

Function: `fit_hdbscan(...)`

- Uses `min_cluster_size=5`, `min_samples=3`, metric `euclidean`.
- Noise label `-1` is remapped:
  - to `max_label + 1` if non-noise clusters exist
  - to `0` if all points are noise
- Returns non-negative labels only.

### 4.3 Agglomerative

Function: `fit_agglomerative(...)`

- Standardized input.
- If `n_clusters` not given:
  - first runs PGMeans and uses its selected `k`.
- `k` is clamped to `[2, n_samples-1]`.
- Linkage options: `ward` (default), `complete`, `average`.

CLI overrides:

- `--agg-k`
- `--agg-linkage`

### 4.4 KNN Connected-Component Clustering

Function: `fit_knn(...)`

- Builds symmetric kNN graph (`NearestNeighbors`, Euclidean).
- Clusters = connected components of graph.
- `k` neighbors controlled by `--knn-k` (or defaults to `k_max` in pipeline).

This is graph connectivity clustering, not k-means.

## 5. Scatterplot / Dimensionality Reduction

Source: `dimensionality_reduction.py`

Implemented reducers:

- PCA (`reduce_pca`)
- MDS with cosine dissimilarity (`reduce_mds`)
- UMAP (`reduce_umap`)
- t-SNE (`reduce_tsne`)

### Shared details

- All target `(n_samples, 2)`.
- For high-dimensional input (`>50` features), PCA-to-50 is applied before MDS/t-SNE (and in PCA path itself) for stability/speed.
- `random_state` defaults to 42 and is threaded through methods.

### Edge-case handling

`reduce_embedding(...)`:

- `n_samples < 2`: returns zeros (x optionally from data).
- Single-feature embeddings: x = original value, y = tiny Gaussian jitter for visibility.
- `_ensure_2d_spread(...)` adds tiny y-jitter when second dimension is nearly degenerate.

### Pipeline behavior for DR failures

In `run_pipeline(...)`, each DR method is attempted; on failure, that method falls back to PCA coordinates for that cell.

`--skip-tsne` sets the `_tsne` coordinates to a copy of PCA directly.

## 6. Cluster Representatives and Sorted Members

Source: `run_pipeline_from_embeddings.py::_cluster_representatives`

For each cluster:

- Compute cluster centroid in embedding space.
- Compute Euclidean distance from each member to centroid.
- Save:
  - `top_k_closest`
  - `top_k_farthest` (excluding closest set overlap)
  - `sorted_indices` (all members, nearest to farthest)

Exported files:

- `topk/cluster_representatives.json`
- `topk/cluster_sorted_members.json`

## 7. Sankey Construction

Sources:

- `_edges_from_transitions(...)`
- `_sankey_sample_rows(...)`

Two Sankey axes in `sankey/all_sankey.json`:

- `epoch`: transitions across adjacent layers within one epoch
- `layer`: transitions across adjacent epochs within one layer

Edge payload includes:

- source/target labels and node IDs
- `value` (count)
- `pct_of_source` (row-normalized percentage from source cluster)

Per-sample linkage for frontend interactions is in `sankey/sankey_samples.json`.

## 8. Substructure (MCS) Extraction

Sources:

- `run_pipeline_from_embeddings.py` (substructure block)
- `mcs_common.common_substructure_smarts_smiles`

Requirements:

- RDKit installed
- SMILES mapping available from `--smiles-csv` or `.pt`-embedded smiles

For each `(model, layer, cluster)` and each group:

- `knn` (nearest members)
- `kfn` (farthest members)

The backend:

1. Selects candidate `real_id`s (from top-k or full sorted list when `--substructure-max-members` is set).
2. Validates SMILES and deduplicates molecules by canonical SMILES.
3. Runs MCS search with:
   - `ring_matches_ring_only=True`
   - `complete_rings_only=True`
   - `match_valences=True`
   - atom compare by element, bond compare by order
4. Writes SMARTS/SMILES MCS plus diagnostics (`missing_real_ids`, invalid SMILES, note).

Output:

- `substructures/cluster_substructures.json`

## 9. FMI/CSPA Stability Summary

Source: `run_pipeline_from_embeddings.py` (final section)

If not disabled by `--no-fmi-cspa`, backend computes per-layer averages across consecutive epoch pairs:

- FMI: `fowlkes_mallows_score(lab_t, lab_t+1)`
- CSPA proxy: Frobenius norm of averaged co-association matrices

Output:

- `bic_aic/avg_fmi_cspa_by_layer.json`

## 10. Resume and Reuse Behavior

`run_pipeline(...)` supports checkpoint resume:

- per-cell JSON cache under `.pipeline_resume/<signature>/`
- signature includes major run parameters (clustering method, k-range, split, random state, etc.)
- disable with `--no-resume`

Optional coordinate reuse:

- `reuse_scatter_coords=True` (internal option) loads existing scatter JSON coordinates when available.

## 11. Output Naming Rules

For each `(model=epoch_key, layer)`:

- Default scatter: `scatterplot/{model}_{layer}.json`
- Method scatters: `scatterplot/{model}_{layer}_pca.json`, `_mds.json`, `_umap.json`, `_tsne.json`
- BIC/AIC: `bic_aic/{model}_{layer}.json`

For multi-method clustering (`all`):

- root = first method (usually PGMeans)
- others under `clustering/hdbscan/`, `clustering/agglomerative/`, `clustering/knn/`

## 12. Practical Notes

- DR coordinates are shared across clustering methods in the same run; only `cluster_id` changes by method.
- `real_id` is serialized as string in scatter JSON.
- Clustering outputs are based on standardized features; representatives use original flattened embedding values for centroid-distance ranking.
