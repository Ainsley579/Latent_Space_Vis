# LatentVis Backend

This README is the lean entry point for the current backend.

Run `uv sync`; `export LATENTVIS_DATA_ROOT="$PWD/data/"`; `uv run python3 flask_backend.py`.


If you need implementation details, use:

- `README_ALGORITHMS.md` for code-accurate algorithm behavior.
- `README_PIPELINE.md` for full CLI options and workflow examples.
- `README_API.md` for Flask/backend API endpoints.
- `README_CLUSTER_CONTAINMENT.md` for user-clustering containment API.

## What This Backend Produces

From embeddings (`.pt` or directory of `.pt`/`.npy`), the backend generates frontend-ready data under an output root (usually `frontend/public/data`):

- `scatterplot/` (default + pca/mds/umap/tsne files)
- `sankey/` (`all_sankey.json`, `sankey_samples.json`)
- `topk/` (`cluster_representatives.json`, `cluster_sorted_members.json`)
- `bic_aic/` (BIC/AIC per model-layer, plus optional FMI/CSPA summary)
- `clustering_metrics/` (`cluster_assignments_with_real_id.csv`)
- `substructures/` (`cluster_substructures.json`, when SMILES + RDKit are available)

When `--clustering-method all` is used, additional method-specific outputs are written under:

- `clustering/hdbscan/`
- `clustering/agglomerative/`
- `clustering/knn/`

## Quick Start

Run from `latentvis/backend`:

```bash
python3 run_pipeline_from_embeddings.py \
  --input /path/to/embeddings.pt \
  --output-dir ./pipeline_output \
  --default-reduction mds \
  --clustering-method pgmeans
```

Write directly to frontend data directory:

```bash
python3 run_pipeline_from_embeddings.py \
  --input /path/to/embeddings.pt \
  --output-to-frontend
```

Common options:

- `--input`: `.pt` file or embedding directory (required)
- `--output-dir`: output root (default `./pipeline_output`)
- `--model-name`: epoch/model name for directory input
- `--clustering-method`: `pgmeans`, `hdbscan`, `agglomerative`, `knn`, `all`
- `--k-min`, `--k-max`: PGMeans component search range
- `--default-reduction`: `pca`, `mds`, `umap`, `tsne`
- `--skip-tsne`: write t-SNE file using PCA coordinates (faster)
- `--smiles-csv`: enable MCS substructure outputs (with RDKit)
- `--data-split`: choose `train`/`val`/`test` for ESOL-style `.pt`

For complete option documentation and examples, see `README_PIPELINE.md`.

## Current Core Files

- `run_pipeline_from_embeddings.py`: end-to-end pipeline entry point
- `embedding_loader.py`: input normalization (`.pt` and directory modes)
- `clustering.py`: clustering method implementations
- `dimensionality_reduction.py`: 2D scatter projection methods
- `mcs_common.py`: shared MCS helper logic
- `scripts/`: utility scripts for jobs, validation, copy/sync workflows

## Notes

- `real_id` is preserved through outputs and is the join key across scatter, Sankey, top-k, and substructures.
- Layer traversal order is normalized in pipeline logic (documented in `README_ALGORITHMS.md`).
- For algorithm-level details (fallback behavior, exact preprocessing, per-method internals), use `README_ALGORITHMS.md`.
