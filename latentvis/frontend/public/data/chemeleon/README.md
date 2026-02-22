# Latent Space Visualization - Chemeleon Case

Pipeline output for **chemeleon** embeddings: `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt`.

## Scatterplot Methods (projections)

Each (epoch, layer) has **five** scatter files:

- `{epoch}_{layer}.json` — Default (PCA)
- `{epoch}_{layer}_pca.json` — PCA projection
- `{epoch}_{layer}_mds.json` — MDS projection  
- `{epoch}_{layer}_umap.json` — UMAP projection
- `{epoch}_{layer}_tsne.json` — t-SNE (or PCA when `--skip-tsne`)

The frontend **Projection** dropdown switches among these. File format: `[{real_id, x, y, cluster_id}, ...]`.

## Clustering Methods

- **GMM** (default) — at root: `sankey/`, `scatterplot/`, `topk/`, `bic_aic/`, `clustering_metrics/`
- **HDBSCAN** — under `clustering/hdbscan/` (same layout)
- **Agglomerative** — under `clustering/agglomerative/` (same layout)

## Directory Structure

| Path | Description |
|------|-------------|
| `sankey/all_sankey.json` | 2D Sankey: epoch and layer axes. |
| `sankey/sankey_samples.json` | Per-sample rows for Sankey-scatter linking. |
| `scatterplot/{epoch}_{layer}.json` | Default scatter (PCA). |
| `scatterplot/{epoch}_{layer}_pca.json`, `_mds.json`, `_umap.json`, `_tsne.json` | Alternate projections. |
| `topk/cluster_representatives.json` | Per (epoch, layer, cluster): top-k closest/farthest real_ids. |
| `topk/cluster_sorted_members.json` | Full sorted list per cluster. |
| `bic_aic/`, `clustering_metrics/` | GMM curves and cluster assignments. |
| `clustering/hdbscan/`, `clustering/agglomerative/` | HDBSCAN and Agglomerative clustering (same layout). |

## Frontend Usage

Load from **`/data/chemeleon/`** (this directory). The frontend can switch projection (PCA, MDS, UMAP, t-SNE) and clustering method (GMM, HDBSCAN, Agglomerative). Structure is identical to chemprop; see parent `../README.md` for full layout and API.

## Source

Pipeline: `latentvis/backend/run_pipeline_from_embeddings.py`
