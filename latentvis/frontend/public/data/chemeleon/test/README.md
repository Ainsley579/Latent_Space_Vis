# Latent Space Visualization - Chemeleon Case

Pipeline output for **chemeleon** embeddings: `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt`.

## Directory Structure

Same as chemprop (see `../README.md` or parent for full layout):

| Path | Description |
|------|-------------|
| `sankey/all_sankey.json` | 2D Sankey: epoch and layer axes. |
| `sankey/sankey_samples.json` | Per-sample rows for Sankey-scatter linking. |
| `scatterplot/{epoch}_{layer}.json` | Default scatter (PCA). |
| `scatterplot/{epoch}_{layer}_pca.json`, `_mds.json`, `_umap.json`, `_tsne.json` | Alternate projections. |
| `topk/cluster_representatives.json` | Top-k closest/farthest per cluster. |
| `topk/cluster_sorted_members.json` | Full sorted list per cluster. |
| `bic_aic/`, `clustering_metrics/` | PGMeans curves and cluster assignments. |
| `clustering/hdbscan/`, `clustering/agglomerative/` | Alternate clustering methods. |

## Frontend Usage

Point frontend to `/data/chemeleon/` (this directory) to serve chemeleon case. Structure is identical to chemprop.

## Source

Pipeline: `latentvis/backend/run_pipeline_from_embeddings.py`
