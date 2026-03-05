# Latent Space Visualization - Frontend Data (Chemprop)

This directory contains pipeline output for the **chemprop** case: embeddings from `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt`.

## Directory Structure

| Path | Description |
|------|-------------|
| `sankey/all_sankey.json` | 2D Sankey: epoch (flow across layers) and layer (flow across epochs). |
| `sankey/sankey_samples.json` | Per-sample rows: real_id, from_layer, to_layer, from_cluster, to_cluster. |
| `scatterplot/{epoch}_{layer}.json` | Default 2D scatter (method: PCA). |
| `scatterplot/{epoch}_{layer}_pca.json` | PCA projection. |
| `scatterplot/{epoch}_{layer}_mds.json` | MDS projection. |
| `scatterplot/{epoch}_{layer}_umap.json` | UMAP projection. |
| `scatterplot/{epoch}_{layer}_tsne.json` | t-SNE projection (or PCA when --skip-tsne). |
| `topk/cluster_representatives.json` | Per (epoch, layer, cluster): top-k closest/farthest real_ids. |
| `topk/cluster_sorted_members.json` | Per (epoch, layer, cluster): full list of real_ids sorted by distance to center. |
| `bic_aic/{epoch}_{layer}.json` | BIC/AIC curves for GMM model selection. |
| `clustering_metrics/cluster_assignments_with_real_id.csv` | real_id, epoch, per-layer cluster columns. |
| `clustering/hdbscan/` | Same layout for HDBSCAN clustering. |
| `clustering/agglomerative/` | Same layout for Agglomerative clustering. |

## Frontend Usage

- **Scatter**: Use `/data/scatterplot/*.json`. Switch projection via dropdown (PCA, MDS, UMAP, t-SNE).
- **Sankey**: Use `/data/sankey/all_sankey.json` and `sankey_samples.json`.
- **Clusters**: Use `/data/topk/cluster_representatives.json` and `cluster_sorted_members.json`.
- **Molecule view**: `real_id` in scatter/topk links to molecule lookups.

## Source

Pipeline: `latentvis/backend/run_pipeline_from_embeddings.py`  
Input: chemprop-embeddings.pt (ESOL structure: epoch -> split -> embeddings -> layer -> tensor)
