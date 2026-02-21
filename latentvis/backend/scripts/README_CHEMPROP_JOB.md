# ESOL Case Studies: Chemprop & Chemeleon

## Overview

The existing pipeline supports both **chemprop** and **chemeleon** embeddings (same ESOL format: epoch -> split -> embeddings -> layer -> tensor). Output is written directly to the frontend data directory.

| Case | Input | Output |
|------|-------|--------|
| Chemprop | `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt` | `frontend/public/data/` (root) |
| Chemeleon | `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt` | `frontend/public/data/chemeleon/` |

## Submission

### Chemprop
```bash
cd latentvis/backend
./scripts/submit_chemprop_full.sh              # validates first, then submits
./scripts/submit_chemprop_full.sh --skip-validate   # submit without validation
sbatch scripts/submit_chemprop_full.slurm
```

### Chemeleon
```bash
cd latentvis/backend
./scripts/submit_chemeleon_full.sh
./scripts/submit_chemeleon_full.sh --skip-validate
sbatch scripts/submit_chemeleon_full.slurm
```

## Job Settings

- **Time limit**: 2 hours
- **Resources**: 1 CPU, exclusive node
- **Auto-resubmit**: If the job times out or fails before completion, it automatically submits a new job until all outputs are written.

## Output Layout (per case)

- `sankey/` - 2D Sankey (epoch × layer)
- `scatterplot/` - PCA, MDS, UMAP, t-SNE projections
- `topk/` - cluster representatives and sorted members
- `bic_aic/` - GMM model selection curves
- `clustering_metrics/` - cluster assignments CSV
- `clustering/hdbscan/`, `clustering/agglomerative/` - alternate clustering outputs
- `README.md` - data layout and frontend usage (written on successful completion)

## Environment

Uses conda env `latentvis` (Python 3.9, umap-learn, hdbscan). See main README_PIPELINE.md for setup.
