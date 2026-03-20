# Backend Pipeline Runbook

This file is the execution runbook (commands and output contracts).

For implementation details (layer parsing, clustering internals, DR behavior, fallback logic), see `README_ALGORITHMS.md`.

## Primary Entry Point

Run from `latentvis/backend`:

```bash
python3 run_pipeline_from_embeddings.py --input /path/to/embeddings.pt --output-dir ./pipeline_output
```

`--input` supports:

- single `.pt` embedding file
- directory of per-layer `.pt`/`.npy` embeddings

## Quick Commands

### 1) Single input `.pt` (Chemprop/Chemeleon style)

```bash
cd /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend

python3 run_pipeline_from_embeddings.py \
  --input /global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt \
  --output-dir ./pipeline_output_chemprop \
  --k-max 10 \
  --default-reduction pca \
  --clustering-method pgmeans
```

### 2) Write directly to frontend data dir

```bash
python3 run_pipeline_from_embeddings.py \
  --input /global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt \
  --output-to-frontend \
  --k-max 8 \
  --default-reduction pca \
  --skip-tsne
```

### 3) Copy existing pipeline output to frontend

```bash
python3 scripts/copy_pipeline_to_frontend.py \
  --pipeline-dir ./pipeline_output_chemprop \
  --merge-root chemprop
```

### 4) Directory input (per-layer files)

```bash
python3 run_pipeline_from_embeddings.py \
  --input /path/to/embedding_dir \
  --output-dir ./output \
  --model-name dimenet_noaug
```

### 5) Two-input run (two epochs/models merged)

```bash
python3 run_pipeline_from_embeddings.py \
  --input /path/to/dir_or_pt_1 \
  --model-name model_a \
  --input2 /path/to/dir_or_pt_2 \
  --model-name2 model_b \
  --output-dir ./pipeline_output
```

### 6) All clustering methods in one run

```bash
python3 run_pipeline_from_embeddings.py \
  --input /path/to/embeddings.pt \
  --output-dir ./pipeline_output \
  --clustering-method all
```

## Most Important Options

- `--input`, `-i`: required input path (`.pt` or directory)
- `--output-dir`, `-o`: output root (default `./pipeline_output`)
- `--model-name`, `-m`: epoch/model name for directory input
- `--layer-file-map`: JSON map `layer_name -> filename` for directory input
- `--input2`, `--model-name2`, `--layer-file-map2`: optional second input
- `--clustering-method`: `pgmeans`, `hdbscan`, `agglomerative`, `knn`, `all`
- `--k-min`, `--k-max`: PGMeans k-range
- `--knn-k`: neighbor count for KNN clustering
- `--agg-k`, `--agg-linkage`: agglomerative overrides
- `--default-reduction`: `pca`, `mds`, `umap`, `tsne`
- `--skip-tsne`: write t-SNE file using PCA coordinates
- `--smiles-csv`, `--smiles-id-column`, `--smiles-smiles-column`: enable substructure output
- `--data-split`: `train`, `val`, or `test` for ESOL-like `.pt`
- `--output-to-frontend`: write directly to `frontend/public/data`
- `--max-epochs`: limit epochs for quick runs
- `--no-fmi-cspa`: skip FMI/CSPA summary
- `--no-resume`: disable checkpoint resume

## Output Layout (Contract)

Under `--output-dir` (or frontend data root when `--output-to-frontend`):

- `scatterplot/{model}_{layer}.json` (default projection)
- `scatterplot/{model}_{layer}_pca.json`
- `scatterplot/{model}_{layer}_mds.json`
- `scatterplot/{model}_{layer}_umap.json`
- `scatterplot/{model}_{layer}_tsne.json`
- `sankey/all_sankey.json`
- `sankey/sankey_samples.json`
- `topk/cluster_representatives.json`
- `topk/cluster_sorted_members.json`
- `bic_aic/{model}_{layer}.json`
- `bic_aic/avg_fmi_cspa_by_layer.json` (unless disabled)
- `clustering_metrics/cluster_assignments_with_real_id.csv`
- `substructures/cluster_substructures.json` (when SMILES + RDKit are available)

When `--clustering-method all` is used, additional method-specific outputs appear under:

- `clustering/hdbscan/`
- `clustering/agglomerative/`
- `clustering/knn/`

## Frontend Integration

### One-shot run + copy

```bash
./scripts/run_all_and_copy_to_frontend.sh \
  --input /path/to/embeddings.pt \
  --output-dir ./pipeline_output \
  --smiles-csv /path/to/smiles.csv
```

Copy-only mode:

```bash
./scripts/run_all_and_copy_to_frontend.sh --copy-only ./pipeline_output
```

## NERSC / SLURM Shortcuts

### Chemprop full pipeline job

```bash
./scripts/submit_chemprop_full.sh
./scripts/submit_chemprop_full.sh --skip-validate
```

### Materialize one ESOL split into backend data + compact overview

```bash
sbatch ./scripts/submit_materialize_esol_split.slurm chemprop val
sbatch ./scripts/submit_materialize_esol_split.slurm chemeleon val
```

This job writes:

- `backend/data/<dataset>/<split>/pgmeans`
- `backend/data/<dataset>/<split>/hdbscan`
- `backend/data/<dataset>/<split>/agglomerative`
- `backend/data/<dataset>/<split>/knn`
- refreshed `backend/data/<dataset>/id_smiles_y_true_y_pred.json`
- refreshed `backend/data_compact/*/overview.json` with the current compact schema
  - DR coordinates rounded to 2 decimal places
  - sparse per-node DR payloads and sparse Sankey connectivity

### Generic run+copy job

```bash
./scripts/submit_run_all_and_copy_to_frontend.sh \
  --input /path/to/embeddings.pt \
  --output-dir ./pipeline_output \
  [--smiles-csv /path/to/smiles.csv] \
  [--model-name NAME] \
  [--time 2:00:00]
```

### Update from existing scatter CSVs (no embeddings)

```bash
./scripts/submit_update_frontend_from_scatter.sh [--time 2:00:00]
```

### Full two-DimeNet pipeline + copy

```bash
./scripts/submit_run_both_dimenet_and_copy.sh [--output-dir ./pipeline_output] [--time 2:00:00]
```

Interactive equivalent:

```bash
./scripts/run_both_dimenet_and_copy_to_frontend.sh [--output-dir ./pipeline_output] [--smiles-csv /path/to/smiles.csv]
```

## Minimal Environment

- required: `numpy`, `pandas`, `scikit-learn`, `scipy`, `torch`
- optional: `umap-learn` (UMAP), `hdbscan` (HDBSCAN), `rdkit` (substructures)

Example:

```bash
module load conda
conda create -n latentvis -y python=3.9 umap-learn hdbscan scikit-learn numpy pandas scipy
conda activate latentvis
pip install torch
```
