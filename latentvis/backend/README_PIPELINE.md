# Latent Space Vis Backend Pipeline

## Entry point: embedding input → frontend JSON

The backend accepts **a single embedding input** (either one `.pt` file or a directory of per-layer `.pt`/`.npy` files), runs all analyses, and writes the JSON/assets the frontend expects.

### Single `.pt` file (e.g. chemprop / chemeleon)

Supported structures:

- **Nested by epoch and layer**: `dict[epoch_key][layer_key] = tensor`  
  Each tensor shape: `(n_samples, n_features)`. Epoch keys can be epoch indices or names; layer keys are layer names.
- **Flat by layer**: `dict[layer_key] = tensor`  
  Treated as a single epoch named `"default"`.
- **Explicit**: `dict` with keys `"epochs"`, `"layers"`, `"embeddings"`  
  `embeddings` can be 4D `(n_epochs, n_layers, n_samples, dim)` or 2D `(n_samples, dim)`.

### Directory

- Directory of `.pt` or `.npy` files: each file stem is used as the layer name; one epoch key (use `--model-name` or directory name).
- Optional `layer_file_map`: map layer names to filenames (e.g. `emb` → `emb.pt`).

## Run the full pipeline

```bash
cd /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend

# Chemprop case (single .pt, esol structure: epoch -> test/val/train -> embeddings -> layer -> tensor)
python3 run_pipeline_from_embeddings.py \
  --input /global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt \
  --output-dir ./pipeline_output_chemprop \
  --k-max 10 \
  --default-reduction pca \
  --clustering-method gmm

# Optional: all 3 clustering methods (GMM to root, HDBSCAN and Agglomerative under clustering/)
# --clustering-method all

# Option A: Write directly to frontend (no copy)
python3 run_pipeline_from_embeddings.py \
  --input /global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt \
  --output-to-frontend \
  --k-max 8 --default-reduction pca --skip-tsne

# Option B: Copy to frontend after run
python3 scripts/copy_pipeline_to_frontend.py --pipeline-dir ./pipeline_output_chemprop --merge-root chemprop

# Submit full chemprop as SLURM job (2hr, single CPU, exclusive node, auto-resubmit until complete)
./scripts/submit_chemprop_full.sh              # runs validation first
./scripts/submit_chemprop_full.sh --skip-validate

python run_pipeline_from_embeddings.py \
  --input /global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt \
  --output-dir ./output_chemeleon

# From a directory of per-layer .pt files
python run_pipeline_from_embeddings.py \
  --input /path/to/embedding_dir \
  --output-dir ./output \
  --model-name dimenet_noaug
```

### Options

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Path to `.pt` file or directory (required) |
| `--output-dir`, `-o` | Output directory (default: `./pipeline_output`) |
| `--model-name`, `-m` | Model/epoch name when input is a directory |
| `--k-min`, `--k-max` | GMM component range for BIC/AIC (default 2–15) |
| `--default-reduction` | Method for default scatter file: `pca`, `mds`, `umap`, `tsne` (default: `mds`) |
| `--smiles-csv` | CSV with `real_id` and SMILES for MCS substructures (optional) |
| `--smiles-id-column` | Column name for IDs (default: `real_id`) |
| `--smiles-smiles-column` | Column name for SMILES (default: `smiles`) |
| `--layer-file-map` | Path to JSON: layer name → filename (e.g. `{"emb": "emb.pt"}`) for directory input |
| `--input2`, `--model-name2`, `--layer-file-map2` | Second embedding dir (second epoch); merged with first for Sankey and output |
| `--clustering-method` | `gmm` (default), `hdbscan`, `agglomerative`, or `all`. With `all`, root gets GMM and `clustering/hdbscan/`, `clustering/agglomerative/` get the same layout for the other methods. |
| `--output-to-frontend` | Write directly to `frontend/public/data` (no copy step). |
| `--max-epochs` | Limit to first N epochs (e.g. `10` for quicker run). |
| `--skip-tsne` | Skip t-SNE (use PCA for tsne coords); much faster. |

### Clustering methods (3)

- **GMM** – Gaussian Mixture Model with BIC/AIC model selection (current default). Writes BIC/AIC curves.
- **HDBSCAN** – density-based; no fixed k. Noise label -1 is remapped so Sankey/representatives work. Requires `pip install hdbscan`.
- **Agglomerative** – ward linkage; uses same k as GMM (from BIC) when run alone, or from k_min/k_max when run with `all`.

Use `--clustering-method all` to generate frontend-ready output for all three: root = GMM, `clustering/hdbscan/` and `clustering/agglomerative/` have the same sankey, scatterplot, topk, etc.

**No fallback**: If any scatter method (PCA, MDS, UMAP, t-SNE) or clustering method fails, the pipeline exits with error.

**Environment**: For multi-method runs (UMAP + 3 clustering), use conda env `latentvis`:

```bash
module load conda
conda create -n latentvis -y python=3.9 umap-learn hdbscan scikit-learn numpy pandas scipy
conda activate latentvis
pip install torch   # for loading .pt files
```

(HDBSCAN does not build on Python 3.6; use Python 3.9+.)

## Pipeline is model-agnostic (all case studies)

The three behaviors below are **permanent** and apply to **any** embedding/model, not only DimeNet:

1. **Multiple scatter projections** – For every `(epoch, layer)` the pipeline writes default + PCA, MDS, UMAP, t-SNE. No hardcoded model names.
2. **Full sorted list per cluster** – `topk/cluster_sorted_members.json` is always written (full list sorted by distance to center) for every run.
3. **MCS substructures (non-overlapping)** – When `--smiles-csv` is provided, substructures use `mcs_common.common_substructure_smarts_smiles` for all clusters.

For **other case studies**: use the same entry points with your paths and names:

- **Single model**: `run_all_and_copy_to_frontend.sh --input /path/to/your_embeddings.pt --output-dir ./out [--model-name your_model] [--smiles-csv ...]`. For a directory of `.pt` files with non-standard names, add a JSON layer map and use `run_pipeline_from_embeddings.py` with `--layer-file-map /path/to/layer_map.json`.
- **Two epochs (e.g. two models)**: `run_pipeline_from_embeddings.py --input dir1 --model-name name1 [--layer-file-map map1.json] --input2 dir2 --model-name2 name2 [--layer-file-map2 map2.json] --output-dir ./out ...`, then `copy_to_frontend.sh ./out`.
- **Update from existing scatter CSVs** (no embeddings): set env `SCATTER_TENCONF`, `SCATTER_NOAUG`, `MODEL_NAME_1`, `MODEL_NAME_2`, `SMILES_CSV`, `SMILES_SMILES_COLUMN`. Default for the update script is `product_smiles` (DimeNet); for other case studies set `SMILES_SMILES_COLUMN=smiles`. Then run `update_frontend_data_from_existing.sh`.

## Output layout (under `--output-dir`)

- **`sankey/all_sankey.json`** – Two axes: `epoch` (flow across layers) and `layer` (flow across epochs). Frontend uses this for the 2D Sankey (epoch vs layer).
- **`sankey/sankey_samples.json`** – Per-sample rows for linking Sankey to scatter (real_id, from_layer, to_layer, from_cluster, to_cluster, from_model, to_model).
- **`scatterplot/{model}_{layer}.json`** – Default 2D scatter (method set by `--default-reduction`).
- **`scatterplot/{model}_{layer}_pca.json`**, **`_mds.json`**, **`_umap.json`**, **`_tsne.json`** – Same scatter with PCA, MDS, UMAP, t-SNE so the frontend can switch methods.
- **`bic_aic/{model}_{layer}.json`** – BIC/AIC curves (k_values, aic, bic).
- **`topk/cluster_representatives.json`** – Per (model, layer, cluster): top-k closest and farthest real_ids (KNN-style data points).
- **`topk/cluster_sorted_members.json`** – Per (model, layer, cluster): **full list** of real_ids in that cluster, **sorted by distance to cluster center (closest first)**. Frontend can slice arbitrarily (e.g. first 5, first 10, range 0–20) without being limited to a fixed k.
- **`substructures/cluster_substructures.json`** – MCS substructures per cluster (knn/kfn) when `--smiles-csv` and RDKit are available.
- **`clustering_metrics/cluster_assignments_with_real_id.csv`** – real_id, epoch, and per-layer cluster columns.

## Frontend usage

Point the frontend’s data base URL to the pipeline output (e.g. copy output into `latentvis/frontend/public/data/` or serve `output-dir` as `/data`). The frontend expects:

- `/data/scatterplot/{model}_{layer}.json` – default projection (see `--default-reduction`)
- `/data/scatterplot/{model}_{layer}_pca.json`, `_mds.json`, `_umap.json`, `_tsne.json` – alternate projections. The UI has a **Projection** dropdown (Default/MDS, PCA, MDS, UMAP, t-SNE) to switch.
- `/data/topk/cluster_representatives.json`, `/data/topk/cluster_sorted_members.json` – cluster members (top-k and full sorted list).
- `/data/substructures/cluster_substructures.json` – MCS substructures when `--smiles-csv` was provided.

## One-shot: run pipeline and copy to frontend

From `latentvis/backend` or `latentvis/backend/scripts`:

```bash
./scripts/run_all_and_copy_to_frontend.sh --input /path/to/embeddings.pt --output-dir ./pipeline_output --smiles-csv /path/to/smiles.csv
```

Optional: `--model-name NAME` when input is a directory. To only refresh frontend from an existing pipeline output (no re-run):

```bash
./scripts/run_all_and_copy_to_frontend.sh --copy-only ./pipeline_output
```

## Submit pipeline as a NERSC job (single CPU)

To run the full pipeline (including PCA/MDS/UMAP/t-SNE scatter) on a single-CPU node and then copy to frontend:

```bash
cd latentvis/backend
./scripts/submit_run_all_and_copy_to_frontend.sh --input /path/to/embeddings.pt --output-dir ./pipeline_output [--smiles-csv /path/to/smiles.csv] [--model-name NAME] [--time 2:00:00]
```

- **`--input`** – Path to `.pt` file or directory of per-layer embeddings (required).
- **`--output-dir`** – Pipeline output directory (default: `./pipeline_output`).
- **`--smiles-csv`** – CSV with `real_id` and SMILES for MCS (optional).
- **`--model-name`** – Model/epoch name when input is a directory (optional).
- **`--time`** – SLURM time limit, e.g. `2:00:00` (default: 2 hours).

The job runs on one CPU (`--cpus-per-task=1`, `--constraint=cpu`). Logs go to `latentvis/backend/logs/slurm/<jobname>_<jobid>.out`. When the job completes, frontend data is updated (scatterplot with all projections, topk, cluster_sorted_members, substructures).

## Two jobs: update from scatter vs full pipeline (both dimenet, 2h, auto-resubmit)

**1. Update frontend from existing scatter CSVs** (no embeddings; refreshes topk + cluster_sorted_members + substructures from `data/scatter_plots/` and SMILES):

```bash
cd latentvis/backend
./scripts/submit_update_frontend_from_scatter.sh [--time 2:00:00]
```

Runs `update_frontend_data_from_existing.sh` in a 2h job. If the run does not finish in time or fails (exit ≠ 0), the job **resubmits itself** so it keeps retrying until complete.

**2. Full pipeline for both DimeNet embeddings** (dimenet_noaug + dimenet10conf, same as old results), then copy to frontend:

```bash
cd latentvis/backend
./scripts/submit_run_both_dimenet_and_copy.sh [--output-dir ./pipeline_output] [--time 2:00:00]
```

Uses fixed paths:

- **dimenet_noaug**: `latent-spaces/dimenet_noaug_embeddings_cas_152` (layer map: `scripts/layer_map_dimenet_noaug_cas.json`)
- **dimenet10conf**: `latent-spaces/dimenet10conf_embeddings` (layer map: `scripts/layer_map_dimenet10conf.json`)

Runs `run_both_dimenet_and_copy_to_frontend.sh` (single pipeline run with two epochs, merged Sankey and scatter/topk/substructures). If the run fails or times out, the job **resubmits itself** (2h each time) until complete.

To run the same pipeline **interactively** (no job):

```bash
./scripts/run_both_dimenet_and_copy_to_frontend.sh [--output-dir ./pipeline_output] [--smiles-csv /path/to/smiles.csv]
```

## Dependencies

- NumPy, pandas, scikit-learn, scipy
- PyTorch (for `.pt` loading)
- Optional: `umap-learn` (UMAP), `rdkit` (substructures)

Install optional: `pip install umap-learn rdkit`
