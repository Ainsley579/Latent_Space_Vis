# Frontend Data — How the App Uses It

This directory holds pipeline output consumed by the Latent Space Visualization frontend. Data is organized by **case study** under three folders: **`dimenet/`**, **`chemprop/`**, and **`chemeleon/`**. The app will load from one of these (e.g. `/data/dimenet/...`, `/data/chemprop/...`, `/data/chemeleon/...`) depending on which case is selected in the UI.

**Structure:** Each case folder has the **same canonical layout** (see table below). The backend copies pipeline output into the appropriate case folder via `copy_to_frontend.sh <output_dir> <case_name>` with `case_name` one of `dimenet`, `chemprop`, or `chemeleon`. The frontend (to be updated) will read from `/data/<case>/...` for the active case.

**Legacy:** `dimenet_benchmark/` is a separate, older layout (e.g. `cluster_representatives`, `scatter_plots`); it is left as-is and not used by the main pipeline copy.

---

## Data Layout (per case: `dimenet/`, `chemprop/`, `chemeleon/`)

All paths below are **relative to each case folder** (e.g. `dimenet/`, `chemprop/`, `chemeleon/`).

| Path | Description |
|------|-------------|
| **`sankey/all_sankey.json`** | 2D Sankey: epoch axis (flow across layers) and layer axis (flow across epochs). Used for the main Sankey view. |
| **`sankey/sankey_samples.json`** | Per-sample rows: `real_id`, `from_layer`, `to_layer`, `from_cluster`, `to_cluster`, `from_model`, `to_model`. |
| **`scatterplot/{model}_{layer}.json`** | Default 2D scatter for each (model, layer). |
| **`scatterplot/{model}_{layer}_pca.json`** | Same scatter, **PCA** projection. |
| **`scatterplot/{model}_{layer}_mds.json`** | Same scatter, **MDS** projection. |
| **`scatterplot/{model}_{layer}_umap.json`** | Same scatter, **UMAP** projection. |
| **`scatterplot/{model}_{layer}_tsne.json`** | Same scatter, **t-SNE** projection. |
| **`topk/cluster_representatives.json`** | Per (model, layer, cluster): `top_k_closest_real_ids`, `top_k_farthest_real_ids`, `k`. |
| **`topk/cluster_sorted_members.json`** | Per (model, layer, cluster): **full list** of `sorted_real_ids` (by distance to cluster center), plus `count`. |
| **`substructures/cluster_substructures.json`** | Per (model, layer, cluster, group): MCS substructure (`mcs_smarts`, `mcs_smiles`), `real_ids`, `k`. |
| **`bic_aic/{model}_{layer}.json`** | BIC/AIC curves for GMM model selection. |
| **`clustering_metrics/cluster_assignments_with_real_id.csv`** | `real_id`, model, per-layer cluster columns. |
| **`clustering/`** | Default clustering (e.g. GMM) at root; **`clustering/hdbscan/`** and **`clustering/agglomerative/`** when pipeline runs with multiple methods. |

---

## How the Frontend Uses These Files

### 1. Scatter plot and projection switching

- The app loads scatter data from **`/data/scatterplot/{model}_{layer}.json`** (default projection) and, when the user changes the **Projection** dropdown, from **`/data/scatterplot/{model}_{layer}_{projection}.json`** where `projection` is one of `pca`, `mds`, `umap`, `tsne`.
- **Model** and **Layer** selectors set `{model}` and `{layer}` (e.g. `dimenet_noaug`, `emb`). So the frontend expects files like:
  - `dimenet_noaug_emb.json` (default)
  - `dimenet_noaug_emb_pca.json`, `dimenet_noaug_emb_mds.json`, `dimenet_noaug_emb_umap.json`, `dimenet_noaug_emb_tsne.json`
  - and the same pattern for `dimenet10conf` and other models/layers.

### 2. Cluster members (top-k and full sorted list)

- **`/data/topk/cluster_representatives.json`** is used to show the top-k closest and farthest points per cluster (KNN/KFN).
- **`/data/topk/cluster_sorted_members.json`** provides the **full sorted list** of members per cluster (by distance to center). The app can use this to show “first N” or any range without being limited to a fixed k.

### 3. Molecule / substructure view (MCS)

- **`/data/substructures/cluster_substructures.json`** is loaded for the molecule view. Each entry includes `mcs_smarts`, `mcs_smiles` (MCS substructure, non-overlapping display), `real_ids`, and metadata per (model, layer, cluster, group).

### 4. Sankey

- The main Sankey view uses **`/data/sankey/all_sankey.json`**.
- Sample-level Sankey data is loaded from paths like **`/data/sankey/noaug_sankey_samples.json`** or **`/data/sankey/10conf_sankey_samples.json`** depending on model selection.

### 5. BIC/AIC

- BIC/AIC charts request **`/data/bic_aic/{model}_{layer}.json`** for the selected model and layer.

### 6. Multiple clustering methods

- The **backend** can run several clustering methods (**GMM**, **HDBSCAN**, **Agglomerative**). Default method is at the case root; others under **`clustering/hdbscan/`** and **`clustering/agglomerative/`** with the same layout. The **frontend** can be extended to load from `clustering/{method}/` when switching method.

---

## Multiple scatter methods (projections)

Every pipeline run that does **not** use `--skip-tsne` produces **five** scatter files per (model, layer):

- **Default** (e.g. MDS): `{model}_{layer}.json`
- **PCA**: `{model}_{layer}_pca.json`
- **MDS**: `{model}_{layer}_mds.json`
- **UMAP**: `{model}_{layer}_umap.json`
- **t-SNE**: `{model}_{layer}_tsne.json`

The frontend **Projection** dropdown (Default/MDS, PCA, MDS, UMAP, t-SNE) switches the request to the matching file. This applies to any case study as long as the data is under `/data/<case>/scatterplot/` with that naming.

---

## Case studies: dimenet, chemprop, chemeleon

- **`dimenet/`** — DimeNet outputs. Pipeline: `run_both_dimenet_and_copy_to_frontend.sh`; copy: `copy_to_frontend.sh <output_dir> dimenet`. "Update from scatter" path: `update_frontend_data_from_existing.sh` (writes to `dimenet/topk`, `dimenet/substructures`; set `CASE_NAME=dimenet` if needed).
- **`chemprop/`** — Chemprop outputs. Same canonical layout. Use `run_all_and_copy_to_frontend.sh --case chemprop` (and optionally `--copy-only <dir> --case chemprop`). See **`chemprop/README.md`** for Chemprop-specific layout.
- **`chemeleon/`** — Chemeleon outputs. Same canonical layout. Use `run_all_and_copy_to_frontend.sh --case chemeleon` (and optionally `--copy-only <dir> --case chemeleon`).

Each case has: **multiple scatter projections** (default + `_pca`, `_mds`, `_umap`, `_tsne`), **topk** (cluster_representatives + cluster_sorted_members), **MCS substructures**, **sankey**, **bic_aic**, **clustering_metrics**, and **clustering/** (with optional hdbscan/agglomerative). The **frontend** will need to be updated to load from `/data/<case>/...` for the selected case.

---

## Source and Regeneration

- **Backend pipeline**: `latentvis/backend/run_pipeline_from_embeddings.py` (and scripts in `latentvis/backend/scripts/`).
- **Copy to frontend**: `latentvis/backend/scripts/copy_to_frontend.sh <output_dir> [case_name]` with `case_name` one of `dimenet`, `chemprop`, or `chemeleon`. Used by `run_both_dimenet_and_copy_to_frontend.sh` (dimenet) and `run_all_and_copy_to_frontend.sh --case <name>`.
- See **`latentvis/backend/README_PIPELINE.md`** for how to run the pipeline and submit jobs.
