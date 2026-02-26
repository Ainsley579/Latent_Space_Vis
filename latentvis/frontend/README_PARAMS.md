## Frontend parameters and new data layout

This doc summarizes how the frontend should interact with the updated backend and data layout. All paths and APIs use English only.

### 0. Regularized data paths (test / train)

- The backend uses a **single path convention** for both splits:
  - **Test:** `public/data/<case>/test/` (e.g. `public/data/chemprop/test/`, `public/data/chemeleon/test/`).
  - **Train:** `public/data/<case>/train/` (e.g. `public/data/chemprop/train/`, `public/data/chemeleon/train/`).
- **Frontend must use the same convention:** build the dataset base path as:
  - `/data/<case>/test` when the user selects **test** split.
  - `/data/<case>/train` when the user selects **train** split.
- Example: for test, request scatter as `/data/chemprop/test/scatterplot/0_fingerprint_pca.json`; for train, `/data/chemprop/train/scatterplot/...`. Do **not** use `/data/<case>/` without a split segment so that test and train data stay consistent.
- Three case studies: **chemprop** and **chemeleon** have full baseline data under `test/` and `train/`. **dimenet** only requires correct format and paths under `public/data/dimenet/`; content can be placeholder.

### 1. Scatterplot: coordinates vs. coloring

- **Baseline coordinates** are written once by the offline pipeline under the split folder:
  - `public/data/<case>/<split>/scatterplot/{epoch}_{layer}.json`
  - `public/data/<case>/<split>/scatterplot/{epoch}_{layer}_{projection}.json` where `projection ∈ {pca, mds, umap, tsne}`.
- All four projections (pca, mds, umap, tsne) are distinct per cell; baseline jobs do **not** use `--skip-tsne` so that t-SNE is computed and differs from PCA.
- When the user changes clustering parameters from the UI (**k range, linkage, etc.**) and the app calls
  `/generate-range`, the backend:
  - **does NOT recompute dimensionality reduction** (no PCA/MDS/UMAP/TSNE).
  - reuses baseline `(x, y)` from `scatterplot/*.json`.
  - recomputes clusters and writes **updated coloring** to (without touching baseline files):
    - `public/data/<case>/<split>/scatterplot_current/{epoch}_{layer}_{projection}.json`
- The scatter JSON shape is unchanged:

```json
[
  { "real_id": "0", "x": 0.12, "y": -0.34, "cluster_id": 3 },
  ...
]
```

#### Frontend behavior

- The helper that builds scatter URLs should use the split in the path and choose:
  - **baseline**: `/data/<case>/<split>/scatterplot/{model}_{layer}_{projection}.json`
  - **current**: `/data/<case>/<split>/scatterplot_current/{model}_{layer}_{projection}.json`
  where `<split>` is `test` or `train` (see section 0).
- State:
  - `useCurrentScatter = false` on initial load and whenever the dataset or clustering method changes.
  - On successful `/generate-range` or agglomerative re-run, `useCurrentScatter` is set to `true`, so the UI reads
    `scatterplot_current/...` (same coords, updated `cluster_id`).

### 2. Substructure (MCS) and the `SUBSTRUCTURE M` parameter (m)

- The control panel exposes **`SUBSTRUCTURE M`** (parameter **m**):
  - Meaning: per cluster, per group (knn/kfn), substructure MCS is computed using **only the nearest m** members (for knn) or **farthest m** (for kfn), i.e. "from nearest to farthest, the first m" for that cluster.
  - Default value: 32.
- This value is sent to the backend on:
  - `/generate-range` body as `substructure_max_members`.
  - agglomerative global re-run body as `substructure_max_members`.
- The backend then:
  - Uses the **full sorted member list** (`topk/cluster_sorted_members.json`) to pick:
    - nearest M for `group = "knn"`.
    - farthest M for `group = "kfn"`.
  - Runs RDKit MCS on those M members only.
  - Writes results to `substructures/cluster_substructures.json` with the same schema as before.

### 3. Layer ordering in Sankey

- For Chemprop/Chemeleon `.pt` inputs, the backend now preserves **layer order** as defined in the model:
  - `embedding_loader.load_embedding_input` no longer sorts `layer_keys` alphabetically for this case.
  - Instead, it records layers in the order they appear under `split["embeddings"]` for each epoch.
- The pipeline and Sankey output (`sankey/all_sankey.json`) respect this ordering, so:
  - `Object.keys(data.layer)` in the frontend will already be in **model layer order**, not alphabetical.
  - No extra sorting is needed on the frontend; just use the key order as-is.

### 4. Transition stats: horizontal vs vertical (optional)

- The backend script `backend/scripts/compute_layer_transition_stats.py` reads
  `clustering_metrics/cluster_assignments_with_real_id.csv` and can write **three** files per method:
  - `layer_transition_stats.json` — combined `{ by_epoch, by_layer }` (backward compatible).
  - `layer_transition_by_epoch.json` — **horizontal** transitions:
    - within a fixed epoch, consecutive layers `from_layer → to_layer`.
  - `layer_transition_by_layer.json` — **vertical** transitions:
    - within a fixed layer, consecutive epochs `from_epoch → to_epoch`.
- For interactive `/generate-range` calls, these transition stats are **skipped by default** to keep runtime low.
  They can be recomputed on demand by running the script directly or by passing
  `compute_transition_stats: true` in the JSON body if the backend exposes this option.

### 5. Baseline vs. current results (originals never overwritten)

- **Baseline** (offline pipeline):
  - Run `run_pipeline_from_embeddings.py` with `output_dir` = `public/data/<case>/test` or `public/data/<case>/train`.
  - This run computes and writes **only** into the baseline dirs:
    - `scatterplot/`, `topk/`, `substructures/`, `sankey/`, `clustering_metrics/`, `bic_aic/`.
  - Baseline JSON/CSV files are **never** overwritten by frontend-triggered operations.
- **Current** (interactive `/generate-range`):
  - All recomputed outputs are written to **separate** subdirs under the same split:
    - `scatterplot_current/` (new cluster colors, reused coords; no DR recompute),
    - `topk_current/`, `substructures_current/`, `sankey_current/`, `clustering_metrics_current/`.
  - The frontend should read from these `*_current` paths when showing results of a generate run; baseline paths remain for the initial view.

### 6. Train vs. test split

- The chemprop/chemeleon `.pt` structure is `epoch → split (train/val/test) → embeddings → layer`.
- The embedding loader accepts `data_split` / `preferred_split` and defaults to `"test"` when not set.
- The pipeline accepts `data_split` (`"test"` or `"train"`) and writes under `public/data/<case>/<split>/`.
- The control panel **DATA SPLIT** dropdown: frontend should set the dataset base path to `/data/<case>/test` or `/data/<case>/train` and include `data_split` in all API payloads (`/generate-range`, `materialize-sankey-files`, agglomerative re-run).

### 7. Cluster containment API (user clustering vs backend clustering)

- **Endpoint:** `POST /cluster-containment`
- The frontend can send a **user-defined clustering** for one cell (e.g. from a spreadsheet) and get back a **containment matrix** between user clusters and the backend’s clustering (e.g. PGMeans).
- **Request body (JSON):**
  - `dataset` (string, required): `"chemprop"` or `"chemeleon"`
  - `epoch`, `layer` (required): cell identifiers
  - `clustering` (string, optional): backend method, e.g. `"pgmeans"` or `"gmm"`
  - `data_split` (string, optional): `"test"` or `"train"`
  - `use_current` (boolean, optional): if true, use `clustering_metrics_current` and backend’s current run; otherwise baseline.
  - `user_clustering` (array, required): list of `{ "real_id": "<id>", "user_cluster_id": <0-based integer> }` (one entry per point in the cell).
- **Response (200):**
  - `user_cluster_ids`, `our_cluster_ids`: sorted lists of ids.
  - `matrix`: 2D array, rows = user clusters, cols = our clusters; value = percentage of that user cluster that falls in that our cluster.
  - `entries`: list of `{ "user_cluster_id", "our_cluster_id", "count", "percentage_of_user_cluster" }`.
- Example: if the user sends 3 clusters and the backend has 4, the result is a 3×4 matrix and 12 entries. See `latentvis/backend/README_CLUSTER_CONTAINMENT.md` for full request/response and usage.


