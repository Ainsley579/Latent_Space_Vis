## Frontend parameters and new data layout

This doc summarizes how the frontend should interact with the updated backend and data layout.

### 1. Scatterplot: coordinates vs. coloring

- **Baseline coordinates** are written once by the offline pipeline:
  - `public/data/<case>/scatterplot/{epoch}_{layer}.json`
  - `public/data/<case>/scatterplot/{epoch}_{layer}_{projection}.json` where `projection ∈ {pca, mds, umap, tsne}`.
- When the user changes clustering parameters from the UI (**k range, linkage, etc.**) and the app calls
  `/generate-range`, the backend:
  - **does NOT recompute dimensionality reduction** (no PCA/MDS/UMAP/TSNE).
  - reuses baseline `(x, y)` from `scatterplot/*.json`.
  - recomputes clusters and writes **updated coloring** to:
    - `public/data/<case>/scatterplot_current/{epoch}_{layer}_{projection}.json`
- The scatter JSON shape is unchanged:

```json
[
  { "real_id": "0", "x": 0.12, "y": -0.34, "cluster_id": 3 },
  ...
]
```

#### Frontend behavior

- The helper `buildScatterPath(model, layer, projection, dataset, clustering)` now chooses between:
  - **baseline**: `/data/<case>/scatterplot/{model}_{layer}_{projection}.json`
  - **current**: `/data/<case>/scatterplot_current/{model}_{layer}_{projection}.json`
- State:
  - `useCurrentScatter = false` on initial load and whenever the dataset or clustering method changes.
  - On successful `/generate-range` or agglomerative re-run, `useCurrentScatter` is set to `true`, so the UI reads
    `scatterplot_current/...` (same coords, updated `cluster_id`).

### 2. Substructure (MCS) and the `SUBSTRUCTURE M` parameter

- The control panel exposes **`SUBSTRUCTURE M`**:
  - Meaning: per cluster, per group (knn/kfn), only the first **M members** are used for substructure MCS.
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

### 5. Baseline vs. current results

- **Baseline** (offline pipeline):
  - Run `run_pipeline_from_embeddings.py` with `output_dir` pointing to a dedicated baseline folder (e.g.
    `latentvis/frontend/public/data/chemprop_baseline` or `.../chemprop/test/baseline`).
  - This run computes:
    - all scatter projections (coords + initial `cluster_id`),
    - topk,
    - substructures,
    - Sankey,
    - clustering metrics.
  - Baseline folders should not be touched by interactive operations.
- **Current** (interactive `/generate-range`):
  - Writes updated results into the standard case layout under `public/data/<case>/...`:
    - `scatterplot_current/` (new cluster colors, reused coords),
    - `topk/`, `substructures/`, `sankey/`, `clustering_metrics/`, and transition stats.
  - The frontend reads from `/data/<case>/...` for the **current** view; baseline is kept only as a reference.

### 6. Train vs. test split

- The chemprop/chemeleon `.pt` structure is `epoch → split (train/val/test) → embeddings → layer`.
- The embedding loader now accepts an explicit `data_split` / `preferred_split` and still defaults to
  `"test"` (then `"val"`, `"train"`) when not specified, so existing behavior is preserved.
- The pipeline accepts a `data_split` argument (`"test"` or `"train"`) and restricts loading to that split.
  - Test baseline: `public/data/chemprop/...`, `public/data/chemeleon/...`
  - Train baseline: `public/data/chemprop/train/...`, `public/data/chemeleon/train/...`
- The control panel exposes a **DATA SPLIT** dropdown (`test` / `train`):
  - `datasetBaseFor(dataset, clustering, dataSplit)` incorporates the split into the base path so that
    URLs under `/data/...` are routed to the correct split-specific directory.
  - API payloads (`/generate-range`, `materialize-sankey-files`, agglomerative re-run) now include `data_split`
    so the backend can read the correct split from the shared `.pt` embeddings.


