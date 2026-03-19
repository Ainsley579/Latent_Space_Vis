## 0) Common Entry Points

### Frontend
- `latentvis/frontend/app/page.tsx`
  - `buildDataFileUrl(...)`
  - `buildAllSankeyPath(...)`
  - `buildSankeyModelPath(...)`
  - `buildVerticalPath(...)`
  - `buildScatterPaths(...)`

### Backend
- `latentvis/backend/flask_backend.py`
  - `data_file()` (`/data-file`)
  - `materialize_sankey_files()` (`/materialize-sankey-files`)
  - `cluster_tanimoto()` (`/cluster-tanimoto`)
  - `substructures_cluster_topm()` (`/substructures/cell-topm`, `/substructures/cluster-topm`)

---

1. 2D Sankey mismatch:
- `SankeyView` -> `data_file` -> `.../sankey/all_sankey.json`

1. Horizontal/Vertical mismatch:
- `SankeyHorizontal/SankeyVertical` -> `data_file` -> compat sample JSONs
- if missing, verify in-memory materialization from `sankey_samples + all_sankey`

1. Scatter missing/slow:
- `ScatterPlot` -> `data_file` -> `_load_xy_all_payload` -> `.../xy_all.json`

1. Top-k/member order wrong:
- `TopkNearestPanel/page.tsx` -> `data_file` -> `.../topk/cluster_sorted_members.json`

1. Substructure/Tanimoto empty:
- `/substructures/cell-topm` or `/cluster-tanimoto`
- verify `_load_id_to_smiles` can read backend `id_smiles_y_true_y_pred.json`

---

## 1) 2D Sankey (`SankeyView`)

### Call chain
- Frontend: `SankeyView` (`latentvis/frontend/app/component/SankeyView.tsx`)
- Frontend request from `page.tsx`: `buildAllSankeyPath(...)`
- Backend function: `data_file()`
- Backend file: `.../sankey/all_sankey.json`

### Notes
- `showFfn2` filtering is done in frontend.
- For split=`all`, frontend loads an extra test `all_sankey` and merges.

---

## 2) Horizontal Sankey (`SankeyHorizontal`)

### Call chain
- Frontend: `SankeyHorizontal` (`latentvis/frontend/app/component/SankeyHorizontal.tsx`)
- Frontend request from `page.tsx`: `buildSankeyModelPath(model)`
- Backend function:
  - `data_file()` tries `<model>_sankey_samples.json`
  - if missing, `data_file()` uses `_materialize_compat_sankey_files(..., write_files=False)` to build in memory (the current system should follow this)
- Backend file:
  - `.../sankey/<model>_sankey_samples.json`
  - or dynamic source: `.../sankey/sankey_samples.json + .../sankey/all_sankey.json` (the current system should follow this)

### File schema
`sankey_samples.json` row:
- `real_id: string | number`
- `from_layer: string`
- `to_layer: string`
- `from_cluster: number`
- `to_cluster: number`
- `from_model: string`
- `to_model: string`

`all_sankey.json`:
- Top-level:
  - `epoch: { [model]: { type, epoch, edges[] } }`
  - `layer: { [layer]: { type, layer, edges[] } }`
- `edges[]` item:
  - `source_node_id: number`
  - `target_node_id: number`
  - `source_label: string` (e.g. `fingerprint_C0`)
  - `target_label: string` (e.g. `gnn_layer_1_C1`)
  - `value: number`
  - `pct_of_source: number`

### Notes
- For split=`all`, train/test rows are merged.

---

## 3) Vertical Sankey (`SankeyVertical`)

### Call chain
- Frontend: `SankeyVertical` (`latentvis/frontend/app/component/SankeyVertical.tsx`)
- Frontend request from `page.tsx`: `buildVerticalPath(layer)`
- Backend function:
  - `data_file()` reads `vertical_model_samples_<layer>.json`
  - if missing, same in-memory materialization fallback (the current system should follow this)
- Backend file:
  - `.../sankey/vertical_model_samples_<layer>.json`
  - or dynamic source: `.../sankey/sankey_samples.json + .../sankey/all_sankey.json` (the current system should follow this)
---

## 4) ScatterPlot (`ScatterPlot`)

### Call chain A: XY coordinates
- Frontend: `ScatterPlot` (`latentvis/frontend/app/component/ScatterPlot.tsx`)
- Frontend request from `page.tsx`: `buildScatterPaths(...).xy`
- Backend function: `data_file()`
- Backend internal functions:
  - `_scatter_parse_request_filename(...)`
  - `_scatter_bundle_lookup_rows(...)`
  - for `kind=xy`, prioritizes `_load_xy_all_payload(...)`
- Backend file:
  - `latentvis/backend/data/<dataset>/xy_all.json` 
  - or split parts `xy_all.part*.json` / `xy_all_*.json` (only for chemeleon as one file is too big for git)

### File schema
`xy_all.json`:
- Top-level:
  - `ok, dataset, kind, projections, models, layers`
  - `cells: { "<model>__<layer>": { pca/mds/umap/tsne: { xy: [...] } } }`
  - optional `cell_meta`
- `xy` row:
  - `real_id: string`
  - `x: number`
  - `y: number`
  - `split: "train" | "test"`

### Call chain B: cluster coloring
- Frontend request: `buildScatterPaths(...).cluster`
- Backend function: `data_file()`
- Backend internal:
  - tries `_scatter_bundle_lookup_rows(..., kind='cluster')`
  - otherwise falls back to `_scatter_cluster_rows_from_sankey_samples(...)` / `_scatter_compute_cluster_rows(...)`
- Backend source file:
  - mainly `.../clustering_metrics/cluster_assignments_with_real_id.csv`
  - fallback `.../sankey/sankey_samples.json`

### File schema
`cluster_assignments_with_real_id.csv`:
- Key columns:
  - `real_id`
  - `epoch`
  - `<layer>_cluster` columns (e.g. `fingerprint_cluster`, `gnn_layer_1_cluster`)

### Call chain C: split + y_true/y_pred metadata
- Frontend metadata request (`splitMapPath`):
  - dimenet: `/dimnet/id_smiles_y_true_y_pred.json`
  - others: `/id_smiles_y_true_y_pred.json`
- Usage in frontend:
  - split coloring
  - y_true / y_pred / log-diff coloring
  - tooltip values
  - CSV export

### File schema
`id_smiles_y_true_y_pred.json` (backend canonical source):
- Paths:
  - `latentvis/backend/data/chemprop/id_smiles_y_true_y_pred.json`
  - `latentvis/backend/data/dimenet/id_smiles_y_true_y_pred.json`
- Row fields:
  - `real_id` (or `id`)
  - `smiles`
  - `split`
  - `y_true`
  - `y_pred` or `y_pred_by_epoch` (dataset-dependent)

---

## 5) Top-k Panel + Cluster Members

### Call chain A: member ordering
- Frontend: `page.tsx` + `TopkNearestPanel.tsx`
- Frontend request: `/data-file?...section=topk&file=cluster_sorted_members.json`
- Backend function: `data_file()`
- Backend file: `.../topk/cluster_sorted_members.json`

### File schema
`cluster_sorted_members.json` row:
- `layer_name: string`
- `model_type: string`
- `cluster_id: number`
- `sorted_real_ids: string[]`
- optional `count`

### Notes
- Frontend top-k nearest/farthest are slices derived from `sorted_real_ids`.
- Cluster member display order follows this same sequence.

### Call chain B: substructure
- Frontend request: `/substructures/cluster-topm`
- Backend function: `substructures_cluster_topm()`
- Backend reads:
  - `cluster_sorted_members.json`
  - smiles map via `_load_id_to_smiles()` -> backend `id_smiles_y_true_y_pred.json`
- Backend response rows include:
  - `real_ids`, `real_id_smiles`, `mcs_smarts`, `mcs_smiles`, etc.

### Call chain C: tanimoto
- Frontend request: `/cluster-tanimoto`
- Backend function: `cluster_tanimoto()`
- Backend reads:
  - cluster rows from assignments (`_load_cluster_rows_for_cell`)
  - smiles mapping from `_load_id_to_smiles()` -> backend `id_smiles_y_true_y_pred.json`
- Backend response:
  - per-cluster tanimoto summary

---

## 6) KNN Special Path Rule

### Call chain
- Frontend sends `clustering=knn`; after Generate it includes `k` in `/data-file` requests.
- Backend function: `data_file()`

### Path resolution
- First try:
  - `.../<split>/knn/<section>/<file>`
- If missing, backend requires `k`, then tries:
  - `.../<split>/sweeps/knn/k_<k>/<section>/<file>`
- If still missing, returns 404 (no extra fallback).

---
