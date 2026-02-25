# Flask Backend API

The Flask app in `flask_backend.py` exposes HTTP endpoints for the frontend. Default base URL: `http://127.0.0.1:5001` (set `NEXT_PUBLIC_FLASK_API_BASE` in the frontend to override).

CORS is enabled for `*` so the frontend can call from any origin.

---

## Endpoints

### GET /health

- **Response:** `{ "ok": true }`
- Use to check that the backend is up.

---

### POST /search-smiles

Search for real_ids by SMILES string (exact match after stripping whitespace).

- **Request body (JSON):**
  - `smiles` (string, required): SMILES to search, e.g. `"C1CCCC1"`
  - `dataset` (string, optional): `"chemprop"` or `"chemeleon"`; if omitted, both are searched.

- **Response (200):**
  ```json
  {
    "ok": true,
    "smiles": "C1CCCC1",
    "matches": {
      "chemprop": ["real_id_1", "real_id_2"],
      "chemeleon": []
    },
    "total_ids": 2
  }
  ```
- **Errors:** 400 if `smiles` is missing or empty.

Data source: `frontend/public/data/<case>/real_id_to_smiles.json` (inside the project).

---

### POST /generate-range

Filter Sankey data to a given epoch range [min, max]. Updates `all_sankey.json` and `sankey_samples.json` in place for the given case(s).

- **Request body (JSON):**
  - `min` (int, required): minimum epoch (inclusive)
  - `max` (int, required): maximum epoch (inclusive)
  - `dataset` (string, optional): `"chemprop"` or `"chemeleon"`; if omitted, both cases are updated.

- **Response (200):**
  ```json
  {
    "ok": true,
    "min": 1,
    "max": 2,
    "summary": {
      "chemprop": { "kept_epochs": 2, "kept_sample_rows": 1200 },
      "chemeleon": { "kept_epochs": 2, "kept_sample_rows": 800 }
    }
  }
  ```
- **Errors:** 400 if `min`/`max` are not valid integers.

After a successful call, the frontend should reload Sankey data (or refetch `all_sankey.json` / `sankey_samples.json`) to show the filtered view.

---

### POST /recluster-agglomerative

Re-run **Agglomerative clustering** for a single (dataset, epoch, layer) with user-chosen **k** and **linkage**. Default pipeline data uses **Ward** linkage and **BIC-selected k**; this API allows the frontend to override both and get new cluster assignments.

- **Request body (JSON):**
  - `dataset` (string, required): `"chemprop"` or `"chemeleon"`
  - `epoch` (string or number, required): e.g. `"0"`, `"99"`
  - `layer` (string, required): e.g. `"fingerprint"`, `"gnn_layer_1"`
  - `k` (int, optional): number of clusters. If omitted or `null`, the backend uses BIC (GMM) to choose k in [2, 15].
  - `linkage` (string, optional): `"ward"` (default), `"complete"`, or `"average"`.

- **Response (200):**
  ```json
  {
    "ok": true,
    "dataset": "chemprop",
    "epoch": "0",
    "layer": "gnn_layer_1",
    "k": 5,
    "linkage": "ward",
    "real_ids": ["id1", "id2", ...],
    "labels": [0, 1, 0, 2, ...],
    "assignments": [
      { "real_id": "id1", "cluster_id": 0 },
      { "real_id": "id2", "cluster_id": 1 },
      ...
    ]
  }
  ```
  - `real_ids`: order matches the embedding rows.
  - `labels`: cluster index per sample (same order as `real_ids`).
  - `assignments`: same data as list of `{ real_id, cluster_id }` for convenience.

- **Errors:**
  - 400: `invalid_dataset` or `missing_epoch_or_layer`
  - 404: `embedding_not_found` (e.g. .pt file missing) or `cell_not_found` (epoch/layer not in .pt)
  - 500: `clustering_failed` (e.g. invalid k or sklearn error)

**Embedding data:** The backend loads embeddings from the same .pt files used by the pipeline:
- Chemprop: `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt`
- Chemeleon: `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt`

Override with environment variables:
- `LATENTVIS_CHEMPROP_PT` – path to chemprop .pt
- `LATENTVIS_CHEMELEON_PT` – path to chemeleon .pt

**Frontend usage:** Send the current (dataset, epoch, layer) plus the user-selected k and linkage (e.g. from dropdowns). Use the returned `assignments` or `labels` + `real_ids` to update the scatter plot colors and any cluster-based panels (e.g. top-k lists, Sankey) for that view.

---

## Running the backend

```bash
cd latentvis/backend
python flask_backend.py
```

Runs on `http://127.0.0.1:5001` by default. Ensure Flask and project dependencies (e.g. `embedding_loader`, `clustering`) are available; the app adds `backend` to `sys.path` for imports.
