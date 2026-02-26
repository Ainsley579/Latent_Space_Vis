# Cluster containment API (user clustering vs backend clustering)

The backend exposes an endpoint that compares a **user-defined clustering** (e.g. from a spreadsheet) with the **backend’s clustering** (e.g. PGMeans) for a given cell (dataset, epoch, layer). It returns a **containment matrix**: for each user cluster, the percentage of its points that fall into each of our clusters.

## Endpoint

**POST** `/cluster-containment`

Base URL is the same as the Flask API (e.g. `http://127.0.0.1:5001`).

## Request body (JSON)

| Field            | Type    | Required | Description |
|------------------|---------|----------|-------------|
| `dataset`        | string  | yes      | `"chemprop"` or `"chemeleon"` |
| `epoch`          | string or number | yes | Epoch id for the cell, e.g. `"0"`, `"26"` |
| `layer`          | string  | yes      | Layer id, e.g. `"fingerprint"`, `"gnn_layer_1"` |
| `clustering`     | string  | no       | Backend clustering method (e.g. `"pgmeans"`, `"gmm"`). Default depends on backend. |
| `data_split`     | string  | no       | `"test"` or `"train"`. Default: `"test"`. |
| `use_current`    | boolean | no       | If `true`, use assignments from the last interactive run (`clustering_metrics_current`). If `false`, use baseline. Default: `false`. |
| `user_clustering`| array   | yes      | List of objects: `{ "real_id": "<id>", "user_cluster_id": <integer> }`. One entry per point; `user_cluster_id` is 0-based. |

- **real_id**: must match the backend’s `real_id` (e.g. from `cluster_assignments_with_real_id.csv` or scatter JSON).
- **user_cluster_id**: the cluster label assigned by the user (0, 1, 2, …). Any non-negative integers are allowed; they are treated as distinct cluster ids.

Example: 3 user clusters, 4 backend clusters → response will include a 3×4 matrix and 12 entries.

## Response (200 OK)

```json
{
  "ok": true,
  "user_cluster_ids": [0, 1, 2],
  "our_cluster_ids": [0, 1, 2, 3],
  "matrix": [
    [0.2, 0.7, 0.1, 0.0],
    [0.0, 0.1, 0.8, 0.1],
    [0.1, 0.2, 0.1, 0.6]
  ],
  "entries": [
    { "user_cluster_id": 0, "our_cluster_id": 0, "count": 20, "percentage_of_user_cluster": 20.0 },
    { "user_cluster_id": 0, "our_cluster_id": 1, "count": 70, "percentage_of_user_cluster": 70.0 },
    ...
  ]
}
```

- **user_cluster_ids**: sorted list of user cluster ids present in the request.
- **our_cluster_ids**: sorted list of backend cluster ids for that cell.
- **matrix**: `matrix[i][j]` = percentage of user cluster `user_cluster_ids[i]` that lies in backend cluster `our_cluster_ids[j]` (0–100).
- **entries**: flat list of `user_cluster_id`, `our_cluster_id`, `count`, and `percentage_of_user_cluster` for each (user, our) pair with non-zero count.

## How to call from the frontend

1. User provides a spreadsheet (or table) with at least: **real_id** and **user_cluster_id** for the selected cell.
2. Frontend sends a POST request with:
   - `dataset`, `epoch`, `layer` for the current cell.
   - `data_split` and optionally `use_current` to match the current view.
   - `user_clustering`: array built from the spreadsheet rows, e.g. `[{ "real_id": "42", "user_cluster_id": 0 }, ...]`.
3. Use the response `matrix` or `entries` to show overlap (e.g. heatmap or table: user clusters × our clusters, cells = percentage).

## Errors

- **400**: Missing required field (`dataset`, `epoch`, `layer`, or `user_clustering`), or assignment file not found for that cell.
- **500**: Server error (e.g. missing data dir).

All content under `Latent_Space_Vis` is in English.
