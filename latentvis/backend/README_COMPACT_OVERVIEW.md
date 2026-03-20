# Compact Overview Format

Generated files live under:

- `latentvis/backend/data_compact/chemprop/overview.json`
- `latentvis/backend/data_compact/chemeleon/overview.json`
- `latentvis/backend/data_compact/dimenet/overview.json`
- `latentvis/backend/data_compact/manifest.json`

Build command:

```bash
python3 latentvis/backend/scripts/build_compact_overview_data.py
# Default DR coordinate rounding keeps 2 decimal places.
# Override when needed:
python3 latentvis/backend/scripts/build_compact_overview_data.py --coord-decimals 2
```

## Design goals

- One compact file per dataset
- Sequential dense molecule IDs: `0..N-1`
- Split membership stored once at the top level
- Node payloads store one sparse DR payload across all molecules, not one payload per split
- Sparse cluster labels and sparse representative maps
- Sparse Sankey connectivity, stored as adjacency lists instead of dense matrices
- Rounded DR coordinates to reduce file size
- Redundant node-name strings removed from node payloads
- Optional molecule labels only when they add information beyond the dense IDs

## Top-level fields

- `api_version`
- `dataset`
- `source_variant`
- `dr_coord_decimals`
- `source_paths`
- `molecule_ids`
- `molecule_labels` (optional; omitted when labels are just `"0"`, `"1"`, ...)
- `splits`
- `epochs`
- `layers`
- `cells`
- `sankey`

## JSON shape

```json
{
  "api_version": "compact_overview_v2",
  "dataset": "chemprop",
  "source_variant": "knn",
  "dr_coord_decimals": 2,
  "source_paths": {
    "train": {
      "scatter_dir": "string-or-null",
      "cluster_csv": "string-or-null",
      "representatives_json": "string-or-null",
      "sankey_json": "string-or-null"
    },
    "val": {
      "scatter_dir": "string-or-null",
      "cluster_csv": "string-or-null",
      "representatives_json": "string-or-null",
      "sankey_json": "string-or-null"
    },
    "test": {
      "scatter_dir": "string-or-null",
      "cluster_csv": "string-or-null",
      "representatives_json": "string-or-null",
      "sankey_json": "string-or-null"
    }
  },
  "molecule_ids": [0, 1, 2],
  "molecule_labels": ["row_10", "row_11", "row_12"],
  "splits": {
    "train": [0, 2],
    "val": [],
    "test": [1]
  },
  "epochs": ["0", "1"],
  "layers": ["fingerprint", "gnn_layer_1"],
  "cells": [
    {
      "node_key": 0,
      "epoch": "0",
      "layer": "fingerprint",
      "epoch_idx": 0,
      "layer_idx": 0,
      "global_node_index": 0,
      "cluster_labels": {
        "0": 0,
        "1": 1,
        "2": 0
      },
      "cluster_representatives": {
        "train": {
          "0": 0
        },
        "val": {},
        "test": {
          "1": 1
        }
      },
      "dr": {
        "default": {
          "0": [0.1, 0.2],
          "1": [0.5, 0.6],
          "2": [0.3, 0.4]
        },
        "pca": {
          "0": [0.11, 0.21],
          "1": [0.51, 0.61]
        }
      }
    }
  ],
  "sankey": {
    "connectivity": {
      "0": [[1, 3], [2, 3]],
      "1": [[3, 3]]
    },
    "split_connectivity": {
      "train": {
        "0": [[1, 2], [2, 2]]
      },
      "val": {},
      "test": {
        "0": [[1, 1], [2, 1]]
      }
    },
    "node_lookup": {
      "0": {
        "fingerprint": 0,
        "gnn_layer_1": 1
      },
      "1": {
        "fingerprint": 2,
        "gnn_layer_1": 3
      }
    }
  }
}
```

## Field types

Top-level object:

- `api_version`: `string`
- `dataset`: `string`
- `source_variant`: `string`
- `dr_coord_decimals`: `int`
- `source_paths`: `{ train: SourcePaths, val: SourcePaths, test: SourcePaths }`
- `molecule_ids`: `int[]`
- `molecule_labels`: `string[]` when present
- `splits`: `{ train: int[], val: int[], test: int[] }`
- `epochs`: `string[]`
- `layers`: `string[]`
- `cells`: `Cell[]`
- `sankey`: `SankeyPayload`

`SourcePaths`:

- `scatter_dir`: `string | null`
- `cluster_csv`: `string | null`
- `representatives_json`: `string | null`
- `sankey_json`: `string | null`

`Cell`:

- `node_key`: `int`
- `epoch`: `string`
- `layer`: `string`
- `epoch_idx`: `int`
- `layer_idx`: `int`
- `global_node_index`: `int`
- `cluster_labels`: `{ [molecule_id: string]: int }`
- `cluster_representatives`: `{ train: SparseRepresentativeMap, val: SparseRepresentativeMap, test: SparseRepresentativeMap }`
- `dr`: `{ [projection_name: string]: SparseDrMap }`

`SparseRepresentativeMap`:

- `{ [cluster_id: string]: molecule_id }`

`SparseDrMap`:

- `{ [molecule_id: string]: [x, y] }`
- Values are rounded to `dr_coord_decimals` decimal places at write time.

`SankeyPayload`:

- `connectivity`: `SparseConnectivity`
- `split_connectivity`: `{ train: SparseConnectivity, val: SparseConnectivity, test: SparseConnectivity }`
- `node_lookup`: `{ [epoch: string]: { [layer: string]: int } }`

`SparseConnectivity`:

- `{ [source_node_key: string]: [[target_node_key, count], ...] }`

## Notes

- `splits` is the only place that stores train/val/test membership.
- `cells[].dr` stores coordinates for all available molecules at that node, regardless of split.
- `cells[].cluster_labels` also stores one sparse mapping across all splits. The split for each molecule ID is recovered from `splits`.
- `cluster_representatives` stays split-aware because cluster ID spaces are split-local.
- `molecule_labels` is omitted when it would only repeat the dense integer IDs as strings.
- `sankey.connectivity` and `sankey.split_connectivity` are sparse adjacency lists; absent keys imply zero flow.
