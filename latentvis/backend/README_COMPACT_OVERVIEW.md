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
- A top-level `split_names` array defines split index semantics
- `splits` is an array aligned with `split_names`
- Cell payloads store one sparse DR payload across all molecules, not one payload per split
- Sparse cluster labels and sparse representative maps
- A global cluster-node array, separate from epoch/layer cells
- Sparse cluster-to-cluster connectivity, stored as adjacency lists instead of dense matrices
- `node_lookup` provides direct reverse indexing from `(split_idx, epoch_idx, layer_idx, cluster_id)` to `node_id`
- Connectivity and lookup structures use array indexing instead of JSON object keys for integer-indexed dimensions
- Rounded DR coordinates to reduce file size
- Redundant node-name strings removed from node payloads
- Optional molecule labels only when they add information beyond the dense IDs

## Top-level fields

- `api_version`
- `dataset`
- `source_variant`
- `molecule_ids`
- `molecule_labels` (optional; omitted when labels are just `"0"`, `"1"`, ...)
- `split_names`
- `splits`
- `epochs`
- `layers`
- `cells`
- `nodes`
- `connectivity`
- `split_connectivity`
- `node_lookup`

## JSON shape

```json
{
  "api_version": "compact_overview_v6",
  "dataset": "chemprop",
  "source_variant": "knn",
  "molecule_ids": [0, 1, 2],
  "molecule_labels": ["row_10", "row_11", "row_12"],
  "split_names": ["train", "test", "val"],
  "splits": [
    [0, 2],
    [1],
    []
  ],
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
        }
      }
    }
  ],
  "nodes": [
    {
      "node_id": 0,
      "split": 0,
      "epoch": "0",
      "layer": "fingerprint",
      "epoch_idx": 0,
      "layer_idx": 0,
      "cell_index": 0,
      "cluster_id": 0,
      "member_count": 2,
      "representative_molecule_id": 0
    },
    {
      "node_id": 1,
      "split": 1,
      "epoch": "0",
      "layer": "fingerprint",
      "epoch_idx": 0,
      "layer_idx": 0,
      "cell_index": 0,
      "cluster_id": 1,
      "member_count": 1,
      "representative_molecule_id": 1
    }
  ],
  "connectivity": [
    [[2, 2]],
    [[3, 1]],
    [],
    []
  ],
  "split_connectivity": [
    [
      [[2, 2]],
      [],
      [],
      []
    ],
    [
      [],
      [[3, 1]],
      [],
      []
    ],
    [
      [],
      [],
      [],
      []
    ]
  ],
  "node_lookup": [
    [
      [
        [[0, 0]],
        [[0, 2]]
      ],
      [
        [],
        []
      ]
    ],
    [
      [
        [[1, 1]],
        [[1, 3]]
      ],
      [
        [],
        []
      ]
    ],
    [
      [
        [],
        []
      ],
      [
        [],
        []
      ]
    ]
  ]
}
```

## Field types

Top-level object:

- `api_version`: `string`
- `dataset`: `string`
- `source_variant`: `string`
- `molecule_ids`: `int[]`
- `molecule_labels`: `string[]` when present
- `split_names`: `string[]`
- `splits`: `int[][]`
- `epochs`: `string[]`
- `layers`: `string[]`
- `cells`: `Cell[]`
- `nodes`: `ClusterNode[]`
- `connectivity`: `EdgeListByNode`
- `split_connectivity`: `EdgeListByNode[]`
- `node_lookup`: `ClusterLookupBySplit`

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

`ClusterNode`:

- `node_id`: `int`
- `split`: `int`
- `epoch`: `string`
- `layer`: `string`
- `epoch_idx`: `int`
- `layer_idx`: `int`
- `cell_index`: `int`
- `cluster_id`: `int`
- `member_count`: `int`
- `representative_molecule_id`: `int | null`

`SparseRepresentativeMap`:

- `{ [cluster_id: string]: molecule_id }`

`SparseDrMap`:

- `{ [molecule_id: string]: [x, y] }`
- Coordinate values are rounded during generation.

`EdgeListByNode`:

- `[[[target_node_id, count], ...], ...]`
- The outer array is indexed by `source_node_id`.

`ClusterLookupBySplit`:

- `node_lookup[split_idx][epoch_idx][layer_idx] = [[cluster_id, node_id], ...]`

## Notes

- `split_names` defines the mapping from split index to split name.
- `splits[i]`, `split_connectivity[i]`, and `node_lookup[i]` all correspond to `split_names[i]`.
- `epochs` and `layers` define the index order used by `node_lookup`.
- `cells[].dr` stores coordinates for all available molecules at that node, regardless of split.
- `cells[].cluster_labels` also stores one sparse mapping across all splits.
- `cluster_representatives` stays split-aware because cluster ID spaces are split-local.
- `nodes` are split-aware for the same reason. A cluster node is uniquely identified by `(split_idx, epoch, layer, cluster_id)`.
- `nodes[*].cell_index` points back to the corresponding entry in `cells`.
- `node_lookup` is only a reverse index for convenience. It does not add new information beyond `nodes`.
- `connectivity[source_node_id]` gives all outgoing edges for that node.
