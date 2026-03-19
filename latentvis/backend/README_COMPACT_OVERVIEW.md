# Compact Overview Format

Generated files live under:

- `latentvis/backend/data_compact/chemprop/overview.json`
- `latentvis/backend/data_compact/chemeleon/overview.json`
- `latentvis/backend/data_compact/dimenet/overview.json`
- `latentvis/backend/data_compact/manifest.json`

Build command:

```bash
python3 latentvis/backend/scripts/build_compact_overview_data.py
```

## Design goals

- Single overview payload per dataset
- Strict sequential integer molecule IDs: `0..N-1`
- Molecule names separated from core flow data
- Explicit split arrays: `train`, `val`, `test`
- Flattened Sankey node indexing with:

```text
layer_idx + (epoch_idx * total_layers_per_epoch)
```

- Per-cell bookkeeping for `(epoch, layer)`
- Per-split DR coordinates, cluster labels, and representative IDs

## Top-level fields

- `molecule_ids`: sequential integer IDs
- `molecule_names`: array indexed by molecule ID
- `splits`: object with `train`, `val`, `test` integer ID arrays
- `epochs`: ordered epoch names
- `layers`: ordered layer names
- `cells`: flattened `(epoch, layer)` cell records ordered by global node index
- `sankey.names`: node names aligned with matrix indices
- `sankey.connectivity_matrix`: aggregated flow matrix
- `sankey.split_connectivity_matrices`: per-split flow matrices
- `sankey.bookkeeping`: epoch/layer lookup metadata

## JSON shape

The compact overview file is a single JSON object with this structure:

```json
{
  "api_version": "compact_overview_v1",
  "dataset": "chemprop",
  "source_variant": "knn",
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
  "molecule_names": ["molecule_a", "molecule_b", "molecule_c"],
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
      "name": "epoch_0__layer_fingerprint",
      "epoch": "0",
      "layer": "fingerprint",
      "epoch_idx": 0,
      "layer_idx": 0,
      "global_node_index": 0,
      "split_data": {
        "train": {
          "cluster_labels": [0, null, 1],
          "cluster_representative_ids": [0, 2],
          "dr": {
            "default": {
              "x": [0.1, null, 0.3],
              "y": [0.2, null, 0.4]
            },
            "pca": {
              "x": [0.1, null, 0.3],
              "y": [0.2, null, 0.4]
            }
          }
        },
        "val": {
          "cluster_labels": [null, null, null],
          "cluster_representative_ids": [],
          "dr": {}
        },
        "test": {
          "cluster_labels": [null, 0, null],
          "cluster_representative_ids": [1],
          "dr": {
            "default": {
              "x": [null, 0.5, null],
              "y": [null, 0.6, null]
            }
          }
        }
      }
    }
  ],
  "sankey": {
    "node_keys": [0, 1, 2, 3],
    "names": [
      "epoch_0__layer_fingerprint",
      "epoch_0__layer_gnn_layer_1",
      "epoch_1__layer_fingerprint",
      "epoch_1__layer_gnn_layer_1"
    ],
    "connectivity_matrix": [
      [0, 3, 3, 0],
      [0, 0, 0, 3],
      [0, 0, 0, 3],
      [0, 0, 0, 0]
    ],
    "split_connectivity_matrices": {
      "train": [[0, 2, 2, 0], [0, 0, 0, 2], [0, 0, 0, 2], [0, 0, 0, 0]],
      "val": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      "test": [[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    },
    "bookkeeping": {
      "epochs": ["0", "1"],
      "layers": ["fingerprint", "gnn_layer_1"],
      "total_layers_per_epoch": 2,
      "global_node_index_formula": "layer_idx + (epoch_idx * total_layers_per_epoch)",
      "nodes": [
        {
          "node_key": 0,
          "name": "epoch_0__layer_fingerprint",
          "epoch": "0",
          "layer": "fingerprint",
          "epoch_idx": 0,
          "layer_idx": 0,
          "global_node_index": 0
        }
      ],
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
}
```

## Field types

Top-level object:

- `api_version`: `string`
- `dataset`: `string`
- `source_variant`: `string`
- `source_paths`: `{ train: SourcePaths, val: SourcePaths, test: SourcePaths }`
- `molecule_ids`: `int[]`
- `molecule_names`: `string[]`
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
- `name`: `string`
- `epoch`: `string`
- `layer`: `string`
- `epoch_idx`: `int`
- `layer_idx`: `int`
- `global_node_index`: `int`
- `split_data`: `{ train: SplitCellData, val: SplitCellData, test: SplitCellData }`

`SplitCellData`:

- `cluster_labels`: `(int | null)[]`
- `cluster_representative_ids`: `(int | null)[]`
- `dr`: `{ [projection_name: string]: DrCoordinates }`

`DrCoordinates`:

- `x`: `(number | null)[]`
- `y`: `(number | null)[]`

`SankeyPayload`:

- `node_keys`: `int[]`
- `names`: `string[]`
- `connectivity_matrix`: `int[][]`
- `split_connectivity_matrices`: `{ train: int[][], val: int[][], test: int[][] }`
- `bookkeeping`: `SankeyBookkeeping`

`SankeyBookkeeping`:

- `epochs`: `string[]`
- `layers`: `string[]`
- `total_layers_per_epoch`: `int`
- `global_node_index_formula`: `string`
- `node_lookup`: `{ [epoch: string]: { [layer: string]: int } }`
- `nodes`: `Cell[]`

## Cell payload

Each item in `cells` contains:

- `node_key`
- `epoch`
- `layer`
- `epoch_idx`
- `layer_idx`
- `global_node_index`
- `split_data`

Each `split_data[split]` contains:

- `cluster_labels`: one array indexed by integer molecule ID
- `cluster_representative_ids`: array indexed by cluster ID
- `dr`: projection name to `{x, y}` arrays

## Indexing rules

- `molecule_ids[i] == i` for all molecules.
- `molecule_names[molecule_id]` returns the display name for that molecule.
- `splits.train`, `splits.val`, and `splits.test` contain molecule IDs only.
- `cluster_labels[molecule_id]` returns the cluster ID for that molecule in one specific `(epoch, layer, split)` cell.
- `cluster_representative_ids[cluster_id]` returns the representative molecule ID for that cluster, or `null` if no representative is available.
- `dr[projection].x[molecule_id]` and `dr[projection].y[molecule_id]` return the 2D coordinates for that molecule in one specific `(epoch, layer, split)` cell.

## Sankey indexing rules

- `sankey.node_keys[node_index] == node_index`.
- `sankey.names[node_index]` is the canonical node name for one `(epoch, layer)` cell.
- `sankey.connectivity_matrix[source_index][target_index]` is the aggregated sample count flowing from the source node to the target node.
- `sankey.split_connectivity_matrices[split][source_index][target_index]` is the same count restricted to one split.
- `cells[global_node_index].node_key`, `sankey.node_keys[global_node_index]`, and `global_node_index` all refer to the same integer node key.
- `cells[global_node_index]` and `sankey.names[global_node_index]` refer to the same `(epoch, layer)` node.
- Global node indices must follow:

```text
global_node_index = layer_idx + (epoch_idx * total_layers_per_epoch)
```

- `sankey.bookkeeping.node_lookup[epoch][layer]` resolves an `(epoch, layer)` pair back to its global node index.
- `sankey.bookkeeping.nodes` repeats the flattened node records so node metadata is also available in array form with integer node keys.

## Null and empty-value rules

- `cluster_labels` always has length `len(molecule_ids)`.
- If a molecule does not belong to the current split or no cluster label is available for that cell, its `cluster_labels[molecule_id]` value is `null`.
- `cluster_representative_ids` may be an empty array when no cluster representatives are available for that split and cell.
- A `cluster_representative_ids[cluster_id]` value may be `null` when the cluster exists but no representative molecule could be resolved.
- `dr` may be an empty object when no projection files are available for that split and cell.
- Each `x` and `y` array always has length `len(molecule_ids)`.
- If a molecule has no coordinate for the current split and cell, the corresponding `x[molecule_id]` and `y[molecule_id]` values are `null`.

## Naming conventions

- Molecule names are not embedded into the core cell arrays. They live only in `molecule_names`.
- Cell names use the format:

```text
epoch_<epoch_value>__layer_<layer_value>
```

- The same naming convention is used in `cells[*].name` and `sankey.names[*]`.

## Notes

- Current compact exports do not modify any existing pipeline output.
- If a split has no generated artifacts in the current case data, its arrays remain empty or `null`-filled.
