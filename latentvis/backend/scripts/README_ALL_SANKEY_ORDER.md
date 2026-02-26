# all_sankey.json layer order (model order)

The frontend reads **layer order** from `all_sankey.json`: the key order of `data.layer` is the canonical layer order and must match the **model's layer order**.

## Existing files (already updated)

All `all_sankey.json` under `frontend/public/data` for **chemprop**, **chemeleon**, and **dimenet** (where applicable) have been reordered so that `"layer"` keys follow model order:

- **Chemprop:** `fingerprint`, `gnn_layer_1`, `gnn_layer_2`, `gnn_layer_3`, `ffn_layer_1`, `ffn_layer_2`
- **Chemeleon:** `fingerprint`, `gnn_layer_1` … `gnn_layer_6`, `ffn_layer_1`, `ffn_layer_2`
- **Dimenet** (clustering dirs with same layer set): same as chemprop for the shared layers.

Backend pipeline test/validation dirs (`pipeline_output_chemprop`, `pipeline_validate_chemeleon`, `pipeline_validate_test/chemprop_short`) have also been reordered.

## Future runs: new all_sankey will have correct order

1. **`run_pipeline_from_embeddings.py`**
   - `layer_keys` comes from `load_embedding_input()` (embedding_loader).
   - For ESOL .pt (chemprop/chemeleon), the loader returns `layer_keys` in **model order** (order of keys in `split["embeddings"]`); it does not sort alphabetically.
   - `all_sankey["layer"]` is built by iterating `for lk in layer_keys`, so the written JSON has keys in **layer_keys order = model order**.

2. **`materialize_sankey_compat_files.py`**
   - `layers` is inferred from the CSV header (`infer_layers(fieldnames)` = column order of `*_cluster` columns).
   - When the CSV was produced by the pipeline, columns were added in `layer_keys` order, so **layers = model order**.
   - `build_all_sankey(..., layers, ...)` iterates over `layers`, so the written `all_sankey.json` has `"layer"` keys in **model order**.
   - If you run materialize on a CSV from another source, pass `--layer-order "fingerprint,gnn_layer_1,..."` to enforce model order.

## Reordering old files

If you have an old `all_sankey.json` with wrong (e.g. alphabetical) layer order, run:

```bash
python3 scripts/reorder_all_sankey_keys.py <path/to/all_sankey.json> \
  --layer-order "fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,ffn_layer_1,ffn_layer_2" \
  --reorder-epochs
```

Or use `--layer-order-from-csv` to take the order from the same case's `clustering_metrics/cluster_assignments_with_real_id.csv`.
