# all_sankey.json layer order (model order)

The frontend reads **layer order** from `all_sankey.json`: the key order of `data.layer` is the canonical layer order and must match the **model's layer order**.

**When you change all_sankey (or layer order), all of the following must stay in sync:**

| Output | Script / code | How it stays in model order |
|--------|----------------|-----------------------------|
| `all_sankey.json` | `run_pipeline_from_embeddings.py`, `materialize_sankey_compat_files.py` | Pipeline uses `layer_keys` from loader; materialize uses CSV column order or `--layer-order`. |
| `layer_transition_*.json` | `compute_layer_transition_stats.py` | Consecutive layer pairs (cell-to-cell % change) must match all_sankey order. Use `--layer-order` (and `--layer-order-chemeleon`) when running on existing data. |
| `sankey_samples.json`, `*_sankey_samples.json` | Pipeline, materialize | Built from same `layer_keys` / `layers` as all_sankey. |
| `vertical_model_samples_*.json` | Materialize, Flask compat | Layers from all_sankey key order or materialize `layers`. |

Re-run **reorder_all_sankey_keys.py** on existing all_sankey, then **compute_layer_transition_stats.py** with the same `--layer-order`, and (if needed) **materialize_sankey_compat_files.py** with `--layer-order`.

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

## Layer transition stats (must match all_sankey order)

Cell-to-cell change percentages (`layer_transition_by_epoch.json`, `layer_transition_by_layer.json`) are computed for **consecutive** layers in the order you specify. That order must match `all_sankey.json` so the frontend’s “from layer A to layer B” matches the stats.

After reordering all_sankey (or when using old CSVs with wrong column order), run:

```bash
# Chemprop (6 layers) and chemeleon (9 layers)
python3 scripts/compute_layer_transition_stats.py \
  --layer-order "fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,ffn_layer_1,ffn_layer_2" \
  --layer-order-chemeleon "fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,gnn_layer_4,gnn_layer_5,gnn_layer_6,ffn_layer_1,ffn_layer_2"
```

When the pipeline has just run, the CSV is already in model order, so you can run the script without `--layer-order`.
