# Cluster Substructures (MCS)

This folder contains the common substructure results for each cluster’s K‑nearest
and K‑farthest samples. The output is generated with RDKit’s MCS algorithm.

## Output

Default output location used by the script:

`latentvis/frontend/public/data/substructures/cluster_substructures.json`

Rendered image output location:

`latentvis/frontend/public/data/substructures/images/`

Each entry represents one cluster and one group (`knn` or `kfn`):

```json
{
  "layer_name": "emb",
  "model_type": "dimenet10conf",
  "cluster_id": 0,
  "group": "knn",
  "k": 5,
  "real_ids": [5, 12, 23, 45, 67],
  "valid_molecules": 5,
  "missing_real_ids": [],
  "invalid_smiles_real_ids": [],
  "mcs_smarts": "C1=CC=CC=C1",
  "mcs_smiles": "c1ccccc1"
}
```

## How To Generate

You need:
1. `cluster_representatives.json` (top‑k closest/farthest real_id lists)
2. A CSV file mapping `real_id` → `SMILES`

Run:

```bash
python latentvis/backend/scripts/generate_cluster_substructures.py \
  --topk-json latentvis/frontend/public/data/topk/cluster_representatives.json \
  --smiles-csv /path/to/real_id_smiles.csv \
  --id-column real_id \
  --smiles-column product_smiles \
  --out latentvis/frontend/public/data/substructures/cluster_substructures.json
```

## Notes

- RDKit must be installed.
- If fewer than 2 valid molecules exist for a group, `mcs_smarts` and
  `mcs_smiles` will be `null`.
- `missing_real_ids` lists ids that were not found in the SMILES CSV.
- `invalid_smiles_real_ids` lists ids whose SMILES could not be parsed by RDKit.

## Render Images

```bash
python latentvis/backend/scripts/generate_cluster_substructure_images.py \
  --substructures-json latentvis/frontend/public/data/substructures/cluster_substructures.json \
  --out-dir latentvis/frontend/public/data/substructures/images \
  --size 240
```
