#!/bin/bash
# Run full pipeline for BOTH dimenet_noaug and dimenet10conf (same embeddings as old results),
# then copy merged output to frontend. Uses layer maps so layer names match (emb, output_block_0, ...).
#
# Usage:
#   ./run_both_dimenet_and_copy_to_frontend.sh [--output-dir ./pipeline_output] [--smiles-csv PATH] [--time 2:00:00]
#
# Embedding paths (fixed):
#   dimenet_noaug:  /global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet_noaug_embeddings_cas_152
#   dimenet10conf:  /global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

EMB_ROOT="/global/cfs/cdirs/m636/nhadler/projects/latent-spaces"
INPUT_NOAUG="${EMB_ROOT}/dimenet_noaug_embeddings_cas_152"
INPUT_10CONF="${EMB_ROOT}/dimenet10conf_embeddings"
LAYER_MAP_NOAUG="${SCRIPT_DIR}/layer_map_dimenet_noaug_cas.json"
LAYER_MAP_10CONF="${SCRIPT_DIR}/layer_map_dimenet10conf_cas.json"
OUTPUT_DIR="./pipeline_output"
SMILES_CSV="${BACKEND_DIR}/data/smiles/real_id_smiles.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --smiles-csv)
      SMILES_CSV="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ ! -d "$INPUT_NOAUG" ]]; then
  echo "Error: dimenet_noaug dir not found: $INPUT_NOAUG"
  exit 1
fi
if [[ ! -d "$INPUT_10CONF" ]]; then
  echo "Error: dimenet10conf dir not found: $INPUT_10CONF"
  exit 1
fi
if [[ ! -f "$LAYER_MAP_NOAUG" ]] || [[ ! -f "$LAYER_MAP_10CONF" ]]; then
  echo "Error: layer map JSONs not found in $SCRIPT_DIR"
  exit 1
fi

SMILES_ARG=""
if [[ -n "$SMILES_CSV" && -f "$SMILES_CSV" ]]; then
  SMILES_ARG="--smiles-csv $SMILES_CSV"
fi

echo "=== Running pipeline for both dimenet_noaug + dimenet10conf (one run, merged Sankey) ==="
python3 run_pipeline_from_embeddings.py \
  --input "$INPUT_NOAUG" \
  --model-name dimenet_noaug \
  --layer-file-map "$LAYER_MAP_NOAUG" \
  --input2 "$INPUT_10CONF" \
  --model-name2 dimenet10conf \
  --layer-file-map2 "$LAYER_MAP_10CONF" \
  --output-dir "$OUTPUT_DIR" \
  $SMILES_ARG \
  --default-reduction mds

echo ""
echo "=== Copying pipeline output to frontend (data/dimenet/) ==="
bash "$SCRIPT_DIR/copy_to_frontend.sh" "$OUTPUT_DIR" dimenet

echo ""
echo "=== Done. Frontend data/dimenet/ has both dimenet_noaug and dimenet10conf (sankey, scatterplot, topk, substructures, etc.). ==="
