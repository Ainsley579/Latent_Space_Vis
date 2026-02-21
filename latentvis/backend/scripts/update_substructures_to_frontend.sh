#!/bin/bash
# Fast path: regenerate MCS substructures with current code and write directly to frontend.
# Uses existing topk + SMILES CSV; no full pipeline or copy step.
# Run from project root (Latent_Space_Vis) or from this script's directory.
#
# Run in background:  nohup ./latentvis/backend/scripts/update_substructures_to_frontend.sh & 
# Or:                 ./latentvis/backend/scripts/update_substructures_to_frontend.sh &

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TOPK_JSON="${1:-$BACKEND_DIR/../frontend/public/data/topk/cluster_representatives.json}"
SMILES_CSV="${2:-$BACKEND_DIR/data/smiles/real_id_smiles.csv}"
OUT_JSON="${3:-$BACKEND_DIR/../frontend/public/data/substructures/cluster_substructures.json}"
# Optional: SMILES column name in CSV (default "smiles" for generic case studies; use "product_smiles" if your CSV uses that)
SMILES_COLUMN="${4:-smiles}"

if [[ ! -f "$TOPK_JSON" ]]; then
  echo "Error: topk JSON not found: $TOPK_JSON"
  exit 1
fi
if [[ ! -f "$SMILES_CSV" ]]; then
  echo "Error: SMILES CSV not found: $SMILES_CSV"
  exit 1
fi

echo "=== Regenerating substructures (MCS) -> frontend ==="
echo "  topk:    $TOPK_JSON"
echo "  smiles:  $SMILES_CSV"
echo "  out:     $OUT_JSON"

python3 -u "$BACKEND_DIR/scripts/generate_cluster_substructures.py" \
  --topk-json "$TOPK_JSON" \
  --smiles-csv "$SMILES_CSV" \
  --smiles-column "$SMILES_COLUMN" \
  --out "$OUT_JSON"

echo "=== Done. Frontend substructures updated. ==="
