#!/bin/bash
# Update frontend data from *existing* backend data (no .pt embeddings required).
# Use this when you already have scatter CSVs and SMILES; it will:
#   1. Generate cluster_representatives.json + cluster_sorted_members.json from scatter CSVs -> frontend topk
#   2. Regenerate MCS substructures from topk + SMILES -> frontend substructures
#
# For full pipeline (multiple scatter projections, etc.) use run_all_and_copy_to_frontend.sh with --input <embedding.pt>.
#
# Usage: ./update_frontend_data_from_existing.sh
# Optional env: SCATTER_TENCONF, SCATTER_NOAUG, SMILES_CSV, MODEL_NAME_1, MODEL_NAME_2, SMILES_SMILES_COLUMN, CASE_NAME
#   MODEL_NAME_1 / MODEL_NAME_2 = labels for the two scatter CSVs when they lack "model_type" (default: dimenet10conf, dimenet_noaug)
#   SMILES_SMILES_COLUMN = column name in SMILES CSV (default: smiles; use product_smiles for DimeNet CSV if needed)
#   CASE_NAME = case study subfolder under frontend/public/data (default: dimenet)

set -e
# Timestamped log (so you can see progress and whether the process is still running)
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# frontend public data is next to backend: latentvis/frontend/public/data
FRONTEND_DATA="$(cd "$BACKEND_DIR/../frontend/public/data" && pwd)"
cd "$BACKEND_DIR"

# So Python prints immediately (no buffering) to the log file
export PYTHONUNBUFFERED=1

CASE_NAME="${CASE_NAME:-dimenet}"
SCATTER_TENCONF="${SCATTER_TENCONF:-$BACKEND_DIR/data/scatter_plots/scatter_by_layer.csv}"
SCATTER_NOAUG="${SCATTER_NOAUG:-$BACKEND_DIR/data/scatter_plots/scatter_noaug_by_layer.csv}"
SMILES_CSV="${SMILES_CSV:-$BACKEND_DIR/data/smiles/real_id_smiles.csv}"
MODEL_NAME_1="${MODEL_NAME_1:-dimenet10conf}"
MODEL_NAME_2="${MODEL_NAME_2:-dimenet_noaug}"
# Default product_smiles for backward compat with current DimeNet CSV; set SMILES_SMILES_COLUMN=smiles for other case studies
SMILES_SMILES_COLUMN="${SMILES_SMILES_COLUMN:-product_smiles}"
TOPK_DIR="$FRONTEND_DATA/$CASE_NAME/topk"
SUBSTRUCTURES_JSON="$FRONTEND_DATA/$CASE_NAME/substructures/cluster_substructures.json"

log "=== Script started ==="
log "  TOPK_DIR=$TOPK_DIR"
log "  SUBSTRUCTURES_JSON=$SUBSTRUCTURES_JSON"

log "=== Step 1/2: Generate topk + cluster_sorted_members from scatter CSVs ==="
log "  (Usually takes 1-3 min. If this hangs, check disk/network.)"
if [[ ! -f "$SCATTER_TENCONF" || ! -f "$SCATTER_NOAUG" ]]; then
  log "Warning: scatter CSVs not found, skip step 1."
  log "  tenconf: $SCATTER_TENCONF"
  log "  noaug:   $SCATTER_NOAUG"
else
  mkdir -p "$TOPK_DIR"
  if python3 -u "$BACKEND_DIR/scripts/generate_cluster_representatives_from_scatter.py" \
    --tenconf "$SCATTER_TENCONF" \
    --noaug "$SCATTER_NOAUG" \
    --out "$TOPK_DIR/cluster_representatives.json" \
    --model-name-1 "$MODEL_NAME_1" \
    --model-name-2 "$MODEL_NAME_2"; then
    log "Step 1 finished OK. Output: $TOPK_DIR/cluster_representatives.json and cluster_sorted_members.json"
  else
    log "Step 1 FAILED (exit code $?). Check paths and CSV format."
    exit 1
  fi
fi

log ""
log "=== Step 2/2: Regenerate MCS substructures (topk + SMILES) -> frontend ==="
log "  (Can take 5-25 min; you will see progress lines below.)"
if [[ ! -f "$TOPK_DIR/cluster_representatives.json" ]]; then
  log "Warning: cluster_representatives.json not found, skip step 2."
elif [[ ! -f "$SMILES_CSV" ]]; then
  log "Warning: SMILES CSV not found ($SMILES_CSV), skip step 2."
else
  if "$SCRIPT_DIR/update_substructures_to_frontend.sh" \
    "$TOPK_DIR/cluster_representatives.json" \
    "$SMILES_CSV" \
    "$SUBSTRUCTURES_JSON" \
    "$SMILES_SMILES_COLUMN"; then
    log "Step 2 finished OK. Output: $SUBSTRUCTURES_JSON"
  else
    log "Step 2 FAILED (exit code $?). Check topk JSON and SMILES CSV."
    exit 1
  fi
fi

log ""
log "=== Done. Frontend data updated (topk, cluster_sorted_members, substructures). ==="
log ""
log "To verify: ls $TOPK_DIR should show cluster_representatives.json and cluster_sorted_members.json; substructures: $SUBSTRUCTURES_JSON"
