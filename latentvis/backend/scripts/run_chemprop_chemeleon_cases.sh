#!/bin/bash
# Run pipeline on chemprop and chemeleon .pt files, then copy to frontend public data.
# Usage: ./run_chemprop_chemeleon_cases.sh [output_base]
# For long runs use batch: sbatch -t 60 -c 4 run_chemprop_chemeleon_cases.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

CHEMPROP_PT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt"
CHEMELEON_PT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt"
OUT_BASE="${1:-./pipeline_output}"
FRONTEND_DATA="${BACKEND_DIR}/../frontend/public/data"

echo "=== Latent_Space_Vis pipeline: chemprop and chemeleon ==="
export PYTHONUNBUFFERED=1

if [[ -f "$CHEMPROP_PT" ]]; then
  echo "--- Running pipeline on chemprop ---"
  python3 run_pipeline_from_embeddings.py --input "$CHEMPROP_PT" --output-dir "${OUT_BASE}_chemprop" --default-reduction pca --k-max 10
  echo "Done: ${OUT_BASE}_chemprop"
else
  echo "Warning: chemprop not found: $CHEMPROP_PT"
fi

if [[ -f "$CHEMELEON_PT" ]]; then
  echo "--- Running pipeline on chemeleon ---"
  python3 run_pipeline_from_embeddings.py --input "$CHEMELEON_PT" --output-dir "${OUT_BASE}_chemeleon" --default-reduction pca --k-max 10
  echo "Done: ${OUT_BASE}_chemeleon"
else
  echo "Warning: chemeleon not found: $CHEMELEON_PT"
fi

echo "--- Copying to frontend data root ---"
mkdir -p "$FRONTEND_DATA"
for case in chemprop chemeleon; do
  SRC="${OUT_BASE}_${case}"
  if [[ ! -d "$SRC" ]]; then continue; fi
  for sub in sankey scatterplot bic_aic topk substructures clustering_metrics; do
    if [[ -d "$SRC/$sub" ]]; then
      rm -rf "$FRONTEND_DATA/$sub"
      cp -r "$SRC/$sub" "$FRONTEND_DATA/$sub"
      echo "  Merged $sub from $case"
    fi
  done
done
echo "=== Done. Frontend data: $FRONTEND_DATA ==="
