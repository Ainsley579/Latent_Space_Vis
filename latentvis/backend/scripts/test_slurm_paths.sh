#!/bin/bash
# Run the same path checks and cd as the real job. Use before sbatch to verify.
# Usage: ./test_slurm_paths.sh [chemprop|chemeleon]
set -e
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
SCRIPT_DIR="$BACKEND_DIR/scripts"
FRONTEND_DATA="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/frontend/public/data"

echo "=== Testing paths (same as job will use) ==="
if [[ ! -d "$BACKEND_DIR" ]]; then echo "FAIL: BACKEND_DIR missing: $BACKEND_DIR"; exit 1; fi
echo "  BACKEND_DIR OK"
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "FAIL: run_pipeline_from_embeddings.py not in BACKEND_DIR"; exit 1; fi
echo "  run_pipeline_from_embeddings.py OK"
if [[ ! -d "$(dirname "$FRONTEND_DATA")" ]]; then echo "FAIL: Frontend base missing"; exit 1; fi
echo "  FRONTEND_DATA parent OK"

cd "$BACKEND_DIR" || { echo "FAIL: cd BACKEND_DIR"; exit 1; }
echo "  cd BACKEND_DIR OK, PWD=$(pwd)"

if [[ "$1" == "chemprop" ]]; then
  INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt"
  if [[ ! -f "$INPUT" ]]; then echo "FAIL: $INPUT missing"; exit 1; fi
  echo "  chemprop INPUT OK"
fi
if [[ "$1" == "chemeleon" ]]; then
  INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt"
  if [[ ! -f "$INPUT" ]]; then echo "FAIL: $INPUT missing"; exit 1; fi
  echo "  chemeleon INPUT OK"
fi

echo "=== All path checks passed ==="
