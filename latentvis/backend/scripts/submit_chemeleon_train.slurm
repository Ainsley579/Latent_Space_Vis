#!/bin/bash
# SBATCH job for chemeleon train-split pipeline: single CPU, exclusive node, 4hr.
#
# Usage: sbatch scripts/submit_chemeleon_train.slurm
#
# Writes to frontend/public/data/chemeleon/train (separate from test baseline).

#SBATCH --job-name=chemeleon_train_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --constraint=cpu
#SBATCH --exclusive
#SBATCH --qos=regular
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemeleon_train_pipeline_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemeleon_train_pipeline_%j.err

set -e
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
SCRIPT_DIR="$BACKEND_DIR/scripts"
INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt"
FRONTEND_DATA="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/frontend/public/data"
OUTPUT_DIR="$FRONTEND_DATA/chemeleon/train"

if [[ ! -d "$BACKEND_DIR" ]]; then echo "ERROR: BACKEND_DIR missing: $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "ERROR: run_pipeline_from_embeddings.py not found in $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$INPUT" ]]; then echo "ERROR: Input file missing: $INPUT"; exit 1; fi
if [[ ! -d "$(dirname "$FRONTEND_DATA")" ]]; then echo "ERROR: Frontend base dir missing: $(dirname "$FRONTEND_DATA")"; exit 1; fi

mkdir -p "$OUTPUT_DIR"

cd "$BACKEND_DIR" || { echo "ERROR: cd $BACKEND_DIR failed"; exit 1; }

module load conda
conda activate latentvis

export PYTHONUNBUFFERED=1

echo "Job started at $(date -Iseconds)"
echo "PWD=$(pwd)"
echo "Input: $INPUT"
echo "Output (train split): $OUTPUT_DIR"
echo ""

python3 run_pipeline_from_embeddings.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --k-max 8 \
  --default-reduction pca \
  --clustering-method all \
  --data-split train

EXIT=$?
echo ""
echo "Pipeline exit code: $EXIT"

COMPLETE=0
if [[ -f "$OUTPUT_DIR/sankey/all_sankey.json" ]] && \
   [[ -d "$OUTPUT_DIR/scatterplot" ]] && \
   [[ -f "$OUTPUT_DIR/topk/cluster_representatives.json" ]]; then
  COMPLETE=1
  echo "Output complete. All required files written for chemeleon train split."
fi

echo "Job finished at $(date -Iseconds)"

if [[ $EXIT -ne 0 ]] || [[ $COMPLETE -eq 0 ]]; then
  echo ""
  echo "Pipeline incomplete or failed for chemeleon train split."
  exit 1
fi

cat > "$OUTPUT_DIR/README.md" << 'README_EOF'
# Latent Space Visualization - Chemeleon Case (train split)

This directory contains pipeline output for the **chemeleon train split**:
embeddings from `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt`
restricted to the **train** split.

The layout mirrors `frontend/public/data/chemeleon` (test split) but lives under
`frontend/public/data/chemeleon/train` so that:

- Train and test scatterplots are kept separate.
- Interactive `/generate-range` updates can write to `scatterplot_current/`
  without overwriting baseline `scatterplot/` JSONs.

All JSON schemas match the test case (scatter, topk, substructures, Sankey, clustering metrics).
README_EOF
echo "Wrote README.md to $OUTPUT_DIR"

if [[ -f "$OUTPUT_DIR/real_id_to_smiles.json" ]]; then
  echo "Generating molecule images for chemeleon train split..."
  python3 "$BACKEND_DIR/scripts/generate_molecule_images_from_smiles.py" --case-dir "$OUTPUT_DIR" || true
fi

exit 0

