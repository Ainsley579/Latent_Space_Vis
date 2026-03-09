#!/bin/bash
# SBATCH job for chemprop train-split pipeline: single CPU, exclusive node, 2hr.
#
# Usage: sbatch scripts/submit_chemprop_train.slurm
#
# Writes to frontend/public/data/chemprop/train (separate from test baseline).

#SBATCH --job-name=chemprop_train_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --constraint=cpu
#SBATCH --exclusive
#SBATCH --qos=regular
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemprop_train_pipeline_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemprop_train_pipeline_%j.err
#SBATCH --signal=B:USR1@300

set -e
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
SCRIPT_DIR="$BACKEND_DIR/scripts"
INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt"
FRONTEND_DATA="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/frontend/public/data"
OUTPUT_DIR="$FRONTEND_DATA/chemprop/train"

# Fail fast: verify all paths exist before running (catches wrong cwd/path on compute node)
if [[ ! -d "$BACKEND_DIR" ]]; then echo "ERROR: BACKEND_DIR missing: $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "ERROR: run_pipeline_from_embeddings.py not found in $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$INPUT" ]]; then echo "ERROR: Input file missing: $INPUT"; exit 1; fi
if [[ ! -d "$(dirname "$FRONTEND_DATA")" ]]; then echo "ERROR: Frontend base dir missing: $(dirname "$FRONTEND_DATA")"; exit 1; fi

mkdir -p "$OUTPUT_DIR"

cd "$BACKEND_DIR" || { echo "ERROR: cd $BACKEND_DIR failed"; exit 1; }

resubmit_self() {
  echo "Resubmitting job script: $SCRIPT_DIR/submit_chemprop_train.slurm"
  sbatch "$SCRIPT_DIR/submit_chemprop_train.slurm"
}
trap 'echo "Received USR1 (near time limit)."; resubmit_self; exit 0' USR1

module load conda
conda activate latentvis

export PYTHONUNBUFFERED=1

echo "Job started at $(date -Iseconds)"
echo "PWD=$(pwd)"
echo "Input: $INPUT"
echo "Output (train split): $OUTPUT_DIR"
echo ""

set +e
python3 run_pipeline_from_embeddings.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --k-max 8 \
  --default-reduction pca \
  --clustering-method all \
  --data-split train \
  --topk-combine-train-test

EXIT=$?
set -e
echo ""
echo "Pipeline exit code: $EXIT"

COMPLETE=0
if [[ -f "$OUTPUT_DIR/sankey/all_sankey.json" ]] && \
   [[ -d "$OUTPUT_DIR/scatterplot" ]] && \
   [[ -f "$OUTPUT_DIR/topk/cluster_representatives.json" ]]; then
  COMPLETE=1
  echo "Output complete. All required files written for chemprop train split."
fi

echo "Job finished at $(date -Iseconds)"

if [[ $EXIT -ne 0 ]] || [[ $COMPLETE -eq 0 ]]; then
  echo ""
  echo "Pipeline incomplete or failed for chemprop train split. Auto-resubmitting..."
  resubmit_self
  exit 1
fi

cat > "$OUTPUT_DIR/README.md" << 'README_EOF'
# Latent Space Visualization - Frontend Data (Chemprop, train split)

This directory contains pipeline output for the **chemprop train split**:
embeddings from `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt`
restricted to the **train** split.

The layout mirrors `frontend/public/data/chemprop` (test split) but lives under
`frontend/public/data/chemprop/train` so that:

- Train and test scatterplots are kept separate.
- Interactive `/generate-range` updates can write to `scatterplot_current/`
  without overwriting baseline `scatterplot/` JSONs.

All JSON schemas match the test case (scatter, topk, substructures, Sankey, clustering metrics).
README_EOF
echo "Wrote README.md to $OUTPUT_DIR"

if [[ -f "$OUTPUT_DIR/real_id_to_smiles.json" ]]; then
  echo "Generating molecule images for chemprop train split..."
  python3 "$BACKEND_DIR/scripts/generate_molecule_images_from_smiles.py" --case-dir "$OUTPUT_DIR" || true
fi

# Layer transition stats (Sankey-consistent) for all case/split dirs
CHEMPROP_ORDER="fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,ffn_layer_1,ffn_layer_2"
CHEMELEON_ORDER="fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,gnn_layer_4,gnn_layer_5,gnn_layer_6,ffn_layer_1,ffn_layer_2"
echo "Running compute_layer_transition_stats (Sankey-aligned)..."
python3 "$SCRIPT_DIR/compute_layer_transition_stats.py" --layer-order "$CHEMPROP_ORDER" --layer-order-chemeleon "$CHEMELEON_ORDER" || true

exit 0
