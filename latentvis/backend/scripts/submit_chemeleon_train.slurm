#!/bin/bash
# Compatibility wrapper for refreshing the chemeleon case with joint clustering
# and validating that the train split was materialized.
#
# Usage: sbatch scripts/submit_chemeleon_train.slurm

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
#SBATCH --signal=B:USR1@300

set -euo pipefail
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
SCRIPT_DIR="$BACKEND_DIR/scripts"
CASE_DIR="$BACKEND_DIR/data/chemeleon"

if [[ ! -d "$BACKEND_DIR" ]]; then echo "ERROR: BACKEND_DIR missing: $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$SCRIPT_DIR/materialize_esol_split_data.sh" ]]; then echo "ERROR: materialize_esol_split_data.sh missing"; exit 1; fi

cd "$BACKEND_DIR" || { echo "ERROR: cd $BACKEND_DIR failed"; exit 1; }

resubmit_self() {
  echo "Resubmitting job script: $SCRIPT_DIR/submit_chemeleon_train.slurm"
  sbatch "$SCRIPT_DIR/submit_chemeleon_train.slurm"
}
RESUBMITTED=0
resubmit_once() {
  if [[ "$RESUBMITTED" -eq 0 ]]; then
    RESUBMITTED=1
    resubmit_self
  fi
}
on_exit() {
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "Job failed with rc=$rc"
    resubmit_once
  fi
  exit "$rc"
}
trap 'echo "Received USR1 (near time limit)."; resubmit_once' USR1
trap on_exit EXIT

source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate latentvis

export PYTHONUNBUFFERED=1

echo "Job started at $(date -Iseconds)"
echo "PWD=$(pwd)"
echo "Case root: $CASE_DIR"
echo ""

set +e
bash "$SCRIPT_DIR/materialize_esol_split_data.sh" \
  --dataset chemeleon \
  --split train

EXIT=$?
set -e
echo ""
echo "Pipeline exit code: $EXIT"

COMPLETE=0
if [[ -f "$CASE_DIR/train/pgmeans/sankey/all_sankey.json" ]] && \
   [[ -d "$CASE_DIR/train/pgmeans/scatterplot" ]] && \
   [[ -f "$CASE_DIR/val/pgmeans/sankey/all_sankey.json" ]] && \
   [[ -f "$CASE_DIR/test/pgmeans/sankey/all_sankey.json" ]]; then
  COMPLETE=1
  echo "Output complete. Joint clustering materialized all chemeleon splits."
fi

echo "Job finished at $(date -Iseconds)"

if [[ $EXIT -ne 0 ]] || [[ $COMPLETE -eq 0 ]]; then
  echo ""
  echo "Pipeline incomplete or failed for chemeleon. Auto-resubmitting..."
  resubmit_once
  exit 1
fi

exit 0
