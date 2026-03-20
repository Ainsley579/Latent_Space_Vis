#!/bin/bash
# Submit a KNN-only replacement job that overwrites only
# frontend/public/data/<case>/<split>/clustering/knn.
#
# Usage:
#   sbatch scripts/submit_refresh_knn_only.slurm <dataset:chemprop|chemeleon> <split:test|train|val> <knn_k> [time_hint]

#SBATCH --job-name=latentvis_knn_refresh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --constraint=cpu
#SBATCH --exclusive
#SBATCH --qos=regular
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/knn_refresh_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/knn_refresh_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail

DATASET="${1:-}"
SPLIT="${2:-}"
KNN_K="${3:-}"
TIME_HINT="${4:-}"

if [[ -z "$DATASET" || -z "$SPLIT" || -z "$KNN_K" ]]; then
  echo "Usage: sbatch $0 <dataset:chemprop|chemeleon> <split:test|train|val> <knn_k> [time_hint]"
  exit 2
fi

case "$DATASET" in
  chemprop|chemeleon) ;;
  *) echo "Invalid dataset: $DATASET"; exit 2 ;;
esac
case "$SPLIT" in
  test|train|val) ;;
  *) echo "Invalid split: $SPLIT"; exit 2 ;;
esac
if ! [[ "$KNN_K" =~ ^[0-9]+$ ]] || [[ "$KNN_K" -lt 1 ]]; then
  echo "Invalid knn_k: $KNN_K"
  exit 2
fi

SCRIPT_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/scripts"
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
REFRESH_SCRIPT="$SCRIPT_DIR/refresh_knn_only.sh"

if [[ ! -f "$REFRESH_SCRIPT" ]]; then
  echo "Missing refresh script: $REFRESH_SCRIPT"
  exit 1
fi

cd "$BACKEND_DIR"

resubmit_self() {
  if [[ -n "$TIME_HINT" ]]; then
    sbatch --time="$TIME_HINT" "$SCRIPT_DIR/submit_refresh_knn_only.slurm" "$DATASET" "$SPLIT" "$KNN_K" "$TIME_HINT"
  else
    sbatch "$SCRIPT_DIR/submit_refresh_knn_only.slurm" "$DATASET" "$SPLIT" "$KNN_K"
  fi
}

RESUBMITTED_ON_TIMEOUT=0
trap 'if [[ "$RESUBMITTED_ON_TIMEOUT" -eq 0 ]]; then RESUBMITTED_ON_TIMEOUT=1; echo "Received USR1 near time limit"; resubmit_self; fi' USR1

module load conda
conda activate latentvis
export PYTHONUNBUFFERED=1

echo "Job started at $(date -Iseconds)"
echo "dataset=$DATASET split=$SPLIT knn_k=$KNN_K"

set +e
bash "$REFRESH_SCRIPT" --dataset "$DATASET" --split "$SPLIT" --knn-k "$KNN_K"
RC=$?
set -e

echo "Refresh exit code: $RC"
echo "Job finished at $(date -Iseconds)"

if [[ $RC -ne 0 ]]; then
  echo "KNN-only refresh failed."
  exit 1
fi

exit 0
