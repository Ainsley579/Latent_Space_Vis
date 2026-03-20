#!/bin/bash
# Submit one ESOL split materialization worker.
#
# Usage:
#   sbatch scripts/submit_materialize_esol_split.slurm <dataset:chemprop|chemeleon> <split:train|val|test> [time_hint]

#SBATCH --job-name=latentvis_split_materialize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --constraint=cpu
#SBATCH --exclusive
#SBATCH --qos=regular
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/materialize_split_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/materialize_split_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail

DATASET="${1:-}"
SPLIT="${2:-}"
TIME_HINT="${3:-}"

if [[ -z "$DATASET" || -z "$SPLIT" ]]; then
  echo "Usage: sbatch $0 <dataset:chemprop|chemeleon> <split:train|val|test> [time_hint]"
  exit 2
fi

case "$DATASET" in
  chemprop|chemeleon) ;;
  *) echo "Invalid dataset: $DATASET"; exit 2 ;;
esac
case "$SPLIT" in
  train|val|test) ;;
  *) echo "Invalid split: $SPLIT"; exit 2 ;;
esac

SCRIPT_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/scripts"
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
RUN_SCRIPT="$SCRIPT_DIR/materialize_esol_split_data.sh"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing materialize script: $RUN_SCRIPT"
  exit 1
fi

cd "$BACKEND_DIR"

resubmit_self() {
  if [[ -n "$TIME_HINT" ]]; then
    sbatch --time="$TIME_HINT" "$SCRIPT_DIR/submit_materialize_esol_split.slurm" "$DATASET" "$SPLIT" "$TIME_HINT"
  else
    sbatch "$SCRIPT_DIR/submit_materialize_esol_split.slurm" "$DATASET" "$SPLIT"
  fi
}

RESUBMITTED_ON_TIMEOUT=0
trap 'if [[ "$RESUBMITTED_ON_TIMEOUT" -eq 0 ]]; then RESUBMITTED_ON_TIMEOUT=1; echo "Received USR1 near time limit"; resubmit_self; fi' USR1

module load conda
conda activate latentvis
export PYTHONUNBUFFERED=1

echo "Job started at $(date -Iseconds)"
echo "dataset=$DATASET split=$SPLIT"

set +e
bash "$RUN_SCRIPT" --dataset "$DATASET" --split "$SPLIT"
RC=$?
set -e

echo "Materialize exit code: $RC"
echo "Job finished at $(date -Iseconds)"

if [[ $RC -ne 0 ]]; then
  echo "Split materialization failed."
  exit 1
fi

exit 0
