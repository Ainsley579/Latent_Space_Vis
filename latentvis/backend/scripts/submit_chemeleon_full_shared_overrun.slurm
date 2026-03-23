#!/bin/bash
# Shared-overrun variant for chemeleon full-case materialization.

#SBATCH --job-name=chemeleon_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=shared_overrun
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemeleon_pipeline_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemeleon_pipeline_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail
LATENTVIS_SUBMIT_MODE="${LATENTVIS_SUBMIT_MODE:-shared_overrun}"
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
SCRIPT_DIR="$BACKEND_DIR/scripts"
INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt"
CASE_ROOT="$BACKEND_DIR/data/chemeleon"
REGULAR_SCRIPT="$SCRIPT_DIR/submit_chemeleon_full.slurm"
SHARED_SCRIPT="$SCRIPT_DIR/submit_chemeleon_full_shared.slurm"
OVERRUN_SCRIPT="$SCRIPT_DIR/submit_chemeleon_full_shared_overrun.slurm"
source "$SCRIPT_DIR/slurm_fallback_lib.sh"

if [[ ! -d "$BACKEND_DIR" ]]; then echo "ERROR: BACKEND_DIR missing: $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "ERROR: run_pipeline_from_embeddings.py not found in $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$INPUT" ]]; then echo "ERROR: Input file missing: $INPUT"; exit 1; fi

cd "$BACKEND_DIR" || { echo "ERROR: cd $BACKEND_DIR failed"; exit 1; }

resubmit_self() {
  local sbatch_args=()
  local script_args=()
  echo "Resubmitting chemeleon baseline with mode=${LATENTVIS_SUBMIT_MODE}"
  latentvis_submit_with_fallback "$REGULAR_SCRIPT" "$SHARED_SCRIPT" "$OVERRUN_SCRIPT" sbatch_args script_args
}
RESUBMITTED=0
resubmit_once() {
  local reason="$1"
  if [[ "$RESUBMITTED" -eq 0 ]]; then
    RESUBMITTED=1
    echo "Resubmitting due to $reason"
    resubmit_self
  fi
}
on_exit() {
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    resubmit_once "exit_rc=$rc"
  fi
  exit "$rc"
}
trap 'echo "Received USR1 (near time limit)."; resubmit_once "signal_usr1"; exit 0' USR1
trap 'echo "Received TERM from scheduler."; resubmit_once "signal_term"; exit 0' TERM
trap on_exit EXIT

source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate latentvis

export PYTHONUNBUFFERED=1

echo "Job started at $(date -Iseconds)"
echo "PWD=$(pwd)"
echo "Input: $INPUT"
echo "Output: $CASE_ROOT"
echo "Submit mode: $LATENTVIS_SUBMIT_MODE"
echo ""

set +e
bash "$SCRIPT_DIR/materialize_esol_split_data.sh" --dataset chemeleon --split test
EXIT=$?
set -e
echo ""
echo "Materialization exit code: $EXIT"

COMPLETE=0
if [[ -f "$CASE_ROOT/train/pgmeans/sankey/all_sankey.json" ]] && \
   [[ -f "$CASE_ROOT/val/pgmeans/sankey/all_sankey.json" ]] && \
   [[ -f "$CASE_ROOT/test/pgmeans/sankey/all_sankey.json" ]]; then
  COMPLETE=1
  echo "Output complete. Joint split-aware artifacts were written."
fi

echo "Job finished at $(date -Iseconds)"

if [[ $EXIT -ne 0 ]] || [[ $COMPLETE -eq 0 ]]; then
  echo ""
  echo "Materialization incomplete or failed. Auto-resubmitting..."
  resubmit_once "completion_check_failed"
  exit 1
fi

exit 0
