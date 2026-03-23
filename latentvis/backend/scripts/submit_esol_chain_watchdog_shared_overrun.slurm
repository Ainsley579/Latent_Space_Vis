#!/bin/bash
# Shared-overrun watchdog for the remaining ESOL chain.

#SBATCH --job-name=latentvis_watch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=shared_overrun
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/watchdog_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/watchdog_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
SCRIPT_DIR="$BACKEND_DIR/scripts"
WATCH_SCRIPT="$SCRIPT_DIR/watch_esol_chain_and_fallback.sh"

cd "$BACKEND_DIR"

resubmit_self() {
  sbatch "$SCRIPT_DIR/submit_esol_chain_watchdog_shared_overrun.slurm"
}
RESUBMITTED=0
resubmit_once() {
  local reason="$1"
  if [[ "$RESUBMITTED" -eq 0 ]]; then
    RESUBMITTED=1
    echo "[watchdog] resubmitting due to $reason"
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
trap 'resubmit_once "signal_usr1"; exit 0' USR1
trap on_exit EXIT

source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate latentvis

export PYTHONUNBUFFERED=1
bash "$WATCH_SCRIPT"
