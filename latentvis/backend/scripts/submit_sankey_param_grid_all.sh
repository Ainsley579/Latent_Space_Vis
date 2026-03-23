#!/usr/bin/env bash
# Submit all sweep jobs:
#   datasets: chemprop, chemeleon
#   splits:   test, train, val
#   methods:  pgmeans (45), agglomerative (27), knn (9: k=2..10), hdbscan (1)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT="$SCRIPT_DIR/submit_sankey_param_grid.slurm"
SHARED_SCRIPT="$SCRIPT_DIR/submit_sankey_param_grid_shared.slurm"
OVERRUN_SCRIPT="$SCRIPT_DIR/submit_sankey_param_grid_shared_overrun.slurm"
PRUNE_SCRIPT="$SCRIPT_DIR/prune_esol_sweeps.py"
FORCE_RERUN="${FORCE_RERUN:-1}"
CLEAN_STALE="${CLEAN_STALE:-1}"
CHEMPROP_AFTEROK_JOB="${CHEMPROP_AFTEROK_JOB:-}"
CHEMELEON_AFTEROK_JOB="${CHEMELEON_AFTEROK_JOB:-}"
AGG_LINKAGES="${AGG_LINKAGES:-ward,complete,average}"
KNN_K_MIN="${KNN_K_MIN:-2}"
KNN_K_MAX="${KNN_K_MAX:-10}"
source "$SCRIPT_DIR/slurm_fallback_lib.sh"

if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing script: $SCRIPT"
  exit 1
fi
if [[ ! -f "$PRUNE_SCRIPT" ]]; then
  echo "Missing prune script: $PRUNE_SCRIPT"
  exit 1
fi

DATASETS=("chemprop" "chemeleon")
SPLITS=("test" "train" "val")
METHODS=("pgmeans" "agglomerative" "knn")
METHODS+=("hdbscan")

time_for_method() {
  case "$1" in
    pgmeans) echo "08:00:00" ;;
    agglomerative) echo "06:00:00" ;;
    knn) echo "06:00:00" ;;
    hdbscan) echo "03:00:00" ;;
    *) echo "04:00:00" ;;
  esac
}

dependency_for_dataset() {
  case "$1" in
    chemprop) echo "$CHEMPROP_AFTEROK_JOB" ;;
    chemeleon) echo "$CHEMELEON_AFTEROK_JOB" ;;
    *) echo "" ;;
  esac
}

echo "Submitting sweep jobs..."
for ds in "${DATASETS[@]}"; do
  for sp in "${SPLITS[@]}"; do
    for m in "${METHODS[@]}"; do
      t="$(time_for_method "$m")"
      if [[ "$CLEAN_STALE" == "1" ]]; then
        python3 "$PRUNE_SCRIPT" --dataset "$ds" --split "$sp" --method "$m"
      fi
      sbatch_args=(
        --time="$t"
        --export="ALL,FORCE_RERUN=$FORCE_RERUN,AGG_LINKAGES=$AGG_LINKAGES,KNN_K_MIN=$KNN_K_MIN,KNN_K_MAX=$KNN_K_MAX"
      )
      script_args=("$ds" "$sp" "$m" "$t")
      dep="$(dependency_for_dataset "$ds")"
      if [[ -n "$dep" ]]; then
        sbatch_args+=(--dependency="afterok:$dep")
      fi
      echo "  submit dataset=$ds split=$sp method=$m mode=${LATENTVIS_SUBMIT_MODE:-auto}"
      latentvis_submit_with_fallback "$SCRIPT" "$SHARED_SCRIPT" "$OVERRUN_SCRIPT" sbatch_args script_args
    done
  done
done

echo "Submitted all jobs."
