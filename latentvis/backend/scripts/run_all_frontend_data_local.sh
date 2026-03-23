#!/usr/bin/env bash
# Local end-to-end refresh for frontend-facing data without changing path layout.
# This script refreshes:
# - chemprop backend/data (joint clustering across train/val/test)
# - chemeleon backend/data (joint clustering across train/val/test)
# - dimenet backend/data plus frontend/public mirror
# - compact overview data
# - all ESOL sweep grids
# - final integrity audit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LATENTVIS_SKIP_ESOL_BASELINES="${LATENTVIS_SKIP_ESOL_BASELINES:-0}"
LATENTVIS_SKIP_DIMENET="${LATENTVIS_SKIP_DIMENET:-0}"
LATENTVIS_SKIP_COMPACT="${LATENTVIS_SKIP_COMPACT:-0}"
LATENTVIS_SKIP_SWEEPS="${LATENTVIS_SKIP_SWEEPS:-0}"
LATENTVIS_SKIP_AUDIT="${LATENTVIS_SKIP_AUDIT:-0}"

retry() {
  local attempts="$1"
  shift
  local i
  for i in $(seq 1 "$attempts"); do
    echo "[run] attempt=$i cmd=$*"
    if "$@"; then
      return 0
    fi
    echo "[warn] command failed on attempt $i"
    sleep 2
  done
  echo "[error] command failed after $attempts attempts: $*"
  return 1
}

remove_resume_dirs() {
  local root="$1"
  if [[ -d "$root" ]]; then
    find "$root" -name '.pipeline_resume' -type d -prune -exec rm -rf {} +
  fi
}

run_sweep_group() {
  local dataset="$1"
  local split="$2"
  local method="$3"
  python3 "$SCRIPT_DIR/prune_esol_sweeps.py" --dataset "$dataset" --split "$split" --method "$method"
  FORCE_RERUN=1 LATENTVIS_DISABLE_RESUBMIT=1 retry 3 \
    bash "$SCRIPT_DIR/submit_sankey_param_grid.slurm" "$dataset" "$split" "$method"
}

cd "$BACKEND_DIR"

if [[ "$LATENTVIS_SKIP_ESOL_BASELINES" != "1" ]]; then
  retry 2 bash "$SCRIPT_DIR/materialize_esol_split_data.sh" --dataset chemprop --split train
  retry 2 bash "$SCRIPT_DIR/materialize_esol_split_data.sh" --dataset chemeleon --split train
fi

if [[ "$LATENTVIS_SKIP_DIMENET" != "1" ]]; then
  retry 2 bash "$SCRIPT_DIR/run_dimenet_case_to_frontend.sh"
fi

remove_resume_dirs "$BACKEND_DIR/data/dimenet"

if [[ "$LATENTVIS_SKIP_COMPACT" != "1" ]]; then
  retry 2 python3 "$SCRIPT_DIR/build_compact_overview_data.py" --output-root "$BACKEND_DIR/data_compact" --coord-decimals 2
fi

if [[ "$LATENTVIS_SKIP_SWEEPS" != "1" ]]; then
  for dataset in chemprop chemeleon; do
    for split in train val test; do
      for method in pgmeans agglomerative knn hdbscan; do
        run_sweep_group "$dataset" "$split" "$method"
      done
    done
  done
fi

if [[ "$LATENTVIS_SKIP_AUDIT" != "1" ]]; then
  python3 "$SCRIPT_DIR/audit_frontend_data_contract.py"
fi
echo "[done] local frontend data refresh completed"
