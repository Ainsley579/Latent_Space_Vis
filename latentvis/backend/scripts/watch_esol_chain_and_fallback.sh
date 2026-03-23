#!/usr/bin/env bash
# Watch the remaining ESOL jobs and replace blocked regular submissions with
# shared-node submissions when needed. Paths and output layout stay unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${LATENTVIS_DATA_ROOT:-$BACKEND_DIR/data}"
WATCH_SLEEP_SECONDS="${LATENTVIS_WATCH_SLEEP_SECONDS:-300}"
MAX_LOOPS="${LATENTVIS_WATCH_MAX_LOOPS:-0}"
SLURM_USER="${LATENTVIS_SLURM_USER:-${USER:-jiaqing}}"
source "$SCRIPT_DIR/slurm_fallback_lib.sh"

time_for_method() {
  case "$1" in
    pgmeans) echo "08:00:00" ;;
    agglomerative) echo "06:00:00" ;;
    knn) echo "06:00:00" ;;
    hdbscan) echo "03:00:00" ;;
    *) echo "04:00:00" ;;
  esac
}

expected_count_for_method() {
  case "$1" in
    pgmeans) echo 45 ;;
    agglomerative) echo 27 ;;
    knn) echo 9 ;;
    hdbscan) echo 1 ;;
    *) echo 0 ;;
  esac
}

is_complete_dir() {
  local d="$1"
  [[ -f "$d/sankey/all_sankey.json" ]] && \
  [[ -f "$d/sankey/sankey_samples.json" ]] && \
  [[ -f "$d/clustering_metrics/cluster_assignments_with_real_id.csv" ]] && \
  [[ -d "$d/scatterplot" ]]
}

baseline_complete() {
  local dataset="$1"
  local split
  local method
  for split in train val test; do
    for method in pgmeans agglomerative knn hdbscan; do
      if ! is_complete_dir "$DATA_ROOT/$dataset/$split/$method"; then
        return 1
      fi
    done
  done
  [[ -f "$DATA_ROOT/$dataset/real_id_to_smiles.json" ]] || return 1
  [[ -f "$DATA_ROOT/$dataset/id_smiles_y_true_y_pred.json" ]] || return 1
  return 0
}

sweep_complete() {
  local dataset="$1"
  local split="$2"
  local method="$3"
  local root="$DATA_ROOT/$dataset/$split/sweeps/$method"
  local expected
  local count
  expected="$(expected_count_for_method "$method")"
  if [[ ! -d "$root" ]]; then
    return 1
  fi
  count="$(find "$root" -mindepth 1 -maxdepth 1 -type d \
    -exec test -f '{}/sankey/all_sankey.json' ';' \
    -exec test -f '{}/sankey/sankey_samples.json' ';' \
    -exec test -f '{}/clustering_metrics/cluster_assignments_with_real_id.csv' ';' \
    -exec test -d '{}/scatterplot' ';' -print | wc -l | tr -d ' ')"
  [[ "$count" == "$expected" ]]
}

load_queue_snapshot() {
  QUEUE_SNAPSHOT="$(squeue -h -u "$SLURM_USER" -o "%i|%T|%R|%o" 2>/dev/null || true)"
}

find_job_line() {
  local needle="$1"
  if [[ -z "${QUEUE_SNAPSHOT:-}" ]]; then
    return 1
  fi
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    IFS='|' read -r jobid state reason cmd <<<"$line"
    if [[ "$cmd" == *"$needle"* ]]; then
      printf '%s\n' "$line"
      return 0
    fi
  done <<<"$QUEUE_SNAPSHOT"
  return 1
}

line_needs_shared() {
  local line="$1"
  local jobid state reason cmd
  IFS='|' read -r jobid state reason cmd <<<"$line"
  latentvis_needs_shared_reason "${state}|${reason}"
}

line_needs_replacement() {
  local line="$1"
  local jobid state reason cmd
  IFS='|' read -r jobid state reason cmd <<<"$line"
  if latentvis_needs_shared_reason "${state}|${reason}"; then
    return 0
  fi
  [[ "$state" == "PENDING" && "$reason" == *"DependencyNeverSatisfied"* ]]
}

submit_baseline() {
  local dataset="$1"
  local regular_script="$SCRIPT_DIR/submit_${dataset}_full.slurm"
  local shared_script="$SCRIPT_DIR/submit_${dataset}_full_shared.slurm"
  local overrun_script="$SCRIPT_DIR/submit_${dataset}_full_shared_overrun.slurm"
  local sbatch_args=()
  local script_args=()
  latentvis_submit_with_fallback "$regular_script" "$shared_script" "$overrun_script" sbatch_args script_args
}

submit_sweep() {
  local dataset="$1"
  local split="$2"
  local method="$3"
  local time_hint
  local sbatch_args=()
  local script_args=("$dataset" "$split" "$method")
  local regular_script="$SCRIPT_DIR/submit_sankey_param_grid.slurm"
  local shared_script="$SCRIPT_DIR/submit_sankey_param_grid_shared.slurm"
  local overrun_script="$SCRIPT_DIR/submit_sankey_param_grid_shared_overrun.slurm"
  time_hint="$(time_for_method "$method")"
  script_args+=("$time_hint")
  python3 "$SCRIPT_DIR/prune_esol_sweeps.py" --dataset "$dataset" --split "$split" --method "$method"
  sbatch_args+=(--time="$time_hint")
  sbatch_args+=(--export="ALL,FORCE_RERUN=1,AGG_LINKAGES=ward,complete,average,KNN_K_MIN=2,KNN_K_MAX=10")
  latentvis_submit_with_fallback "$regular_script" "$shared_script" "$overrun_script" sbatch_args script_args
}

ensure_baseline() {
  local dataset="$1"
  local line
  local jobid state reason cmd
  if baseline_complete "$dataset"; then
    echo "[baseline] complete: $dataset"
    return 0
  fi
  if line="$(find_job_line "submit_${dataset}_full")"; then
    IFS='|' read -r jobid state reason cmd <<<"$line"
    if line_needs_replacement "$line" && [[ "$cmd" != *"_shared.slurm"* ]]; then
      echo "[baseline] replacing blocked regular job $jobid for $dataset because reason=$reason"
      scancel "$jobid" >/dev/null 2>&1 || true
      if [[ "$reason" == *"DependencyNeverSatisfied"* ]]; then
        submit_baseline "$dataset"
      else
        LATENTVIS_SUBMIT_MODE=shared submit_baseline "$dataset"
      fi
    else
      echo "[baseline] existing job for $dataset: jobid=$jobid state=$state reason=$reason"
    fi
    return 0
  fi
  echo "[baseline] submitting missing job for $dataset"
  submit_baseline "$dataset"
}

ensure_sweep() {
  local dataset="$1"
  local split="$2"
  local method="$3"
  local line
  local jobid state reason cmd
  local needle="submit_sankey_param_grid"
  if ! baseline_complete "$dataset"; then
    echo "[sweep] waiting for baseline: $dataset/$split/$method"
    return 0
  fi
  if sweep_complete "$dataset" "$split" "$method"; then
    echo "[sweep] complete: $dataset/$split/$method"
    return 0
  fi
  if line="$(find_job_line "$needle $dataset $split $method")"; then
    IFS='|' read -r jobid state reason cmd <<<"$line"
    if line_needs_replacement "$line" && [[ "$cmd" != *"_shared.slurm"* ]]; then
      echo "[sweep] replacing blocked regular job $jobid for $dataset/$split/$method because reason=$reason"
      scancel "$jobid" >/dev/null 2>&1 || true
      if [[ "$reason" == *"DependencyNeverSatisfied"* ]]; then
        submit_sweep "$dataset" "$split" "$method"
      else
        LATENTVIS_SUBMIT_MODE=shared submit_sweep "$dataset" "$split" "$method"
      fi
    else
      echo "[sweep] existing job for $dataset/$split/$method: jobid=$jobid state=$state reason=$reason"
    fi
    return 0
  fi
  echo "[sweep] submitting missing job for $dataset/$split/$method"
  submit_sweep "$dataset" "$split" "$method"
}

all_complete() {
  local dataset
  local split
  local method
  for dataset in chemprop chemeleon; do
    baseline_complete "$dataset" || return 1
    for split in train val test; do
      for method in pgmeans agglomerative knn hdbscan; do
        sweep_complete "$dataset" "$split" "$method" || return 1
      done
    done
  done
  return 0
}

loop_no=0
while true; do
  loop_no=$((loop_no + 1))
  echo "[watch] loop=$loop_no start=$(date -Iseconds)"
  load_queue_snapshot

  ensure_baseline chemprop
  ensure_baseline chemeleon

  for dataset in chemprop chemeleon; do
    for split in train val test; do
      for method in pgmeans agglomerative knn hdbscan; do
        ensure_sweep "$dataset" "$split" "$method"
      done
    done
  done

  if all_complete; then
    echo "[watch] all ESOL outputs complete; running final audit"
    python3 "$SCRIPT_DIR/audit_frontend_data_contract.py"
    echo "[watch] final audit passed"
    break
  fi

  if [[ "$MAX_LOOPS" != "0" && "$loop_no" -ge "$MAX_LOOPS" ]]; then
    echo "[watch] reached MAX_LOOPS=$MAX_LOOPS before completion"
    exit 1
  fi

  echo "[watch] incomplete; sleeping ${WATCH_SLEEP_SECONDS}s"
  sleep "$WATCH_SLEEP_SECONDS"
done
