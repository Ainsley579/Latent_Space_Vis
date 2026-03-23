#!/bin/bash
# Shared-overrun variant for one parameter-sweep worker.

#SBATCH --job-name=latentvis_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=shared_overrun
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/sweep_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/sweep_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail
LATENTVIS_SUBMIT_MODE="${LATENTVIS_SUBMIT_MODE:-shared_overrun}"

DATASET="${1:-}"
SPLIT="${2:-}"
METHOD="${3:-}"
TIME_HINT="${4:-}"

if [[ -z "$DATASET" || -z "$SPLIT" || -z "$METHOD" ]]; then
  echo "Usage: sbatch $0 <dataset:chemprop|chemeleon> <split:test|train|val> <method:pgmeans|agglomerative|knn|hdbscan> [time_hint]"
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
case "$METHOD" in
  pgmeans|agglomerative|knn|hdbscan) ;;
  *) echo "Invalid method: $METHOD"; exit 2 ;;
esac

AGG_LINKAGES_RAW="${AGG_LINKAGES:-ward,complete,average}"
KNN_K_MIN="${KNN_K_MIN:-2}"
KNN_K_MAX="${KNN_K_MAX:-10}"
FORCE_RERUN="${FORCE_RERUN:-0}"
LATENTVIS_DISABLE_RESUBMIT="${LATENTVIS_DISABLE_RESUBMIT:-0}"

SCRIPT_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/scripts"
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
DATA_ROOT="${LATENTVIS_DATA_ROOT:-$BACKEND_DIR/data}"
INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/${DATASET}-embeddings.pt"
OUT_ROOT="$DATA_ROOT/$DATASET/$SPLIT/sweeps/$METHOD"
REGULAR_SCRIPT="$SCRIPT_DIR/submit_sankey_param_grid.slurm"
SHARED_SCRIPT="$SCRIPT_DIR/submit_sankey_param_grid_shared.slurm"
OVERRUN_SCRIPT="$SCRIPT_DIR/submit_sankey_param_grid_shared_overrun.slurm"
source "$SCRIPT_DIR/slurm_fallback_lib.sh"

if [[ ! -d "$BACKEND_DIR" ]]; then echo "ERROR: BACKEND_DIR missing: $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "ERROR: run_pipeline_from_embeddings.py not found"; exit 1; fi
if [[ ! -f "$INPUT" ]]; then echo "ERROR: Input missing: $INPUT"; exit 1; fi
echo "[paths] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "[paths] SCRIPT_DIR=$SCRIPT_DIR"
echo "[paths] BACKEND_DIR=$BACKEND_DIR"
echo "[paths] DATA_ROOT=$DATA_ROOT"
echo "[paths] submit_mode=$LATENTVIS_SUBMIT_MODE"

mkdir -p "$OUT_ROOT"
cd "$BACKEND_DIR"

resubmit_self() {
  local sbatch_args=()
  local script_args=("$DATASET" "$SPLIT" "$METHOD" "$TIME_HINT")
  sbatch_args+=(--export="ALL,FORCE_RERUN=$FORCE_RERUN,AGG_LINKAGES=$AGG_LINKAGES_RAW,KNN_K_MIN=$KNN_K_MIN,KNN_K_MAX=$KNN_K_MAX,LATENTVIS_SUBMIT_MODE=$LATENTVIS_SUBMIT_MODE")
  if [[ -n "$TIME_HINT" ]]; then
    sbatch_args+=(--time="$TIME_HINT")
  fi
  echo "[resubmit] dataset=$DATASET split=$SPLIT method=$METHOD mode=$LATENTVIS_SUBMIT_MODE"
  latentvis_submit_with_fallback "$REGULAR_SCRIPT" "$SHARED_SCRIPT" "$OVERRUN_SCRIPT" sbatch_args script_args
}

RESUBMITTED=0
resubmit_once() {
  local reason="$1"
  if [[ "$LATENTVIS_DISABLE_RESUBMIT" == "1" ]]; then
    echo "[trap] resubmit disabled reason=$reason"
    return 0
  fi
  if [[ "$RESUBMITTED" -eq 0 ]]; then
    RESUBMITTED=1
    echo "[trap] resubmit reason=$reason"
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
trap 'resubmit_once "signal_usr1"' USR1
trap on_exit EXIT

source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate latentvis
export PYTHONUNBUFFERED=1

is_complete_dir() {
  local d="$1"
  [[ -f "$d/sankey/all_sankey.json" ]] && \
  [[ -f "$d/sankey/sankey_samples.json" ]] && \
  [[ -f "$d/clustering_metrics/cluster_assignments_with_real_id.csv" ]] && \
  [[ -d "$d/scatterplot" ]]
}

write_meta() {
  local d="$1"
  local params_json="$2"
  cat > "$d/run_meta.json" <<EOF
{
  "dataset": "$DATASET",
  "split": "$SPLIT",
  "method": "$METHOD",
  "params": $params_json
}
EOF
}

run_one() {
  local tag="$1"
  local params_json="$2"
  shift 2
  local out_dir="$OUT_ROOT/$tag"
  local tmp_root
  local tmp_case_root
  local tmp_output_root
  local split_method_dir

  if [[ "$FORCE_RERUN" != "1" ]] && is_complete_dir "$out_dir"; then
    echo "[skip] complete: $out_dir"
    return 0
  fi

  if [[ "$FORCE_RERUN" == "1" ]] && [[ -d "$out_dir" ]]; then
    echo "[overwrite] removing existing output: $out_dir"
    rm -rf "$out_dir"
  fi

  mkdir -p "$OUT_ROOT"
  tmp_root="$(mktemp -d "/tmp/latentvis_sweep_${DATASET}_${SPLIT}_${METHOD}_${tag}.XXXXXX")"
  tmp_case_root="$tmp_root/case"
  tmp_output_root="$tmp_root/all_samples"
  mkdir -p "$tmp_case_root" "$tmp_output_root"
  for seed_split in train val test; do
    local seed_scatter_dir="$DATA_ROOT/$DATASET/$seed_split/pgmeans/scatterplot"
    if [[ -d "$seed_scatter_dir" ]]; then
      mkdir -p "$tmp_case_root/$seed_split/pgmeans"
      cp -a "$seed_scatter_dir" "$tmp_case_root/$seed_split/pgmeans/"
    fi
  done
  echo "[run] $DATASET/$SPLIT/$METHOD/$tag"
  set +e
  python3 run_pipeline_from_embeddings.py \
    --input "$INPUT" \
    --output-dir "$tmp_output_root" \
    --split-output-root "$tmp_case_root" \
    --reuse-scatter-coords \
    --default-reduction pca \
    "$@"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[error] pipeline failed (rc=$rc): $out_dir"
    rm -rf "$tmp_root"
    return $rc
  fi
  if [[ "$METHOD" == "pgmeans" ]]; then
    split_method_dir="$tmp_case_root/$SPLIT/pgmeans"
  else
    split_method_dir="$tmp_case_root/$SPLIT/$METHOD"
  fi
  if [[ ! -d "$split_method_dir" ]]; then
    echo "[error] missing split output: $split_method_dir"
    rm -rf "$tmp_root"
    return 1
  fi
  rm -rf "$out_dir"
  cp -a "$split_method_dir" "$out_dir"
  rm -rf "$out_dir/.pipeline_resume"
  if ! is_complete_dir "$out_dir"; then
    echo "[error] incomplete outputs: $out_dir"
    rm -rf "$tmp_root"
    return 1
  fi
  write_meta "$out_dir" "$params_json"
  rm -rf "$tmp_root"
  return 0
}

echo "=== Sweep start $(date -Iseconds) ==="
echo "dataset=$DATASET split=$SPLIT method=$METHOD out_root=$OUT_ROOT"

case "$METHOD" in
  pgmeans)
    for kmin in {2..10}; do
      for kmax in $(seq "$kmin" 10); do
        tag="kmin_${kmin}_kmax_${kmax}"
        params="{\"k_min\": $kmin, \"k_max\": $kmax}"
        run_one "$tag" "$params" --clustering-method pgmeans --k-min "$kmin" --k-max "$kmax" || exit 1
      done
    done
    ;;
  agglomerative)
    IFS=',' read -r -a _raw_links <<< "$AGG_LINKAGES_RAW"
    LINKS=()
    for lk in "${_raw_links[@]}"; do
      ll="$(echo "$lk" | tr '[:upper:]' '[:lower:]' | xargs)"
      case "$ll" in
        ward|complete|average)
          LINKS+=("$ll")
          ;;
      esac
    done
    if [[ ${#LINKS[@]} -eq 0 ]]; then
      LINKS=("ward" "complete" "average")
    fi
    for linkage in "${LINKS[@]}"; do
      for k in {2..10}; do
        tag="k_${k}_linkage_${linkage}"
        params="{\"k\": $k, \"linkage\": \"${linkage}\"}"
        run_one "$tag" "$params" \
          --clustering-method agglomerative \
          --agg-k "$k" \
          --agg-linkage "$linkage" \
          --k-min 2 --k-max 10 || exit 1
      done
    done
    ;;
  knn)
    for k in $(seq "$KNN_K_MIN" "$KNN_K_MAX"); do
      tag="k_${k}"
      params="{\"k\": $k}"
      run_one "$tag" "$params" \
        --clustering-method knn \
        --knn-k "$k" \
        --k-min 2 --k-max 10 || exit 1
    done
    ;;
  hdbscan)
    tag="default"
    params="{}"
    run_one "$tag" "$params" --clustering-method hdbscan --k-min 2 --k-max 10 || exit 1
    ;;
esac

export LATENTVIS_SWEEP_OUT_ROOT="$OUT_ROOT"
python3 - <<'PY'
import json
import os
from pathlib import Path
out_root = Path(os.environ["LATENTVIS_SWEEP_OUT_ROOT"])
rows = []
for d in sorted([p for p in out_root.iterdir() if p.is_dir()]):
    m = d / "run_meta.json"
    if not m.is_file():
        continue
    try:
        meta = json.loads(m.read_text(encoding="utf-8"))
    except Exception:
        continue
    meta["path"] = str(d)
    rows.append(meta)
(out_root / "manifest.json").write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
print("Wrote manifest:", out_root / "manifest.json", "runs=", len(rows))
PY

echo "=== Sweep complete $(date -Iseconds) ==="
exit 0
