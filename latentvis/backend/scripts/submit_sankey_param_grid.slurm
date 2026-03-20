#!/bin/bash
# Submit one parameter-sweep worker for a single (dataset, split, method).
# Auto-resubmits itself on failure or near time limit; completed configs are skipped.
#
# Usage:
#   sbatch scripts/submit_sankey_param_grid.slurm <dataset> <split> <method> [time_hint]
# Example:
#   sbatch --time=08:00:00 scripts/submit_sankey_param_grid.slurm chemprop test pgmeans 08:00:00

#SBATCH --job-name=latentvis_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --constraint=cpu
#SBATCH --exclusive
#SBATCH --qos=regular
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/sweep_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/sweep_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail

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

# For METHOD=agglomerative, optionally restrict linkage list via env:
#   AGG_LINKAGES=complete,average  (default: ward,complete,average)
AGG_LINKAGES_RAW="${AGG_LINKAGES:-ward,complete,average}"
# For METHOD=knn, default to the narrower range that most often yields 3-5 clusters.
# Override with env when needed, e.g. KNN_K_MIN=2 KNN_K_MAX=5 sbatch ...
KNN_K_MIN="${KNN_K_MIN:-2}"
KNN_K_MAX="${KNN_K_MAX:-4}"
# Set FORCE_RERUN=1 to replace existing completed sweep outputs instead of skipping them.
FORCE_RERUN="${FORCE_RERUN:-0}"

SCRIPT_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/scripts"
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
DATA_ROOT="${LATENTVIS_DATA_ROOT:-$BACKEND_DIR/data}"
INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/${DATASET}-embeddings.pt"
OUT_ROOT="$DATA_ROOT/$DATASET/$SPLIT/sweeps/$METHOD"

if [[ ! -d "$BACKEND_DIR" ]]; then echo "ERROR: BACKEND_DIR missing: $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "ERROR: run_pipeline_from_embeddings.py not found"; exit 1; fi
if [[ ! -f "$INPUT" ]]; then echo "ERROR: Input missing: $INPUT"; exit 1; fi
echo "[paths] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "[paths] SCRIPT_DIR=$SCRIPT_DIR"
echo "[paths] BACKEND_DIR=$BACKEND_DIR"
echo "[paths] DATA_ROOT=$DATA_ROOT"

mkdir -p "$OUT_ROOT"
cd "$BACKEND_DIR"

resubmit_self() {
  echo "[resubmit] dataset=$DATASET split=$SPLIT method=$METHOD"
  if [[ -n "$TIME_HINT" ]]; then
    sbatch --time="$TIME_HINT" "$SCRIPT_DIR/submit_sankey_param_grid.slurm" "$DATASET" "$SPLIT" "$METHOD" "$TIME_HINT"
  else
    sbatch "$SCRIPT_DIR/submit_sankey_param_grid.slurm" "$DATASET" "$SPLIT" "$METHOD"
  fi
}

RESUBMITTED_ON_TIMEOUT=0
trap 'if [[ "$RESUBMITTED_ON_TIMEOUT" -eq 0 ]]; then RESUBMITTED_ON_TIMEOUT=1; echo "[trap] Received USR1 near time limit"; resubmit_self; fi' USR1

module load conda
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

  if [[ "$FORCE_RERUN" != "1" ]] && is_complete_dir "$out_dir"; then
    echo "[skip] complete: $out_dir"
    return 0
  fi

  if [[ "$FORCE_RERUN" == "1" ]] && [[ -d "$out_dir" ]]; then
    echo "[overwrite] removing existing output: $out_dir"
    rm -rf "$out_dir"
  fi

  mkdir -p "$out_dir"
  echo "[run] $DATASET/$SPLIT/$METHOD/$tag"
  set +e
  python3 run_pipeline_from_embeddings.py \
    --input "$INPUT" \
    --output-dir "$out_dir" \
    --data-split "$SPLIT" \
    --default-reduction pca \
    "$@"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[error] pipeline failed (rc=$rc): $out_dir"
    return $rc
  fi
  if ! is_complete_dir "$out_dir"; then
    echo "[error] incomplete outputs: $out_dir"
    return 1
  fi
  write_meta "$out_dir" "$params_json"
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

# Refresh manifest
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
