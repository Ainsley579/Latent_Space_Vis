#!/usr/bin/env bash
# Submit all sweep jobs:
#   datasets: chemprop, chemeleon
#   splits:   test, train
#   methods:  pgmeans (45), agglomerative (9), knn (default 3: k=2..4)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT="$SCRIPT_DIR/submit_sankey_param_grid.slurm"

if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing script: $SCRIPT"
  exit 1
fi

DATASETS=("chemprop" "chemeleon")
SPLITS=("test" "train")
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

echo "Submitting sweep jobs..."
for ds in "${DATASETS[@]}"; do
  for sp in "${SPLITS[@]}"; do
    for m in "${METHODS[@]}"; do
      t="$(time_for_method "$m")"
      echo "  sbatch --time=$t $SCRIPT $ds $sp $m $t"
      sbatch --time="$t" "$SCRIPT" "$ds" "$sp" "$m" "$t"
    done
  done
done

echo "Submitted all jobs."
