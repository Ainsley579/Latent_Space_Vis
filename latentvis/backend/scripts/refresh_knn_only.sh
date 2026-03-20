#!/usr/bin/env bash
# Recompute only KNN outputs for a case/split and replace the existing baseline
# at frontend/public/data/<case>/<split>/clustering/knn in place.
#
# Usage:
#   bash scripts/refresh_knn_only.sh --dataset chemprop --split val --knn-k 4

set -euo pipefail

DATASET=""
SPLIT=""
KNN_K=""
KEEP_TEMP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    --split)
      SPLIT="${2:-}"
      shift 2
      ;;
    --knn-k)
      KNN_K="${2:-}"
      shift 2
      ;;
    --keep-temp)
      KEEP_TEMP=1
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

case "$DATASET" in
  chemprop|chemeleon) ;;
  *) echo "Invalid or missing --dataset (chemprop|chemeleon)"; exit 2 ;;
esac
case "$SPLIT" in
  test|train|val) ;;
  *) echo "Invalid or missing --split (test|train|val)"; exit 2 ;;
esac
if ! [[ "$KNN_K" =~ ^[0-9]+$ ]] || [[ "$KNN_K" -lt 1 ]]; then
  echo "Invalid or missing --knn-k (positive integer)"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${LATENTVIS_DATA_ROOT:-$BACKEND_DIR/data}"
INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/${DATASET}-embeddings.pt"
TARGET_DIR="$DATA_ROOT/$DATASET/$SPLIT"
TARGET_KNN_DIR="$TARGET_DIR/knn"
TMP_OUT="$(mktemp -d "/tmp/latentvis_knn_${DATASET}_${SPLIT}_k${KNN_K}.XXXXXX")"

cleanup() {
  if [[ "$KEEP_TEMP" -eq 0 && -d "$TMP_OUT" ]]; then
    rm -rf "$TMP_OUT"
  fi
}
trap cleanup EXIT

if [[ ! -d "$BACKEND_DIR" ]]; then echo "Missing backend dir: $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "Missing pipeline entrypoint"; exit 1; fi
if [[ ! -f "$INPUT" ]]; then echo "Missing input embeddings: $INPUT"; exit 1; fi
if [[ ! -d "$TARGET_DIR" ]]; then echo "Missing target dataset dir: $TARGET_DIR"; exit 1; fi

cd "$BACKEND_DIR"

echo "=== KNN-only refresh start $(date -Iseconds) ==="
echo "dataset=$DATASET split=$SPLIT knn_k=$KNN_K"
echo "input=$INPUT"
echo "target=$TARGET_KNN_DIR"
echo "temp_out=$TMP_OUT"

python3 run_pipeline_from_embeddings.py \
  --input "$INPUT" \
  --output-dir "$TMP_OUT" \
  --data-split "$SPLIT" \
  --default-reduction pca \
  --clustering-method knn \
  --knn-k "$KNN_K" \
  --k-min 2 \
  --k-max 10 \
  --no-fmi-cspa \
  --reuse-scatter-coords \
  --quiet-dr

for required in \
  "$TMP_OUT/scatterplot" \
  "$TMP_OUT/topk/cluster_representatives.json" \
  "$TMP_OUT/sankey/all_sankey.json" \
  "$TMP_OUT/clustering_metrics/cluster_assignments_with_real_id.csv"
do
  if [[ ! -e "$required" ]]; then
    echo "Missing required temp output: $required"
    exit 1
  fi
done

rm -rf "$TARGET_KNN_DIR"
mkdir -p "$TARGET_KNN_DIR"
for name in scatterplot bic_aic topk sankey clustering_metrics substructures; do
  if [[ -e "$TMP_OUT/$name" ]]; then
    cp -a "$TMP_OUT/$name" "$TARGET_KNN_DIR/"
  fi
done

cat > "$TARGET_KNN_DIR/run_meta.json" <<EOF
{
  "dataset": "$DATASET",
  "split": "$SPLIT",
  "method": "knn",
  "params": {
    "k": $KNN_K
  }
}
EOF

echo "Replaced baseline KNN outputs at $TARGET_KNN_DIR"
echo "=== KNN-only refresh complete $(date -Iseconds) ==="
