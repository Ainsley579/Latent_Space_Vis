#!/usr/bin/env bash
# Materialize one ESOL-format dataset split into backend/data/<dataset>/<split>,
# then refresh metadata and compact overview outputs.
#
# Usage:
#   bash scripts/materialize_esol_split_data.sh --dataset chemprop --split val

set -euo pipefail

DATASET=""
SPLIT=""
KEEP_TEMP=0
SKIP_COMPACT=0
SKIP_METADATA=0

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
    --keep-temp)
      KEEP_TEMP=1
      shift
      ;;
    --skip-compact)
      SKIP_COMPACT=1
      shift
      ;;
    --skip-metadata)
      SKIP_METADATA=1
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
  train|val|test) ;;
  *) echo "Invalid or missing --split (train|val|test)"; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${LATENTVIS_DATA_ROOT:-$BACKEND_DIR/data}"
COMPACT_ROOT="$BACKEND_DIR/data_compact"
INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/${DATASET}-embeddings.pt"
TARGET_SPLIT_DIR="$DATA_ROOT/$DATASET/$SPLIT"
TMP_OUT="$(mktemp -d "/tmp/latentvis_${DATASET}_${SPLIT}.XXXXXX")"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/latentvis_numba_cache}"
mkdir -p "$NUMBA_CACHE_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cleanup() {
  if [[ "$KEEP_TEMP" -eq 0 && -d "$TMP_OUT" ]]; then
    rm -rf "$TMP_OUT"
  fi
}
trap cleanup EXIT

if [[ ! -d "$BACKEND_DIR" ]]; then echo "Missing backend dir: $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "Missing pipeline entrypoint"; exit 1; fi
if [[ ! -f "$INPUT" ]]; then echo "Missing input embeddings: $INPUT"; exit 1; fi

mkdir -p "$TARGET_SPLIT_DIR"
cd "$BACKEND_DIR"

echo "=== Materialize ESOL split start $(date -Iseconds) ==="
echo "dataset=$DATASET split=$SPLIT"
echo "input=$INPUT"
echo "target=$TARGET_SPLIT_DIR"
echo "temp_out=$TMP_OUT"

python3 run_pipeline_from_embeddings.py \
  --input "$INPUT" \
  --output-dir "$TMP_OUT" \
  --data-split "$SPLIT" \
  --default-reduction pca \
  --clustering-method all \
  --k-max 8

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required output: $path"
    exit 1
  fi
}

require_path "$TMP_OUT/scatterplot"
require_path "$TMP_OUT/topk/cluster_representatives.json"
require_path "$TMP_OUT/sankey/all_sankey.json"
require_path "$TMP_OUT/clustering_metrics/cluster_assignments_with_real_id.csv"
require_path "$TMP_OUT/clustering/hdbscan/scatterplot"
require_path "$TMP_OUT/clustering/agglomerative/scatterplot"
require_path "$TMP_OUT/clustering/knn/scatterplot"

replace_dir() {
  local src="$1"
  local dst="$2"
  rm -rf "$dst"
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
}

replace_baseline_dir() {
  local src_root="$1"
  local dst_root="$2"
  rm -rf "$dst_root"
  mkdir -p "$dst_root"
  for name in scatterplot bic_aic topk sankey clustering_metrics substructures; do
    if [[ -e "$src_root/$name" ]]; then
      cp -a "$src_root/$name" "$dst_root/"
    fi
  done
}

replace_baseline_dir "$TMP_OUT" "$TARGET_SPLIT_DIR/pgmeans"
replace_dir "$TMP_OUT/clustering/hdbscan" "$TARGET_SPLIT_DIR/hdbscan"
replace_dir "$TMP_OUT/clustering/agglomerative" "$TARGET_SPLIT_DIR/agglomerative"
replace_dir "$TMP_OUT/clustering/knn" "$TARGET_SPLIT_DIR/knn"

if [[ "$SKIP_METADATA" -ne 1 ]]; then
  python3 "$SCRIPT_DIR/write_id_smiles_y_with_split.py" \
    --input "$INPUT" \
    --output "$DATA_ROOT/$DATASET/id_smiles_y_true_y_pred.json" \
    --splits "test,train,val"

  CHEMPROP_ORDER="fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,ffn_layer_1,ffn_layer_2"
  CHEMELEON_ORDER="fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,gnn_layer_4,gnn_layer_5,gnn_layer_6,ffn_layer_1,ffn_layer_2"
  python3 "$SCRIPT_DIR/compute_layer_transition_stats.py" \
    --layer-order "$CHEMPROP_ORDER" \
    --layer-order-chemeleon "$CHEMELEON_ORDER"
fi

if [[ "$SKIP_METADATA" -ne 1 && "$SKIP_COMPACT" -ne 1 ]]; then
  python3 "$SCRIPT_DIR/build_compact_overview_data.py" \
    --output-root "$COMPACT_ROOT" \
    --coord-decimals 2
fi

echo "Materialized baseline outputs:"
echo "  $TARGET_SPLIT_DIR/pgmeans"
echo "  $TARGET_SPLIT_DIR/hdbscan"
echo "  $TARGET_SPLIT_DIR/agglomerative"
echo "  $TARGET_SPLIT_DIR/knn"
echo "=== Materialize ESOL split complete $(date -Iseconds) ==="
