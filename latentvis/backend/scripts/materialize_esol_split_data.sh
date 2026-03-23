#!/usr/bin/env bash
# Materialize ESOL-format dataset outputs into the existing
# backend/data/<dataset>/{train,val,test}/... layout.
# Clustering is performed jointly across all available splits, then the outputs are
# filtered back into the existing split directories so paths stay unchanged.
#
# Usage:
#   bash scripts/materialize_esol_split_data.sh --dataset chemprop --split val

set -euo pipefail

DATASET=""
SPLIT=""
KEEP_TEMP=0
SKIP_COMPACT=0
SKIP_METADATA=0
REUSE_SCATTER_COORDS="${LATENTVIS_REUSE_SCATTER_COORDS:-0}"

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
TARGET_CASE_DIR="$DATA_ROOT/$DATASET"
PERSISTENT_WORK_ROOT="${LATENTVIS_MATERIALIZE_WORK_ROOT:-$BACKEND_DIR/logs/resume/materialize/$DATASET}"
EPHEMERAL_WORK_ROOT="${LATENTVIS_EPHEMERAL_WORK_ROOT:-0}"
if [[ "$EPHEMERAL_WORK_ROOT" == "1" ]]; then
  TMP_OUT="$(mktemp -d "/tmp/latentvis_${DATASET}_joint.XXXXXX")"
  PERSIST_WORK=0
else
  TMP_OUT="$PERSISTENT_WORK_ROOT"
  PERSIST_WORK=1
  mkdir -p "$TMP_OUT"
fi
TMP_CASE_ROOT="$TMP_OUT/case_root"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/latentvis_numba_cache}"
mkdir -p "$NUMBA_CACHE_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cleanup() {
  if [[ "$PERSIST_WORK" -eq 0 && "$KEEP_TEMP" -eq 0 && -d "$TMP_OUT" ]]; then
    rm -rf "$TMP_OUT"
  fi
}
trap cleanup EXIT

if [[ ! -d "$BACKEND_DIR" ]]; then echo "Missing backend dir: $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "Missing pipeline entrypoint"; exit 1; fi
if [[ ! -f "$INPUT" ]]; then echo "Missing input embeddings: $INPUT"; exit 1; fi

mkdir -p "$TARGET_CASE_DIR"
for stale_rel in \
  case_root \
  scatterplot \
  bic_aic \
  topk \
  sankey \
  clustering_metrics \
  clustering \
  substructures \
  real_id_to_smiles.json
do
  if [[ -e "$TMP_OUT/$stale_rel" ]]; then
    rm -rf "$TMP_OUT/$stale_rel"
  fi
done
reuse_args=()
if [[ "$REUSE_SCATTER_COORDS" == "1" ]]; then
  for seed_split in train val test; do
    seed_scatter_dir="$TARGET_CASE_DIR/$seed_split/pgmeans/scatterplot"
    if [[ -d "$seed_scatter_dir" ]]; then
      mkdir -p "$TMP_CASE_ROOT/$seed_split/pgmeans"
      cp -a "$seed_scatter_dir" "$TMP_CASE_ROOT/$seed_split/pgmeans/"
    fi
  done
  reuse_args+=(--reuse-scatter-coords)
fi
cd "$BACKEND_DIR"

echo "=== Materialize ESOL case start $(date -Iseconds) ==="
echo "dataset=$DATASET requested_split=$SPLIT"
echo "input=$INPUT"
echo "target_case_root=$TARGET_CASE_DIR"
echo "temp_out=$TMP_OUT"
echo "temp_case_root=$TMP_CASE_ROOT"
echo "persistent_work_root=$PERSISTENT_WORK_ROOT"
echo "persistent_resume_enabled=$PERSIST_WORK"

python3 run_pipeline_from_embeddings.py \
  --input "$INPUT" \
  --output-dir "$TMP_OUT" \
  --split-output-root "$TMP_CASE_ROOT" \
  --default-reduction pca \
  --clustering-method all \
  --k-max 8 \
  "${reuse_args[@]}"

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

for split_name in train val test; do
  require_path "$TMP_CASE_ROOT/$split_name/pgmeans/scatterplot"
  require_path "$TMP_CASE_ROOT/$split_name/pgmeans/topk/cluster_representatives.json"
  require_path "$TMP_CASE_ROOT/$split_name/pgmeans/sankey/all_sankey.json"
  require_path "$TMP_CASE_ROOT/$split_name/pgmeans/clustering_metrics/cluster_assignments_with_real_id.csv"
  require_path "$TMP_CASE_ROOT/$split_name/hdbscan/scatterplot"
  require_path "$TMP_CASE_ROOT/$split_name/agglomerative/scatterplot"
  require_path "$TMP_CASE_ROOT/$split_name/knn/scatterplot"
done

for split_name in train val test; do
  mkdir -p "$TARGET_CASE_DIR/$split_name"
  for method_name in pgmeans hdbscan agglomerative knn; do
    rm -rf "$TARGET_CASE_DIR/$split_name/$method_name"
    cp -a "$TMP_CASE_ROOT/$split_name/$method_name" "$TARGET_CASE_DIR/$split_name/$method_name"
  done
done

if [[ -f "$TMP_OUT/real_id_to_smiles.json" ]]; then
  cp -f "$TMP_OUT/real_id_to_smiles.json" "$TARGET_CASE_DIR/real_id_to_smiles.json"
fi

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

  python3 "$SCRIPT_DIR/build_scatter_bundle.py" \
    --dataset "$DATASET" \
    --clustering "pgmeans,agglomerative,hdbscan,knn"

  python3 "$SCRIPT_DIR/build_scatter_xy_bundle.py" \
    --dataset "$DATASET" \
    --input-pt "$INPUT"
fi

if [[ "$SKIP_METADATA" -ne 1 && "$SKIP_COMPACT" -ne 1 ]]; then
  python3 "$SCRIPT_DIR/build_compact_overview_data.py" \
    --output-root "$COMPACT_ROOT" \
    --coord-decimals 2
fi

echo "Materialized joint-clustering outputs:"
echo "  $TARGET_CASE_DIR/train/pgmeans"
echo "  $TARGET_CASE_DIR/train/hdbscan"
echo "  $TARGET_CASE_DIR/train/agglomerative"
echo "  $TARGET_CASE_DIR/train/knn"
echo "  $TARGET_CASE_DIR/val/pgmeans"
echo "  $TARGET_CASE_DIR/val/hdbscan"
echo "  $TARGET_CASE_DIR/val/agglomerative"
echo "  $TARGET_CASE_DIR/val/knn"
echo "  $TARGET_CASE_DIR/test/pgmeans"
echo "  $TARGET_CASE_DIR/test/hdbscan"
echo "  $TARGET_CASE_DIR/test/agglomerative"
echo "  $TARGET_CASE_DIR/test/knn"
echo "=== Materialize ESOL case complete $(date -Iseconds) ==="
