#!/bin/bash
# Run full pipeline (with all 3 improvements: MCS substructures, cluster_sorted_members, multi-projection scatter)
# then copy output to frontend, replacing existing data.
#
# Usage:
#   ./run_all_and_copy_to_frontend.sh --input /path/to/embeddings.pt --output-dir ./pipeline_output [--smiles-csv /path/to/smiles.csv] [--model-name NAME]
#   ./run_all_and_copy_to_frontend.sh --copy-only ./pipeline_output   # skip pipeline, only copy
#
# Options (when running pipeline):
#   --input         Path to .pt file or directory of embeddings (required unless --copy-only)
#   --output-dir    Pipeline output directory (default: ./pipeline_output)
#   --smiles-csv    CSV with real_id and SMILES for MCS substructures (optional)
#   --model-name    Model/epoch name when input is a directory (optional)
#   --copy-only     Only copy given directory to frontend; do not run pipeline
#   --case          Case study: dimenet, chemprop, or chemeleon (where to copy under frontend/public/data/)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

COPY_ONLY=false
INPUT=""
OUTPUT_DIR="./pipeline_output"
SMILES_CSV=""
MODEL_NAME=""
CASE_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --copy-only)
      COPY_ONLY=true
      OUTPUT_DIR="${2:-./pipeline_output}"
      shift 2
      ;;
    --case)
      CASE_NAME="$2"
      shift 2
      ;;
    --input)
      INPUT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --smiles-csv)
      SMILES_CSV="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    *)
      if [[ -z "$INPUT" && "$1" != --* ]]; then
        INPUT="$1"
      elif [[ -z "$OUTPUT_DIR" && "$1" != --* ]]; then
        OUTPUT_DIR="$1"
      fi
      shift
      ;;
  esac
done

FRONTEND_DATA="${BACKEND_DIR}/../frontend/public/data"

if [[ "$COPY_ONLY" == true ]]; then
  echo "=== Copy-only: copying $OUTPUT_DIR -> frontend${CASE_NAME:+/$CASE_NAME} ==="
  if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Error: directory not found: $OUTPUT_DIR"
    exit 1
  fi
  bash "$SCRIPT_DIR/copy_to_frontend.sh" "$OUTPUT_DIR" $CASE_NAME
  echo "Done. Frontend data: $FRONTEND_DATA${CASE_NAME:+/$CASE_NAME}"
  exit 0
fi

if [[ -z "$INPUT" ]]; then
  echo "Error: --input is required (path to .pt or embedding directory). Use --copy-only <dir> to only copy."
  exit 1
fi

echo "=== 1/2 Running pipeline (MCS, cluster_sorted_members, PCA/MDS/UMAP/t-SNE scatter) ==="
PIPELINE_CMD=(python3 run_pipeline_from_embeddings.py --input "$INPUT" --output-dir "$OUTPUT_DIR")
[[ -n "$SMILES_CSV" ]] && PIPELINE_CMD+=(--smiles-csv "$SMILES_CSV")
[[ -n "$MODEL_NAME" ]] && PIPELINE_CMD+=(--model-name "$MODEL_NAME")
"${PIPELINE_CMD[@]}"

echo ""
echo "=== 2/2 Copying pipeline output to frontend${CASE_NAME:+/$CASE_NAME} ==="
bash "$SCRIPT_DIR/copy_to_frontend.sh" "$OUTPUT_DIR" $CASE_NAME

echo ""
echo "=== Done. Frontend${CASE_NAME:+ data/$CASE_NAME/} now has: sankey, scatterplot (pca/mds/umap/tsne + default), topk, cluster_sorted_members, substructures (if smiles-csv was given), clustering_metrics. ==="
