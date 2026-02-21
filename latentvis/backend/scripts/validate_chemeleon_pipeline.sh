#!/bin/bash
# Quick validation for chemeleon case: run short pipeline to verify compatibility.
# Usage: ./validate_chemeleon_pipeline.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

module load conda 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate latentvis 2>/dev/null || { source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null && conda activate latentvis; }

TEST_OUT="$BACKEND_DIR/pipeline_validate_chemeleon"
rm -rf "$TEST_OUT"
mkdir -p "$TEST_OUT"

echo "=== Chemeleon validation (2 epochs, 9 layers, 113 samples) ==="
python3 run_pipeline_from_embeddings.py \
  --input /global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt \
  --output-dir "$TEST_OUT" \
  --k-max 8 \
  --max-epochs 2 \
  --skip-tsne \
  --clustering-method all

echo ""
echo "=== Validation checks ==="
[[ -f "$TEST_OUT/sankey/all_sankey.json" ]] && echo "  [OK] Sankey"
[[ -d "$TEST_OUT/scatterplot" ]] && echo "  [OK] Scatterplot"
[[ -f "$TEST_OUT/topk/cluster_representatives.json" ]] && echo "  [OK] Topk"
[[ -d "$TEST_OUT/clustering/hdbscan" ]] && echo "  [OK] HDBSCAN"
[[ -d "$TEST_OUT/clustering/agglomerative" ]] && echo "  [OK] Agglomerative"
echo "Validation done. Output: $TEST_OUT"
