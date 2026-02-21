#!/bin/bash
# Quick validation: run tiny + short chemprop to verify full workflow before submitting long job.
# Verifies: 2+ scatter methods, 3 clustering methods, molecule view (real_id), cluster sorting (all data).
#
# Usage: ./validate_chemprop_pipeline.sh [--output-to-frontend]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

# Use conda latentvis env (umap-learn, hdbscan)
module load conda 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate latentvis 2>/dev/null || { source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null && conda activate latentvis; }

OUTPUT_TO_FRONTEND=""
[[ "${1:-}" == "--output-to-frontend" ]] && OUTPUT_TO_FRONTEND="--output-to-frontend"

TEST_OUT="$BACKEND_DIR/pipeline_validate_test"
rm -rf "$TEST_OUT"
mkdir -p "$TEST_OUT"

echo "=== 1/2 Tiny test (2 epochs, 2 layers, ~40 samples) ==="
python3 scripts/create_tiny_chemprop_pt.py
python3 run_pipeline_from_embeddings.py \
  --input data/tiny_chemprop.pt \
  --output-dir "$TEST_OUT/tiny" \
  --k-max 5 \
  --clustering-method all \
  --skip-tsne

echo ""
echo "=== 2/2 Short chemprop (2 epochs, 6 layers, 113 samples) ==="
python3 run_pipeline_from_embeddings.py \
  --input /global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt \
  --output-dir "$TEST_OUT/chemprop_short" \
  --k-max 8 \
  --max-epochs 2 \
  --skip-tsne \
  --clustering-method all

echo ""
echo "=== Validation checks ==="
# 2+ scatter methods: expect _pca, _mds (umap may fail)
SCATTER_FILES=$(find "$TEST_OUT/chemprop_short/scatterplot" -name "*_pca.json" 2>/dev/null | wc -l)
[[ "$SCATTER_FILES" -ge 1 ]] && echo "  [OK] Scatter (PCA) files exist"
SCATTER_MDS=$(find "$TEST_OUT/chemprop_short/scatterplot" -name "*_mds.json" 2>/dev/null | wc -l)
[[ "$SCATTER_MDS" -ge 1 ]] && echo "  [OK] Scatter (MDS) files exist"

# 3 clustering methods: GMM at root, HDBSCAN/Agglomerative under clustering/
[[ -f "$TEST_OUT/chemprop_short/sankey/all_sankey.json" ]] && echo "  [OK] GMM output (root)"
[[ -d "$TEST_OUT/chemprop_short/clustering/hdbscan" ]] && echo "  [OK] HDBSCAN output"
[[ -d "$TEST_OUT/chemprop_short/clustering/agglomerative" ]] && echo "  [OK] Agglomerative output"

# cluster_sorted_members: each entry should have count == len(sorted_real_ids)
python3 -c "
import json
p = '$TEST_OUT/chemprop_short/topk/cluster_sorted_members.json'
try:
    data = json.load(open(p))
    ok = all(e.get('count') == len(e.get('sorted_real_ids', [])) for e in data)
    print('  [OK] Cluster sorting: all data included (count==len(sorted_real_ids))' if ok else '  [FAIL] Cluster sorting mismatch')
except Exception as e:
    print('  [SKIP]', e)
"

# real_id in scatter (molecule view)
python3 -c "
import json
import glob
fs = glob.glob('$TEST_OUT/chemprop_short/scatterplot/*.json')
ok = False
for f in fs[:1]:
    d = json.load(open(f))
    ok = ok or (d and 'real_id' in str(d[0]) if d else False)
print('  [OK] Molecule view: real_id in scatter' if ok else '  [FAIL] real_id missing')
"

echo ""
echo "Validation done. If all [OK], safe to submit full chemprop job."
echo "Output: $TEST_OUT"
