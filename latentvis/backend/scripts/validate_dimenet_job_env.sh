#!/bin/bash
# Validate environment and code before submitting DimeNet jobs.
# Run from backend: ./scripts/validate_dimenet_job_env.sh
# Exit 0 only if all checks pass; submit scripts should run this before sbatch.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/global/homes/j/jiaqing/.conda/envs/latentvis}"
export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
PY="${CONDA_ENV_PATH}/bin/python"

echo "=== Validate DimeNet job environment ==="
echo "  Python: $PY"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Python not found at $PY. Set CONDA_ENV_PATH or install latentvis env."
  exit 1
fi

echo ""
echo "1. Required packages..."
"$PY" -c "
import sys
missing = []
for name, imp in [('numpy','numpy'),('pandas','pandas'),('sklearn','sklearn'),('torch','torch'),('umap','umap'),('rdkit','rdkit')]:
    try:
        if imp=='rdkit': __import__('rdkit')
        else: __import__(imp)
    except Exception:
        missing.append(name)
if missing:
    print('MISSING:', missing)
    sys.exit(1)
print('  numpy, pandas, sklearn, torch, umap-learn, rdkit OK')
"

echo ""
echo "2. Pipeline CLI and embedding loader..."
"$PY" run_pipeline_from_embeddings.py --help > /dev/null
echo "  run_pipeline_from_embeddings.py --help OK"

echo ""
echo "3. Both DimeNet inputs load with n_samples=152..."
"$PY" -c "
import json, os
from pathlib import Path
os.chdir('$BACKEND_DIR')
import sys
sys.path.insert(0, '.')
from embedding_loader import load_embedding_input
p1 = Path('/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet_noaug_embeddings_cas_152')
p2 = Path('/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings')
m1 = json.loads(Path('$SCRIPT_DIR/layer_map_dimenet_noaug_cas.json').read_text())
m2 = json.loads(Path('$SCRIPT_DIR/layer_map_dimenet10conf_cas.json').read_text())
d1 = load_embedding_input(p1, model_name='noaug', layer_file_map=m1)
d2 = load_embedding_input(p2, model_name='10conf', layer_file_map=m2)
assert d1['n_samples'] == 152 and d2['n_samples'] == 152, (d1['n_samples'], d2['n_samples'])
print('  dimenet_noaug n_samples=152, dimenet10conf n_samples=152 OK')
"

echo ""
echo "4. generate_cluster_substructures.py (no BooleanOptionalAction)..."
"$PY" "$SCRIPT_DIR/generate_cluster_substructures.py" --help > /dev/null
echo "  generate_cluster_substructures.py --help OK"

echo ""
echo "5. Minimal pipeline run (one layer, one epoch, dry run)..."
OUT_TEST="$BACKEND_DIR/pipeline_validate_test_out"
rm -rf "$OUT_TEST"
"$PY" run_pipeline_from_embeddings.py \
  --input /global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet_noaug_embeddings_cas_152 \
  --model-name dimenet_noaug \
  --layer-file-map "$SCRIPT_DIR/layer_map_dimenet_noaug_cas.json" \
  --output-dir "$OUT_TEST" \
  --k-max 5 \
  --skip-tsne \
  --smiles-csv "$BACKEND_DIR/data/smiles/real_id_smiles.csv" 2>&1 | tail -25
if [[ ! -f "$OUT_TEST/scatterplot/dimenet_noaug_emb.json" ]]; then
  echo "ERROR: Pipeline did not produce scatterplot JSON"
  exit 1
fi
if [[ ! -f "$OUT_TEST/topk/cluster_sorted_members.json" ]]; then
  echo "ERROR: Pipeline did not produce cluster_sorted_members.json"
  exit 1
fi
rm -rf "$OUT_TEST"
echo "  Pipeline run OK (scatter + topk + cluster_sorted_members)"

echo ""
echo "=== All validations passed. Safe to submit jobs. ==="
