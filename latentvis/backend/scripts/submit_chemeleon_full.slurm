#!/bin/bash
# SBATCH job for chemeleon full pipeline: single CPU, exclusive node, 4hr.
# Auto-resubmits if not complete (timeout before writing outputs).
#
# Usage: sbatch scripts/submit_chemeleon_full.slurm
#
# Writes directly to frontend/public/data/chemeleon/
# Same structure as chemprop; chemeleon uses same ESOL embedding format.

#SBATCH --job-name=chemeleon_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --constraint=cpu
#SBATCH --exclusive
#SBATCH --qos=regular
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemeleon_pipeline_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemeleon_pipeline_%j.err

set -e
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
SCRIPT_DIR="$BACKEND_DIR/scripts"
INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt"
FRONTEND_DATA="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/frontend/public/data"
OUTPUT_DIR="$FRONTEND_DATA/chemeleon/test"

# Fail fast: verify all paths exist before running
if [[ ! -d "$BACKEND_DIR" ]]; then echo "ERROR: BACKEND_DIR missing: $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "ERROR: run_pipeline_from_embeddings.py not found in $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$INPUT" ]]; then echo "ERROR: Input file missing: $INPUT"; exit 1; fi
if [[ ! -d "$(dirname "$FRONTEND_DATA")" ]]; then echo "ERROR: Frontend base dir missing: $(dirname "$FRONTEND_DATA")"; exit 1; fi

cd "$BACKEND_DIR" || { echo "ERROR: cd $BACKEND_DIR failed"; exit 1; }

module load conda
conda activate latentvis

export PYTHONUNBUFFERED=1

echo "Job started at $(date -Iseconds)"
echo "PWD=$(pwd)"
echo "Input: $INPUT"
echo "Output: $OUTPUT_DIR"
echo ""

python3 run_pipeline_from_embeddings.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --k-max 8 \
  --default-reduction pca \
  --clustering-method all \
  --data-split test

EXIT=$?
echo ""
echo "Pipeline exit code: $EXIT"

COMPLETE=0
if [[ -f "$OUTPUT_DIR/sankey/all_sankey.json" ]] && \
   [[ -d "$OUTPUT_DIR/scatterplot" ]] && \
   [[ -f "$OUTPUT_DIR/topk/cluster_representatives.json" ]]; then
  COMPLETE=1
  echo "Output complete. All required files written."
fi

echo "Job finished at $(date -Iseconds)"

if [[ $EXIT -ne 0 ]] || [[ $COMPLETE -eq 0 ]]; then
  echo ""
  echo "Pipeline incomplete or failed. Auto-resubmitting..."
  sbatch "$SCRIPT_DIR/submit_chemeleon_full.slurm"
  exit 1
fi

cat > "$OUTPUT_DIR/README.md" << 'README_EOF'
# Latent Space Visualization - Chemeleon Case

Pipeline output for **chemeleon** embeddings: `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt`.

## Directory Structure

Same as chemprop (see `../README.md` or parent for full layout):

| Path | Description |
|------|-------------|
| `sankey/all_sankey.json` | 2D Sankey: epoch and layer axes. |
| `sankey/sankey_samples.json` | Per-sample rows for Sankey-scatter linking. |
| `scatterplot/{epoch}_{layer}.json` | Default scatter (PCA). |
| `scatterplot/{epoch}_{layer}_pca.json`, `_mds.json`, `_umap.json`, `_tsne.json` | Alternate projections. |
| `topk/cluster_representatives.json` | Top-k closest/farthest per cluster. |
| `topk/cluster_sorted_members.json` | Full sorted list per cluster. |
| `bic_aic/`, `clustering_metrics/` | GMM curves and cluster assignments. |
| `clustering/hdbscan/`, `clustering/agglomerative/` | Alternate clustering methods. |

## Frontend Usage

Point frontend to `/data/chemeleon/` (this directory) to serve chemeleon case. Structure is identical to chemprop.

## Source

Pipeline: `latentvis/backend/run_pipeline_from_embeddings.py`
README_EOF
echo "Wrote README.md to $OUTPUT_DIR"

# Generate molecule images from real_id_to_smiles.json for frontend KNN view
if [[ -f "$OUTPUT_DIR/real_id_to_smiles.json" ]]; then
  echo "Generating molecule images..."
  python3 "$BACKEND_DIR/scripts/generate_molecule_images_from_smiles.py" --case-dir "$OUTPUT_DIR" || true
fi

# Layer transition stats (Sankey-consistent) for all case/split dirs
CHEMPROP_ORDER="fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,ffn_layer_1,ffn_layer_2"
CHEMELEON_ORDER="fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,gnn_layer_4,gnn_layer_5,gnn_layer_6,ffn_layer_1,ffn_layer_2"
echo "Running compute_layer_transition_stats (Sankey-aligned)..."
python3 "$SCRIPT_DIR/compute_layer_transition_stats.py" --layer-order "$CHEMPROP_ORDER" --layer-order-chemeleon "$CHEMELEON_ORDER" || true

exit 0
