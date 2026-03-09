#!/bin/bash
# SBATCH job for chemprop full pipeline: single CPU, exclusive node, 2hr.
# Auto-resubmits if not complete (timeout before writing outputs).
#
# Usage: sbatch scripts/submit_chemprop_full.slurm
#   Or: ./submit_chemprop_full.slurm  (generates job script and submits)
#
# Writes to frontend/public/data/chemprop (avoids overwriting DimeNet data in data/).
# Run validate_chemprop_pipeline.sh first to confirm code works.

#SBATCH --job-name=chemprop_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --constraint=cpu
#SBATCH --exclusive
#SBATCH --qos=regular
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemprop_pipeline_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemprop_pipeline_%j.err
#SBATCH --signal=B:USR1@300

set -e
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
SCRIPT_DIR="$BACKEND_DIR/scripts"
INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt"
FRONTEND_DATA="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/frontend/public/data"
OUTPUT_DIR="$FRONTEND_DATA/chemprop/test"

# Fail fast: verify all paths exist before running (catches wrong cwd/path on compute node)
if [[ ! -d "$BACKEND_DIR" ]]; then echo "ERROR: BACKEND_DIR missing: $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "ERROR: run_pipeline_from_embeddings.py not found in $BACKEND_DIR"; exit 1; fi
if [[ ! -f "$INPUT" ]]; then echo "ERROR: Input file missing: $INPUT"; exit 1; fi
if [[ ! -d "$(dirname "$FRONTEND_DATA")" ]]; then echo "ERROR: Frontend base dir missing: $(dirname "$FRONTEND_DATA")"; exit 1; fi

cd "$BACKEND_DIR" || { echo "ERROR: cd $BACKEND_DIR failed"; exit 1; }

resubmit_self() {
  echo "Resubmitting job script: $SCRIPT_DIR/submit_chemprop_full.slurm"
  sbatch "$SCRIPT_DIR/submit_chemprop_full.slurm"
}
trap 'echo "Received USR1 (near time limit)."; resubmit_self; exit 0' USR1

module load conda
conda activate latentvis

export PYTHONUNBUFFERED=1

echo "Job started at $(date -Iseconds)"
echo "PWD=$(pwd)"
echo "Input: $INPUT"
echo "Output: $OUTPUT_DIR"
echo ""

# Run full pipeline: all epochs, all four scatter projections (pca/mds/umap/tsne), all 3 clustering methods
set +e
python3 run_pipeline_from_embeddings.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --k-max 8 \
  --default-reduction pca \
  --clustering-method all \
  --data-split test \
  --topk-combine-train-test

EXIT=$?
set -e
echo ""
echo "Pipeline exit code: $EXIT"

# Check if complete: sankey, scatterplot, topk exist under OUTPUT_DIR
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
  resubmit_self
  exit 1
fi

# Write README for this run's data directory (English)
cat > "$OUTPUT_DIR/README.md" << 'README_EOF'
# Latent Space Visualization - Frontend Data (Chemprop)

This directory contains pipeline output for the **chemprop** case: embeddings from `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt`.

## Directory Structure

| Path | Description |
|------|-------------|
| `sankey/all_sankey.json` | 2D Sankey: epoch (flow across layers) and layer (flow across epochs). |
| `sankey/sankey_samples.json` | Per-sample rows: real_id, from_layer, to_layer, from_cluster, to_cluster. |
| `scatterplot/{epoch}_{layer}.json` | Default 2D scatter (method: PCA). |
| `scatterplot/{epoch}_{layer}_pca.json` | PCA projection. |
| `scatterplot/{epoch}_{layer}_mds.json` | MDS projection. |
| `scatterplot/{epoch}_{layer}_umap.json` | UMAP projection. |
| `scatterplot/{epoch}_{layer}_tsne.json` | t-SNE projection. |
| `topk/cluster_representatives.json` | Per (epoch, layer, cluster): top-k closest/farthest real_ids. |
| `topk/cluster_sorted_members.json` | Per (epoch, layer, cluster): full list of real_ids sorted by distance to center. |
| `bic_aic/{epoch}_{layer}.json` | BIC/AIC curves for PGMeans model selection. |
| `clustering_metrics/cluster_assignments_with_real_id.csv` | real_id, epoch, per-layer cluster columns. |
| `clustering/hdbscan/` | Same layout for HDBSCAN clustering. |
| `clustering/agglomerative/` | Same layout for Agglomerative clustering. |

## Frontend Usage

- **Scatter**: Use `/data/scatterplot/*.json`. Switch projection via dropdown (PCA, MDS, UMAP, t-SNE).
- **Sankey**: Use `/data/sankey/all_sankey.json` and `sankey_samples.json`.
- **Clusters**: Use `/data/topk/cluster_representatives.json` and `cluster_sorted_members.json`.
- **Molecule view**: `real_id` in scatter/topk links to molecule lookups.

## Source

Pipeline: `latentvis/backend/run_pipeline_from_embeddings.py`  
Input: chemprop-embeddings.pt (ESOL structure: epoch -> split -> embeddings -> layer -> tensor)
README_EOF
echo "Wrote README.md to $OUTPUT_DIR"

# Generate molecule images from real_id_to_smiles.json for frontend KNN view
if [[ -f "$OUTPUT_DIR/real_id_to_smiles.json" ]]; then
  echo "Generating molecule images..."
  python3 "$BACKEND_DIR/scripts/generate_molecule_images_from_smiles.py" --case-dir "$OUTPUT_DIR" || true
fi

# Layer transition stats (Sankey-consistent percentages) for all case/split dirs that have CSVs
CHEMPROP_ORDER="fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,ffn_layer_1,ffn_layer_2"
CHEMELEON_ORDER="fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,gnn_layer_4,gnn_layer_5,gnn_layer_6,ffn_layer_1,ffn_layer_2"
echo "Running compute_layer_transition_stats (Sankey-aligned)..."
python3 "$SCRIPT_DIR/compute_layer_transition_stats.py" --layer-order "$CHEMPROP_ORDER" --layer-order-chemeleon "$CHEMELEON_ORDER" || true

exit 0
