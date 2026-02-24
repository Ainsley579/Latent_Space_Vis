#!/bin/bash
#SBATCH --job-name=chemprop_scatter_regen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemprop_scatter_regen_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/chemprop_scatter_regen_%j.err

set -euo pipefail

BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
INPUT="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt"
OUTPUT_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/frontend/public/data/chemprop"

cd "$BACKEND_DIR"

module load conda
conda activate latentvis

export PYTHONUNBUFFERED=1

echo "Regenerating scatterplots for chemprop at $(date -Iseconds)"
echo "Input: $INPUT"
echo "Case dir: $OUTPUT_DIR"

python3 "$BACKEND_DIR/scripts/regenerate_scatterplots.py" \
  --case-dir "$OUTPUT_DIR" \
  --input-pt "$INPUT" \
  --tsne-iters 350 \
  --random-state 42

echo "Done at $(date -Iseconds)"

