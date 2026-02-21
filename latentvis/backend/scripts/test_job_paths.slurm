#!/bin/bash
# Minimal 1-min job to verify paths on compute node. Run: sbatch scripts/test_job_paths.slurm
#SBATCH --job-name=test_paths
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:02:00
#SBATCH --constraint=cpu
#SBATCH -A m636
#SBATCH --qos=regular
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/test_job_paths_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/test_job_paths_%j.err

set -e
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
INPUT_C="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt"
INPUT_L="/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt"

echo "Initial PWD=$PWD"
if [[ ! -d "$BACKEND_DIR" ]]; then echo "ERROR: BACKEND_DIR missing"; exit 1; fi
if [[ ! -f "$BACKEND_DIR/run_pipeline_from_embeddings.py" ]]; then echo "ERROR: pipeline script missing"; exit 1; fi
if [[ ! -f "$INPUT_C" ]]; then echo "ERROR: chemprop input missing"; exit 1; fi
if [[ ! -f "$INPUT_L" ]]; then echo "ERROR: chemeleon input missing"; exit 1; fi
cd "$BACKEND_DIR" || exit 1
echo "After cd PWD=$PWD"
echo "SUCCESS: All paths valid on compute node"
