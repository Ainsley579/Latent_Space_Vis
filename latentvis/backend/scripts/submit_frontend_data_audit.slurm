#!/bin/bash
# Run the frontend-facing data integrity audit on a compute node.

#SBATCH --job-name=latentvis_audit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/audit_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/audit_%j.err

set -euo pipefail

BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"

export PATH="/global/homes/j/jiaqing/.conda/envs/latentvis/bin:${PATH}"
cd "$BACKEND_DIR"

python3 scripts/audit_frontend_data_contract.py
