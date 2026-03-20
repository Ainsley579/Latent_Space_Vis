#!/bin/bash
#SBATCH --job-name=latentvis_dimenet_case
#SBATCH --output=/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/dimenet_case_%j.out
#SBATCH --error=/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/dimenet_case_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -A m636
#SBATCH --signal=B:USR1@300

set -euo pipefail

SCRIPT_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/scripts"
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"

mkdir -p "${BACKEND_DIR}/logs/slurm"

RESUBMITTED_ON_TIMEOUT=0

resubmit_self() {
  if [[ "$RESUBMITTED_ON_TIMEOUT" -eq 1 ]]; then
    return 0
  fi
  RESUBMITTED_ON_TIMEOUT=1
  echo "Time limit warning received; resubmitting ${BASH_SOURCE[0]}"
  sbatch "${BASH_SOURCE[0]}"
}

trap 'resubmit_self' USR1

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/global/homes/j/jiaqing/.conda/envs/latentvis}"
if [[ -x "${CONDA_ENV_PATH}/bin/python3" ]]; then
  export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
else
  for conda_sh in \
    /global/common/software/nersc/2024/software/conda/py39-anaconda-2021.11/etc/profile.d/conda.sh \
    /global/common/software/nersc/2023/software/conda/py39-anaconda-2021.11/etc/profile.d/conda.sh
  do
    if [[ -f "${conda_sh}" ]]; then
      # shellcheck disable=SC1090
      source "${conda_sh}"
      conda activate "${CONDA_ENV_PATH}"
      break
    fi
  done
fi

export PYTHONUNBUFFERED=1
cd "${BACKEND_DIR}"

echo "Started at $(date -Iseconds)"
echo "python3: $(which python3)"
bash "${SCRIPT_DIR}/run_dimenet_case_to_frontend.sh"
echo "Finished at $(date -Iseconds)"
