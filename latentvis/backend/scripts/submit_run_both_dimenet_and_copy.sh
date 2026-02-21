#!/bin/bash
# Submit run_both_dimenet_and_copy_to_frontend.sh as a 2h NERSC job. Auto-resubmits if the run fails (exit != 0).
# Uses same embeddings as old results: dimenet_noaug_embeddings_cas_152 + dimenet10conf_embeddings.
#
# Usage: ./submit_run_both_dimenet_and_copy.sh [--output-dir ./pipeline_output] [--time 2:00:00]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

OUTPUT_DIR="./pipeline_output"
JOB_TIME="2:00:00"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --time)
      JOB_TIME="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR_ABS="$BACKEND_DIR/$OUTPUT_DIR"
else
  OUTPUT_DIR_ABS="$OUTPUT_DIR"
fi

JOB_NAME="latentvis_both_dimenet_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKEND_DIR/logs/slurm"
LOG_PREFIX="$BACKEND_DIR/logs/slurm/${JOB_NAME}"

# Only validate when user runs submit (not when job auto-resubmits)
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Pre-submit validation (env + code)..."
  if ! bash "$SCRIPT_DIR/validate_dimenet_job_env.sh"; then
    echo "Validation failed. Fix environment (conda env latentvis, packages: umap-learn, rdkit) and code, then re-run."
    exit 1
  fi
  echo ""
fi

echo "Submitting job: $JOB_NAME (both dimenet_noaug + dimenet10conf pipeline, then copy to frontend)"
echo "  Output dir:  $OUTPUT_DIR_ABS"
echo "  Time limit:  $JOB_TIME (auto-resubmit if not done)"
echo "  Logs:        ${LOG_PREFIX}_<jobid>.out"
echo ""

# Submit via stdin (avoids Slurm not finding script on some nodes)
sbatch 2>&1 << JOBEOF | tee /tmp/sbatch_latentvis_both_dimenet.txt
#!/bin/bash
#SBATCH --job-name=latentvis_both_dimenet
#SBATCH --output=${LOG_PREFIX}_%j.out
#SBATCH --error=${LOG_PREFIX}_%j.err
#SBATCH --time=${JOB_TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -A m636

SCRIPT_DIR="$SCRIPT_DIR"
BACKEND_DIR="$BACKEND_DIR"
OUTPUT_DIR_ABS="$OUTPUT_DIR_ABS"

# Activate conda env with Python 3.9+, umap-learn, rdkit (required on compute nodes)
CONDA_ENV_PATH="\${CONDA_ENV_PATH:-/global/homes/j/jiaqing/.conda/envs/latentvis}"
if [ -f "\${CONDA_ENV_PATH}/bin/python" ]; then
  export PATH="\${CONDA_ENV_PATH}/bin:\${PATH}"
else
  for conda_sh in /global/common/software/nersc/2024/software/conda/py39-anaconda-2021.11/etc/profile.d/conda.sh /global/common/software/nersc/2023/software/conda/py39-anaconda-2021.11/etc/profile.d/conda.sh; do
    if [ -f "\$conda_sh" ]; then source "\$conda_sh"; conda activate "\$CONDA_ENV_PATH"; break; fi
  done
fi

echo "Job started at \$(date -Iseconds)"
echo "Python: \$(which python3 2>/dev/null || which python)"
export PYTHONUNBUFFERED=1
cd "\$BACKEND_DIR"
bash "\$SCRIPT_DIR/run_both_dimenet_and_copy_to_frontend.sh" --output-dir "\$OUTPUT_DIR_ABS"
EXIT=\$?
echo "Job finished at \$(date -Iseconds) with exit code \$EXIT"

if [ \$EXIT -ne 0 ]; then
  echo ""
  echo "Pipeline not complete - resubmitting job (2h)..."
  bash "\$SCRIPT_DIR/submit_run_both_dimenet_and_copy.sh" --output-dir "\$OUTPUT_DIR_ABS" --time "$JOB_TIME" > /dev/null 2>&1
  exit 0
fi
echo "Pipeline job completed successfully"
exit 0
JOBEOF

if grep -q "Submitted batch job" /tmp/sbatch_latentvis_both_dimenet.txt 2>/dev/null; then
  JOB_ID=$(grep "Submitted batch job" /tmp/sbatch_latentvis_both_dimenet.txt | awk '{print $4}')
  echo ""
  echo "Job submitted. Job ID: $JOB_ID"
  echo "  Monitor: squeue -u $USER"
  exit 0
else
  echo "Submission failed."
  exit 1
fi
