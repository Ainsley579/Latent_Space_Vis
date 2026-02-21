#!/bin/bash
# Submit full pipeline (multi-projection scatter + topk + cluster_sorted_members + MCS) as a
# single-CPU job on NERSC. Pipeline runs run_all_and_copy_to_frontend.sh, then copies to frontend.
#
# Usage (same args as run_all_and_copy_to_frontend.sh, plus optional --time):
#   ./submit_run_all_and_copy_to_frontend.sh --input /path/to/embeddings.pt [--output-dir ./pipeline_output] [--smiles-csv /path/to/smiles.csv] [--model-name NAME] [--time 2:00:00]
#
# Example (embedding directory, e.g. DimeNet layers):
#   ./submit_run_all_and_copy_to_frontend.sh --input /global/cfs/cdirs/m636/nhadler/projects/latent-spaces/some_dir --output-dir ./pipeline_output --smiles-csv ./data/smiles/real_id_smiles.csv --model-name dimenet_noaug
#
# Example (single .pt file):
#   ./submit_run_all_and_copy_to_frontend.sh --input /path/to/chemprop-embeddings.pt --output-dir ./pipeline_output --smiles-csv ./data/smiles/real_id_smiles.csv
#   ./submit_run_all_and_copy_to_frontend.sh --input /path/to/emb.pt --output-dir ./out --case chemprop   # copy to frontend/public/data/chemprop/

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

INPUT=""
OUTPUT_DIR="./pipeline_output"
SMILES_CSV=""
MODEL_NAME=""
JOB_TIME="2:00:00"
CASE_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --smiles-csv)
      SMILES_CSV="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --case)
      CASE_NAME="$2"
      shift 2
      ;;
    --time)
      JOB_TIME="$2"
      shift 2
      ;;
    *)
      if [[ -z "$INPUT" && "$1" != --* ]]; then
        INPUT="$1"
      fi
      shift
      ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "Error: --input is required (path to .pt file or embedding directory)."
  echo "Usage: $0 --input /path/to/embeddings.pt [--output-dir ./pipeline_output] [--smiles-csv PATH] [--model-name NAME] [--time 2:00:00]"
  exit 1
fi

# Resolve to absolute paths for the job (so it works from any cwd on the compute node)
if [[ "$INPUT" == /* ]]; then INPUT_ABS="$INPUT"; else INPUT_ABS="$BACKEND_DIR/$INPUT"; fi
if [[ "$OUTPUT_DIR" == /* ]]; then OUTPUT_DIR_ABS="$OUTPUT_DIR"; else OUTPUT_DIR_ABS="$BACKEND_DIR/$OUTPUT_DIR"; fi
if [[ -n "$SMILES_CSV" ]]; then
  if [[ "$SMILES_CSV" == /* ]]; then SMILES_CSV_ABS="$SMILES_CSV"; else SMILES_CSV_ABS="$BACKEND_DIR/$SMILES_CSV"; fi
else
  SMILES_CSV_ABS=""
fi

JOB_NAME="latentvis_pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKEND_DIR/logs/slurm"
JOB_SCRIPT="$BACKEND_DIR/logs/slurm/${JOB_NAME}.sh"

FULL_CMD="bash \"$SCRIPT_DIR/run_all_and_copy_to_frontend.sh\" --input \"$INPUT_ABS\" --output-dir \"$OUTPUT_DIR_ABS\""
[[ -n "$SMILES_CSV_ABS" ]] && FULL_CMD="$FULL_CMD --smiles-csv \"$SMILES_CSV_ABS\""
[[ -n "$MODEL_NAME" ]] && FULL_CMD="$FULL_CMD --model-name \"$MODEL_NAME\""
[[ -n "$CASE_NAME" ]] && FULL_CMD="$FULL_CMD --case \"$CASE_NAME\""

cat > "$JOB_SCRIPT" << JOBEOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${BACKEND_DIR}/logs/slurm/${JOB_NAME}_%j.out
#SBATCH --error=${BACKEND_DIR}/logs/slurm/${JOB_NAME}_%j.err
#SBATCH --time=${JOB_TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -A m636

echo "Job started at \$(date -Iseconds)"
echo "Input: $INPUT_ABS"
echo "Output dir: $OUTPUT_DIR_ABS"

export PYTHONUNBUFFERED=1
cd "$BACKEND_DIR"
$FULL_CMD
EXIT=\$?
echo "Job finished at \$(date -Iseconds) with exit code \$EXIT"
exit \$EXIT
JOBEOF

chmod +x "$JOB_SCRIPT"

echo "Submitting job: $JOB_NAME"
echo "  Input:       $INPUT_ABS"
echo "  Output dir:  $OUTPUT_DIR_ABS"
echo "  Time limit:  $JOB_TIME"
echo "  Logs:        $BACKEND_DIR/logs/slurm/${JOB_NAME}_<jobid>.out"
echo ""

sbatch "$JOB_SCRIPT" 2>&1 | tee /tmp/sbatch_latentvis_pipeline.txt

if grep -q "Submitted batch job" /tmp/sbatch_latentvis_pipeline.txt 2>/dev/null; then
  JOB_ID=$(grep "Submitted batch job" /tmp/sbatch_latentvis_pipeline.txt | awk '{print $4}')
  echo ""
  echo "Job submitted successfully. Job ID: $JOB_ID"
  echo "  Monitor: squeue -u \$USER"
  echo "  Log:     tail -f $BACKEND_DIR/logs/slurm/${JOB_NAME}_${JOB_ID}.out"
  exit 0
else
  echo "Submission failed."
  exit 1
fi
