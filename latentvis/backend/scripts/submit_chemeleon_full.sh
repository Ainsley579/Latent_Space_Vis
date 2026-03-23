#!/bin/bash
# Submit chemeleon full pipeline job.
# Usage: ./submit_chemeleon_full.sh [--skip-validate]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/slurm_fallback_lib.sh"
cd "$BACKEND_DIR"

SKIP_VALIDATE=false
[[ "${1:-}" == "--skip-validate" ]] && SKIP_VALIDATE=true

if [[ "$SKIP_VALIDATE" != true ]]; then
  echo "=== Running chemeleon validation ==="
  if ! "$SCRIPT_DIR/validate_chemeleon_pipeline.sh"; then
    echo "Validation failed. Use --skip-validate to bypass."
    exit 1
  fi
  echo ""
fi

echo "=== Submitting chemeleon SLURM job (auto shared fallback enabled) ==="
mkdir -p "$BACKEND_DIR/logs/slurm"
sbatch_args=()
script_args=()
latentvis_submit_with_fallback \
  "$SCRIPT_DIR/submit_chemeleon_full.slurm" \
  "$SCRIPT_DIR/submit_chemeleon_full_shared.slurm" \
  "$SCRIPT_DIR/submit_chemeleon_full_shared_overrun.slurm" \
  sbatch_args \
  script_args
