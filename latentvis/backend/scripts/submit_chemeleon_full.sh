#!/bin/bash
# Submit chemeleon full pipeline job.
# Usage: ./submit_chemeleon_full.sh [--skip-validate]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
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

echo "=== Submitting chemeleon SLURM job (2hr, auto-resubmit) ==="
mkdir -p "$BACKEND_DIR/logs/slurm"
sbatch "$SCRIPT_DIR/submit_chemeleon_full.slurm"
