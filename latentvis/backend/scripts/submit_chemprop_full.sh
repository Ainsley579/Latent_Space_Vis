#!/bin/bash
# Submit chemprop full pipeline job. Runs validation first to confirm code works.
#
# Usage:
#   ./submit_chemprop_full.sh              # validate then submit
#   ./submit_chemprop_full.sh --skip-validate   # submit without validation

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/slurm_fallback_lib.sh"
cd "$BACKEND_DIR"

SKIP_VALIDATE=false
[[ "${1:-}" == "--skip-validate" ]] && SKIP_VALIDATE=true

if [[ "$SKIP_VALIDATE" != true ]]; then
  echo "=== 1/2 Running validation (tiny + short chemprop) ==="
  if ! "$SCRIPT_DIR/validate_chemprop_pipeline.sh"; then
    echo "Validation failed. Fix code before submitting. Use --skip-validate to bypass."
    exit 1
  fi
  echo ""
  echo "Validation passed."
  echo ""
fi

echo "=== Submitting SLURM job (auto shared fallback enabled) ==="
mkdir -p "$BACKEND_DIR/logs/slurm"
sbatch_args=()
script_args=()
latentvis_submit_with_fallback \
  "$SCRIPT_DIR/submit_chemprop_full.slurm" \
  "$SCRIPT_DIR/submit_chemprop_full_shared.slurm" \
  "$SCRIPT_DIR/submit_chemprop_full_shared_overrun.slurm" \
  sbatch_args \
  script_args
