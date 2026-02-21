#!/bin/bash
# Copy pipeline output into frontend public data, optionally under a case-study subfolder.
#
# Usage:
#   ./copy_to_frontend.sh <pipeline_output_dir> [case_name]
#   ./copy_to_frontend.sh <dir1> [case1] [dir2] [case2] ...
#
# case_name: one of dimenet, chemprop, chemeleon. If given, copies to frontend/public/data/<case_name>/.
# If omitted, copies to frontend/public/data/ (root) for backward compatibility.
#
# Subdirs copied: sankey, scatterplot, bic_aic, topk, substructures, clustering_metrics, clustering

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DATA="${BACKEND_DIR}/../frontend/public/data"
mkdir -p "$FRONTEND_DATA"

SUBDIRS="sankey scatterplot bic_aic topk substructures clustering_metrics clustering"

copy_one() {
  local SRC="$1"
  local CASE_NAME="${2:-}"
  local DST
  if [[ -n "$CASE_NAME" ]]; then
    DST="${FRONTEND_DATA}/${CASE_NAME}"
    mkdir -p "$DST"
  else
    DST="$FRONTEND_DATA"
  fi
  if [[ ! -d "$SRC" ]]; then
    echo "Skip (not found): $SRC"
    return 0
  fi
  echo "Copying $SRC -> $DST"
  for sub in $SUBDIRS; do
    if [[ -d "$SRC/$sub" ]]; then
      rm -rf "$DST/$sub"
      cp -r "$SRC/$sub" "$DST/$sub"
      echo "  $sub"
    fi
  done
}

# Parse args: (dir [case]) (dir [case]) ... with case one of dimenet|chemprop|chemeleon
if [[ $# -eq 0 ]]; then
  # Legacy: no args -> copy pipeline_output_chemprop and pipeline_output_chemeleon to root
  for SRC in pipeline_output_chemprop pipeline_output_chemeleon; do
    SRC="$(cd "$BACKEND_DIR" 2>/dev/null && readlink -f "$SRC" 2>/dev/null)" || SRC="$BACKEND_DIR/$SRC"
    copy_one "$SRC" ""
  done
else
  while [[ $# -gt 0 ]]; do
    DIR="$1"
    shift
    DIR="$(cd "$BACKEND_DIR" 2>/dev/null && readlink -f "$DIR" 2>/dev/null)" || DIR="$BACKEND_DIR/$DIR"
    CASE=""
    if [[ $# -gt 0 && ( "$1" == "dimenet" || "$1" == "chemprop" || "$1" == "chemeleon" ) ]]; then
      CASE="$1"
      shift
    fi
    copy_one "$DIR" "$CASE"
  done
fi

echo "Done. Frontend data: $FRONTEND_DATA"
