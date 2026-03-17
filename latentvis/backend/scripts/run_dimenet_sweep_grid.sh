#!/usr/bin/env bash

set -euo pipefail

METHOD=""
SPLIT="${DIMENET_SPLIT:-test}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      METHOD="${2:-}"
      shift 2
      ;;
    --split)
      SPLIT="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

case "$METHOD" in
  pgmeans|agglomerative|knn|hdbscan) ;;
  *) echo "Invalid or missing --method"; exit 2 ;;
esac
case "$SPLIT" in
  test) ;;
  *) echo "Invalid or unsupported --split: $SPLIT"; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${LATENTVIS_DATA_ROOT:-$BACKEND_DIR/data}"
CASE_ROOT="$DATA_ROOT/dimenet/$SPLIT"
OUT_ROOT="$CASE_ROOT/sweeps/$METHOD"
WORK_ROOT="${BACKEND_DIR}/pipeline_output_dimenet_sweeps"
SMILES_JSON_PATH="${WORK_ROOT}/real_id_to_smiles.json"
SMILES_CSV_PATH="${WORK_ROOT}/real_id_smiles.csv"

EMB_ROOT="/global/cfs/cdirs/m636/nhadler/projects/latent-spaces"
INPUT_NOAUG="${DIMENET_NOAUG_INPUT:-${EMB_ROOT}/dimenet_noaug_embeddings_cas_152}"
INPUT_10CONF="${DIMENET_10CONF_INPUT:-${EMB_ROOT}/dimenet10conf_embeddings}"
LAYER_MAP_NOAUG="${SCRIPT_DIR}/layer_map_dimenet_noaug_cas.json"
LAYER_MAP_10CONF="${SCRIPT_DIR}/layer_map_dimenet10conf_cas.json"
DIMENET_SPLIT_JSON="${DIMENET_SPLIT_JSON:-/global/cfs/cdirs/m636/nhadler/projects/Libra-ML-Modeling/model-training/butene/dimenet-noaug/data/jsons/split_0.json}"
HF_LITERATURE_TSV="${HF_LITERATURE_TSV:-/global/cfs/cdirs/m636/nhadler/projects/Libra-ML-Modeling/data/HF-Literature.tsv}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/global/homes/j/jiaqing/.conda/envs/latentvis}"

AGG_LINKAGES_RAW="${AGG_LINKAGES:-ward,complete,average}"
KNN_K_MIN="${KNN_K_MIN:-2}"
KNN_K_MAX="${KNN_K_MAX:-10}"
FORCE_RERUN="${FORCE_RERUN:-0}"

if [[ ! -d "$INPUT_NOAUG" ]]; then
  echo "Missing DimeNet noaug embedding dir: $INPUT_NOAUG" >&2
  exit 1
fi
if [[ ! -d "$INPUT_10CONF" ]]; then
  echo "Missing DimeNet 10conf embedding dir: $INPUT_10CONF" >&2
  exit 1
fi
if [[ ! -f "$LAYER_MAP_NOAUG" || ! -f "$LAYER_MAP_10CONF" ]]; then
  echo "Missing required DimeNet layer map JSON files." >&2
  exit 1
fi

if [[ -x "${CONDA_ENV_PATH}/bin/python3" ]]; then
  export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
fi

mkdir -p "$OUT_ROOT" "$WORK_ROOT"
cd "$BACKEND_DIR"

python3 "${SCRIPT_DIR}/build_dimenet_real_id_to_smiles.py" \
  --split-json "$DIMENET_SPLIT_JSON" \
  --literature-tsv "$HF_LITERATURE_TSV" \
  --output "$SMILES_JSON_PATH" \
  --csv-output "$SMILES_CSV_PATH"

is_complete_dir() {
  local d="$1"
  [[ -f "$d/sankey/all_sankey.json" ]] && \
  [[ -f "$d/sankey/sankey_samples.json" ]] && \
  [[ -f "$d/clustering_metrics/cluster_assignments_with_real_id.csv" ]] && \
  [[ -f "$d/topk/cluster_representatives.json" ]] && \
  [[ -f "$d/substructures/cluster_substructures.json" ]] && \
  [[ -d "$d/scatterplot" ]]
}

write_meta() {
  local d="$1"
  local params_json="$2"
  cat > "$d/run_meta.json" <<EOF
{
  "dataset": "dimenet",
  "split": "$SPLIT",
  "method": "$METHOD",
  "params": $params_json
}
EOF
}

run_one() {
  local tag="$1"
  local params_json="$2"
  shift 2
  local out_dir="$OUT_ROOT/$tag"

  if [[ "$FORCE_RERUN" != "1" ]] && is_complete_dir "$out_dir"; then
    echo "[skip] complete: $out_dir"
    return 0
  fi

  if [[ "$FORCE_RERUN" == "1" ]] && [[ -d "$out_dir" ]]; then
    echo "[overwrite] removing existing output: $out_dir"
    rm -rf "$out_dir"
  fi

  mkdir -p "$out_dir"
  echo "[run] dimenet/$SPLIT/$METHOD/$tag"
  set +e
  python3 run_pipeline_from_embeddings.py \
    --input "$INPUT_NOAUG" \
    --model-name dimenet_noaug \
    --layer-file-map "$LAYER_MAP_NOAUG" \
    --input2 "$INPUT_10CONF" \
    --model-name2 dimenet10conf \
    --layer-file-map2 "$LAYER_MAP_10CONF" \
    --output-dir "$out_dir" \
    --default-reduction mds \
    --smiles-csv "$SMILES_CSV_PATH" \
    "$@"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[error] pipeline failed (rc=$rc): $out_dir"
    return $rc
  fi
  if ! is_complete_dir "$out_dir"; then
    echo "[error] incomplete outputs: $out_dir"
    return 1
  fi

  cp "$SMILES_JSON_PATH" "$out_dir/real_id_to_smiles.json"
  python3 "${SCRIPT_DIR}/materialize_sankey_compat_files.py" \
    --dataset-root "$out_dir" \
    --vertical-mode adjacent
  write_meta "$out_dir" "$params_json"
  return 0
}

echo "=== DimeNet sweep start $(date -Iseconds) ==="
echo "split=$SPLIT method=$METHOD out_root=$OUT_ROOT"

case "$METHOD" in
  pgmeans)
    for kmin in {2..10}; do
      for kmax in $(seq "$kmin" 10); do
        tag="kmin_${kmin}_kmax_${kmax}"
        params="{\"k_min\": $kmin, \"k_max\": $kmax}"
        run_one "$tag" "$params" \
          --clustering-method pgmeans \
          --k-min "$kmin" \
          --k-max "$kmax" || exit 1
      done
    done
    ;;
  agglomerative)
    IFS=',' read -r -a _raw_links <<< "$AGG_LINKAGES_RAW"
    LINKS=()
    for lk in "${_raw_links[@]}"; do
      ll="$(echo "$lk" | tr '[:upper:]' '[:lower:]' | xargs)"
      case "$ll" in
        ward|complete|average)
          LINKS+=("$ll")
          ;;
      esac
    done
    if [[ ${#LINKS[@]} -eq 0 ]]; then
      LINKS=("ward" "complete" "average")
    fi
    for linkage in "${LINKS[@]}"; do
      for k in {2..10}; do
        tag="k_${k}_linkage_${linkage}"
        params="{\"k\": $k, \"linkage\": \"${linkage}\"}"
        run_one "$tag" "$params" \
          --clustering-method agglomerative \
          --agg-k "$k" \
          --agg-linkage "$linkage" \
          --k-min 2 \
          --k-max 10 || exit 1
      done
    done
    ;;
  knn)
    for k in $(seq "$KNN_K_MIN" "$KNN_K_MAX"); do
      tag="k_${k}"
      params="{\"k\": $k}"
      run_one "$tag" "$params" \
        --clustering-method knn \
        --knn-k "$k" \
        --k-min 2 \
        --k-max 10 || exit 1
    done
    ;;
  hdbscan)
    tag="default"
    params="{}"
    run_one "$tag" "$params" \
      --clustering-method hdbscan \
      --k-min 2 \
      --k-max 10 || exit 1
    ;;
esac

export LATENTVIS_SWEEP_OUT_ROOT="$OUT_ROOT"
python3 - <<'PY'
import json
import os
from pathlib import Path

out_root = Path(os.environ["LATENTVIS_SWEEP_OUT_ROOT"])
rows = []
for d in sorted([p for p in out_root.iterdir() if p.is_dir()]):
    m = d / "run_meta.json"
    if not m.is_file():
        continue
    try:
        meta = json.loads(m.read_text(encoding="utf-8"))
    except Exception:
        continue
    meta["path"] = str(d)
    rows.append(meta)
(out_root / "manifest.json").write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
print("Wrote manifest:", out_root / "manifest.json", "runs=", len(rows))
PY

echo "=== DimeNet sweep complete $(date -Iseconds) ==="
