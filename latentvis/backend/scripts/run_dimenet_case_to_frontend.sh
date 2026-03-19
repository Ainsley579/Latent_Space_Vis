#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DATA_ROOT="${BACKEND_DIR}/data"
FRONTEND_DATA_ROOT="${BACKEND_DIR}/../frontend/public/data"
BACKEND_CASE_ROOT="${BACKEND_DATA_ROOT}/dimenet"
FRONTEND_CASE_ROOT="${FRONTEND_DATA_ROOT}/dimenet"
TARGET_BACKEND_ROOT="${BACKEND_CASE_ROOT}/test"
TARGET_FRONTEND_ROOT="${FRONTEND_CASE_ROOT}/test"
WORK_ROOT="${BACKEND_DIR}/pipeline_output_dimenet_case"
SMILES_JSON_PATH="${WORK_ROOT}/real_id_to_smiles.json"
SMILES_CSV_PATH="${WORK_ROOT}/real_id_smiles.csv"
DIMENET_KNN_K="${DIMENET_KNN_K:-2}"
COMMON_LAYER_ORDER="fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,ffn_layer_1,ffn_layer_2"
CHEMELEON_LAYER_ORDER="fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,gnn_layer_4,gnn_layer_5,gnn_layer_6,ffn_layer_1,ffn_layer_2"

EMB_ROOT="/global/cfs/cdirs/m636/nhadler/projects/latent-spaces"
INPUT_NOAUG="${DIMENET_NOAUG_INPUT:-${EMB_ROOT}/dimenet_noaug_embeddings_cas_152}"
INPUT_10CONF="${DIMENET_10CONF_INPUT:-${EMB_ROOT}/dimenet10conf_embeddings}"
INPUT_100CONF="${DIMENET_100CONF_INPUT:-}"

LAYER_MAP_NOAUG="${SCRIPT_DIR}/layer_map_dimenet_noaug_cas.json"
LAYER_MAP_10CONF="${SCRIPT_DIR}/layer_map_dimenet10conf_cas.json"
LAYER_MAP_100CONF="${DIMENET_100CONF_LAYER_MAP:-}"

DIMENET_SPLIT_JSON="${DIMENET_SPLIT_JSON:-/global/cfs/cdirs/m636/nhadler/projects/Libra-ML-Modeling/model-training/butene/dimenet-noaug/data/jsons/split_0.json}"
HF_LITERATURE_TSV="${HF_LITERATURE_TSV:-/global/cfs/cdirs/m636/nhadler/projects/Libra-ML-Modeling/data/HF-Literature.tsv}"

cd "$BACKEND_DIR"

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

PIPELINE_ARGS=(
  --input "$INPUT_NOAUG"
  --model-name dimenet_noaug
  --layer-file-map "$LAYER_MAP_NOAUG"
  --input2 "$INPUT_10CONF"
  --model-name2 dimenet10conf
  --layer-file-map2 "$LAYER_MAP_10CONF"
  --output-dir "$WORK_ROOT"
  --clustering-method all
  --knn-k "$DIMENET_KNN_K"
  --default-reduction mds
  --smiles-csv "$SMILES_CSV_PATH"
)

if [[ -n "$INPUT_100CONF" ]]; then
  if [[ ! -d "$INPUT_100CONF" ]]; then
    echo "DIMENET_100CONF_INPUT was set but not found: $INPUT_100CONF" >&2
    exit 1
  fi
  if [[ -z "$LAYER_MAP_100CONF" || ! -f "$LAYER_MAP_100CONF" ]]; then
    echo "DIMENET_100CONF_INPUT requires DIMENET_100CONF_LAYER_MAP." >&2
    exit 1
  fi
  PIPELINE_ARGS+=(
    --input3 "$INPUT_100CONF"
    --model-name3 dimenet100conf
    --layer-file-map3 "$LAYER_MAP_100CONF"
  )
fi

rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT" "$TARGET_BACKEND_ROOT" "$TARGET_FRONTEND_ROOT"

python3 "${SCRIPT_DIR}/build_dimenet_real_id_to_smiles.py" \
  --split-json "$DIMENET_SPLIT_JSON" \
  --literature-tsv "$HF_LITERATURE_TSV" \
  --output "$SMILES_JSON_PATH" \
  --csv-output "$SMILES_CSV_PATH"

python3 "${SCRIPT_DIR}/build_dimenet_id_smiles_y_true_y_pred.py"

echo "Running isolated DimeNet backend pipeline..."
python3 run_pipeline_from_embeddings.py "${PIPELINE_ARGS[@]}"

for target_root in "$TARGET_BACKEND_ROOT" "$TARGET_FRONTEND_ROOT"; do
  for subdir in sankey scatterplot bic_aic topk substructures clustering_metrics clustering; do
    if [[ -e "${target_root}/${subdir}" ]]; then
      rm -rf "${target_root:?}/${subdir}"
    fi
    if [[ -e "${WORK_ROOT}/${subdir}" ]]; then
      cp -r "${WORK_ROOT}/${subdir}" "${target_root}/${subdir}"
    fi
  done
  cp "$SMILES_JSON_PATH" "${target_root}/real_id_to_smiles.json"
done

python3 "${SCRIPT_DIR}/materialize_sankey_compat_files.py" \
  --dataset-root "$TARGET_BACKEND_ROOT" \
  --vertical-mode adjacent

for method in hdbscan agglomerative knn; do
  if [[ -d "${TARGET_BACKEND_ROOT}/clustering/${method}" ]]; then
    python3 "${SCRIPT_DIR}/materialize_sankey_compat_files.py" \
      --dataset-root "${TARGET_BACKEND_ROOT}/clustering/${method}" \
      --vertical-mode adjacent
  fi
done

for target_root in "$TARGET_FRONTEND_ROOT"; do
  python3 "${SCRIPT_DIR}/materialize_sankey_compat_files.py" \
    --dataset-root "$target_root" \
    --vertical-mode adjacent
  for method in hdbscan agglomerative knn; do
    if [[ -d "${target_root}/clustering/${method}" ]]; then
      python3 "${SCRIPT_DIR}/materialize_sankey_compat_files.py" \
        --dataset-root "${target_root}/clustering/${method}" \
        --vertical-mode adjacent
    fi
  done
done

for data_root in "$BACKEND_DATA_ROOT" "$FRONTEND_DATA_ROOT"; do
  python3 "${SCRIPT_DIR}/compute_layer_transition_stats.py" \
    --data-root "$data_root" \
    --cases "chemprop,chemeleon,dimenet" \
    --layer-order "$COMMON_LAYER_ORDER" \
    --layer-order-chemeleon "$CHEMELEON_LAYER_ORDER"
done

echo "DimeNet case refreshed at ${TARGET_BACKEND_ROOT} and mirrored to ${TARGET_FRONTEND_ROOT}"
