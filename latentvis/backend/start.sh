#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-5001}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-2}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-600}"
GUNICORN_GRACEFUL_TIMEOUT="${GUNICORN_GRACEFUL_TIMEOUT:-60}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMBEDDING_DOWNLOAD_DIR="${EMBEDDING_DOWNLOAD_DIR:-${SCRIPT_DIR}/data/embeddings}"

mkdir -p "${EMBEDDING_DOWNLOAD_DIR}"

download_if_needed() {
  local target_path="$1"
  local source_url="$2"
  local label="$3"

  if [[ -z "${source_url}" ]]; then
    return 0
  fi
  if [[ -f "${target_path}" ]]; then
    return 0
  fi
  echo "[start.sh] downloading ${label} embeddings from ${source_url}"
  curl -fL --retry 3 --retry-delay 2 -o "${target_path}" "${source_url}"
}

# Optional download flow for cloud deployment:
# 1) provide *_PT directly, OR
# 2) provide *_PT_URL and let start.sh download once per container boot.
if [[ -z "${LATENTVIS_CHEMPROP_PT:-}" ]]; then
  LATENTVIS_CHEMPROP_PT="${EMBEDDING_DOWNLOAD_DIR}/chemprop-embeddings.pt"
fi
if [[ -z "${LATENTVIS_CHEMELEON_PT:-}" ]]; then
  LATENTVIS_CHEMELEON_PT="${EMBEDDING_DOWNLOAD_DIR}/chemeleon-embeddings.pt"
fi

download_if_needed "${LATENTVIS_CHEMPROP_PT}" "${LATENTVIS_CHEMPROP_PT_URL:-}" "chemprop"
download_if_needed "${LATENTVIS_CHEMELEON_PT}" "${LATENTVIS_CHEMELEON_PT_URL:-}" "chemeleon"

if [[ ! -f "${LATENTVIS_CHEMPROP_PT}" ]]; then
  echo "[start.sh] warning: chemprop embeddings not found at ${LATENTVIS_CHEMPROP_PT}"
fi
if [[ ! -f "${LATENTVIS_CHEMELEON_PT}" ]]; then
  echo "[start.sh] warning: chemeleon embeddings not found at ${LATENTVIS_CHEMELEON_PT}"
fi

export LATENTVIS_CHEMPROP_PT
export LATENTVIS_CHEMELEON_PT

cd "${SCRIPT_DIR}"
exec micromamba run -n base gunicorn \
  -w "${WEB_CONCURRENCY}" \
  --timeout "${GUNICORN_TIMEOUT}" \
  --graceful-timeout "${GUNICORN_GRACEFUL_TIMEOUT}" \
  -b "0.0.0.0:${PORT}" \
  flask_backend:app
