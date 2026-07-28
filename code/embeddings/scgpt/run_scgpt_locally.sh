#!/usr/bin/env bash
# Run scGPT embeddings on the host (macOS / Linux).
#
# Examples:
#   bash code/embeddings/scgpt/run_scgpt_locally.sh --dataset dicty --variants tier12 --overwrite
#   bash code/embeddings/scgpt/run_scgpt_locally.sh --dataset dicty --variants full,tier12 --max-cells 256
#
# Environment:
#   SCGPT_ENV_NAME       Conda env name (default: crossspecies-scgpt)
#   SCGPT_LOCAL_UPDATE   Set to 1 to refresh pip packages from scgpt_pip_requirements.txt
#   SCGPT_DOWNLOAD_MODEL Set to 0 to require an already-populated models/scGPT/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIP_REQ="${SCRIPT_DIR}/scgpt_pip_requirements.txt"

if [[ ! -f "${PIP_REQ}" ]]; then
  echo "Missing pip requirements: ${PIP_REQ}" >&2
  exit 1
fi

resolve_mamba() {
  if [[ -n "${MAMBA_EXE:-}" ]]; then
    echo "${MAMBA_EXE}"
    return 0
  fi
  if command -v mamba >/dev/null 2>&1; then
    command -v mamba
    return 0
  fi
  if command -v micromamba >/dev/null 2>&1; then
    command -v micromamba
    return 0
  fi
  return 1
}

MAMBA=""
if ! MAMBA="$(resolve_mamba)"; then
  echo "Neither mamba nor micromamba found. Install Miniforge/Mambaforge or micromamba." >&2
  exit 1
fi

ENV_NAME="${SCGPT_ENV_NAME:-crossspecies-scgpt}"

_env_exists() {
  local want="$1"
  while read -r name; do
    [[ "${name}" == "${want}" ]] && return 0
  done < <("${MAMBA}" env list 2>/dev/null | awk 'NF && $1 !~ /^#/ && $1 !~ /^\$/ {print $1}')
  return 1
}

_pip_install() {
  echo "=== pip install -r scgpt_pip_requirements.txt (pinned top-level) ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install -r "${PIP_REQ}"
}

if ! _env_exists "${ENV_NAME}"; then
  echo "=== Creating conda env: ${ENV_NAME} (python 3.11 + pip from conda-forge) ==="
  "${MAMBA}" create -y -n "${ENV_NAME}" -c conda-forge python=3.11 pip
  _pip_install
elif [[ "${SCGPT_LOCAL_UPDATE:-0}" == "1" ]]; then
  echo "=== Refreshing pip packages in: ${ENV_NAME} ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install --upgrade -r "${PIP_REQ}"
else
  echo "=== Using existing conda env: ${ENV_NAME} (SCGPT_LOCAL_UPDATE=1 to refresh pip pins) ==="
fi

cd "${REPO_ROOT}"
PY_ARGS=("$@")
if [[ "${SCGPT_DOWNLOAD_MODEL:-1}" == "1" ]]; then
  has_download_flag=0
  for arg in "${PY_ARGS[@]}"; do
    if [[ "${arg}" == "--download-model" ]]; then
      has_download_flag=1
      break
    fi
  done
  if [[ "${has_download_flag}" == "0" ]]; then
    PY_ARGS+=("--download-model")
  fi
fi

exec "${MAMBA}" run -n "${ENV_NAME}" python "${SCRIPT_DIR}/embed.py" "${PY_ARGS[@]}"
