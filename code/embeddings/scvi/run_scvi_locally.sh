#!/usr/bin/env bash
# Run scVI embeddings on the host (macOS / Linux).
#
# Examples:
#   bash code/embeddings/scvi/run_scvi_locally.sh --dataset dicty --variants tier12 --overwrite
#   bash code/embeddings/scvi/run_scvi_locally.sh --dataset dicty --variants full --max-cells 256 --overwrite
#
# Environment:
#   SCVI_ENV_NAME       Conda env name (default: scRNA_scVI)
#   SCVI_LOCAL_UPDATE   Set to 1 to refresh pip packages from scvi_pip_requirements.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIP_REQ="${SCRIPT_DIR}/scvi_pip_requirements.txt"

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

ENV_NAME="${SCVI_ENV_NAME:-scRNA_scVI}"

_env_exists() {
  local want="$1"
  while read -r name; do
    [[ "${name}" == "${want}" ]] && return 0
  done < <("${MAMBA}" env list 2>/dev/null | awk 'NF && $1 !~ /^#/ && $1 !~ /^\$/ {print $1}')
  return 1
}

_pip_install() {
  echo "=== pip install -r scvi_pip_requirements.txt (pinned top-level) ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install -r "${PIP_REQ}"
}

if ! _env_exists "${ENV_NAME}"; then
  echo "=== Creating conda env: ${ENV_NAME} (python 3.11 + pip from conda-forge) ==="
  "${MAMBA}" create -y -n "${ENV_NAME}" -c conda-forge python=3.11 pip
  _pip_install
elif [[ "${SCVI_LOCAL_UPDATE:-0}" == "1" ]]; then
  echo "=== Refreshing pip packages in: ${ENV_NAME} ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install --upgrade -r "${PIP_REQ}"
else
  echo "=== Using existing conda env: ${ENV_NAME} (SCVI_LOCAL_UPDATE=1 to refresh pip pins) ==="
fi

cd "${REPO_ROOT}"
exec "${MAMBA}" run -n "${ENV_NAME}" python "${SCRIPT_DIR}/embed.py" "$@"
