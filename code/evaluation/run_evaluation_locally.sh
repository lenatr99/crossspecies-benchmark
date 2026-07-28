#!/usr/bin/env bash
# Run evaluation workflows in their own reproducible conda environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_YAML="${SCRIPT_DIR}/environment.yml"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
PIP_LOCK="${SCRIPT_DIR}/requirements.lock.txt"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash code/evaluation/run_evaluation_locally.sh SCRIPT.py [args...]" >&2
  echo "Example: bash code/evaluation/run_evaluation_locally.sh code/evaluation/run_marker_presets.py --dataset dicty" >&2
  echo "Example: bash code/evaluation/run_evaluation_locally.sh code/evaluation/compute_avgbio.py --dataset dicty" >&2
  exit 2
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

ENV_NAME="${CROSSSPECIES_EVALUATION_ENV_NAME:-crossspecies-evaluation}"

env_exists() {
  local want="$1"
  while read -r name; do
    [[ "${name}" == "${want}" ]] && return 0
  done < <("${MAMBA}" env list 2>/dev/null | awk 'NF && $1 !~ /^#/ && $1 !~ /^\$/ {print $1}')
  return 1
}

if ! env_exists "${ENV_NAME}"; then
  echo "=== Creating conda env: ${ENV_NAME} from code/evaluation/environment.yml ==="
  "${MAMBA}" env create -y -n "${ENV_NAME}" -f "${ENV_YAML}"
  echo "=== Installing fully pinned Python packages from code/evaluation/requirements.lock.txt ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install -r "${PIP_LOCK}"
elif [[ "${CROSSSPECIES_EVALUATION_UPDATE:-0}" == "1" ]]; then
  echo "=== Refreshing Python packages in: ${ENV_NAME} ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install -r "${PIP_LOCK}"
else
  echo "=== Using existing conda env: ${ENV_NAME} (CROSSSPECIES_EVALUATION_UPDATE=1 to refresh pins) ==="
fi

cd "${REPO_ROOT}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "${MAMBA}" run -n "${ENV_NAME}" python "$@"
