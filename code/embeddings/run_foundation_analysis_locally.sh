#!/usr/bin/env bash
# Run shared foundation-model postprocessing scripts in a pinned lightweight env.
#
# Conda solver output is pinned by environment.foundation_analysis.conda-lock.yml
# (regenerate after editing environment.foundation_analysis.yml via conda-lock, see README).
# PyPI packages come from foundation_analysis_pip_requirements.txt / _pip_lock.txt.
#
# Examples:
#   bash code/embeddings/run_foundation_analysis_locally.sh \
#     code/embeddings/plot_umaps.py --dataset dicty --model scgpt --overwrite
#
# Environment:
#   FOUNDATION_ANALYSIS_ENV_NAME          Conda env name (default: crossspecies-foundation-analysis)
#   FOUNDATION_ANALYSIS_UPDATE=1          Reinstall pinned pip stack from foundation_analysis_pip_lock.txt before running
#   FOUNDATION_ANALYSIS_SKIP_CONDA_LOCK=1 Create from environment.foundation_analysis.yml only (no conda-lock CLI)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_YAML="${SCRIPT_DIR}/environment.foundation_analysis.yml"
CONDA_LOCK="${SCRIPT_DIR}/environment.foundation_analysis.conda-lock.yml"
PIP_LOCK="${SCRIPT_DIR}/foundation_analysis_pip_lock.txt"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash code/embeddings/run_foundation_analysis_locally.sh SCRIPT.py [args...]" >&2
  exit 2
fi

if [[ ! -f "${ENV_YAML}" ]]; then
  echo "Missing conda env spec: ${ENV_YAML}" >&2
  exit 1
fi
if [[ ! -f "${PIP_LOCK}" ]]; then
  echo "Missing full pip lockfile: ${PIP_LOCK}" >&2
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

conda_lock_cli() {
  if [[ -n "${CONDA_LOCK_EXE:-}" && -x "${CONDA_LOCK_EXE}" ]]; then
    echo "${CONDA_LOCK_EXE}"
    return 0
  fi
  if command -v conda-lock >/dev/null 2>&1; then
    command -v conda-lock
    return 0
  fi
  local bindir
  bindir="$(dirname "${MAMBA}")"
  if [[ -x "${bindir}/conda-lock" ]]; then
    echo "${bindir}/conda-lock"
    return 0
  fi
  return 1
}

_create_conda_env() {
  if [[ "${FOUNDATION_ANALYSIS_SKIP_CONDA_LOCK:-0}" == "1" ]]; then
    echo "=== Creating conda env: ${ENV_NAME} from environment.foundation_analysis.yml (no conda-lock) ==="
    "${MAMBA}" env create -y -n "${ENV_NAME}" -f "${ENV_YAML}"
    return 0
  fi
  if [[ -f "${CONDA_LOCK}" ]] && cl="$(conda_lock_cli)" && [[ -n "${cl}" ]]; then
    echo "=== Creating conda env: ${ENV_NAME} via conda-lock (pinned conda packages) ==="
    "${cl}" install -n "${ENV_NAME}" --conda "${MAMBA}" "${CONDA_LOCK}"
    return 0
  fi
  echo "WARN: conda-lock unavailable or missing ${CONDA_LOCK}; using unlocked ${ENV_YAML}" >&2
  echo "      Install conda-lock (e.g. mamba install -n base conda-lock) for paper-grade conda pinning." >&2
  "${MAMBA}" env create -y -n "${ENV_NAME}" -f "${ENV_YAML}"
}

MAMBA=""
if ! MAMBA="$(resolve_mamba)"; then
  echo "Neither mamba nor micromamba found. Install Miniforge/Mambaforge or micromamba." >&2
  exit 1
fi

ENV_NAME="${FOUNDATION_ANALYSIS_ENV_NAME:-crossspecies-foundation-analysis}"

_env_exists() {
  local want="$1"
  while read -r name; do
    [[ "${name}" == "${want}" ]] && return 0
  done < <("${MAMBA}" env list 2>/dev/null | awk 'NF && $1 !~ /^#/ && $1 !~ /^\$/ {print $1}')
  return 1
}

_pip_install() {
  echo "=== pip install -r foundation_analysis_pip_lock.txt (full pinned closure) ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install -r "${PIP_LOCK}"
}

if ! _env_exists "${ENV_NAME}"; then
  _create_conda_env
  _pip_install
elif [[ "${FOUNDATION_ANALYSIS_UPDATE:-0}" == "1" ]]; then
  echo "=== Refreshing pip packages in: ${ENV_NAME} ==="
  _pip_install
else
  echo "=== Using existing conda env: ${ENV_NAME} (FOUNDATION_ANALYSIS_UPDATE=1 to refresh pip pins) ==="
fi

cd "${REPO_ROOT}"

exec "${MAMBA}" run -n "${ENV_NAME}" python "$@"
