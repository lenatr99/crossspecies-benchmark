#!/usr/bin/env bash
# Run PCA embeddings on the host (macOS / Linux).
#
# Examples:
#   bash code/embeddings/pca/run_pca_locally.sh --dataset dicty --variants tier12 --overwrite
#   bash code/embeddings/pca/run_pca_locally.sh --dataset dicty --recipes raw,hvg_scaled --overwrite
#   bash code/embeddings/pca/run_pca_locally.sh --publication-script code/figures/validate_publication_revision.py --help
#
# Environment:
#   PCA_ENV_NAME       Conda env name (default: crossspecies-pca)
#   PCA_LOCAL_UPDATE   Set to 1 to refresh the fully pinned dependency closure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIP_REQ="${SCRIPT_DIR}/pca_pip_requirements.txt"
PIP_LOCK="${SCRIPT_DIR}/pca_pip_lock.txt"

if [[ ! -f "${PIP_REQ}" || ! -f "${PIP_LOCK}" ]]; then
  echo "Missing PCA dependency specification: ${PIP_REQ} or ${PIP_LOCK}" >&2
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

ENV_NAME="${PCA_ENV_NAME:-crossspecies-pca}"
PUBLICATION_SCRIPT=""
if [[ "${1:-}" == "--publication-script" ]]; then
  PUBLICATION_SCRIPT="${2:-}"
  [[ -n "${PUBLICATION_SCRIPT}" ]] || { echo "--publication-script requires a path" >&2; exit 2; }
  shift 2
fi

_env_exists() {
  local want="$1"
  while read -r name; do
    [[ "${name}" == "${want}" ]] && return 0
  done < <("${MAMBA}" env list 2>/dev/null | awk 'NF && $1 !~ /^#/ && $1 !~ /^\$/ {print $1}')
  return 1
}

_pip_install() {
  echo "=== pip install -r pca_pip_lock.txt ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install -r "${PIP_LOCK}"
}

if ! _env_exists "${ENV_NAME}"; then
  echo "=== Creating conda env: ${ENV_NAME} (python 3.11 + pip from conda-forge) ==="
  "${MAMBA}" create -y -n "${ENV_NAME}" -c conda-forge python=3.11 pip
  _pip_install
elif [[ "${PCA_LOCAL_UPDATE:-0}" == "1" ]]; then
  echo "=== Refreshing pip packages in: ${ENV_NAME} ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install --upgrade -r "${PIP_LOCK}"
else
  echo "=== Using existing conda env: ${ENV_NAME} (PCA_LOCAL_UPDATE=1 to refresh pip pins) ==="
fi

cd "${REPO_ROOT}"
if [[ -n "${PUBLICATION_SCRIPT}" ]]; then
  case "${PUBLICATION_SCRIPT}" in
    "${REPO_ROOT}/"*) publication_path="${PUBLICATION_SCRIPT}" ;;
    /*) echo "Publication script must live under ${REPO_ROOT}" >&2; exit 1 ;;
    *) publication_path="${REPO_ROOT}/${PUBLICATION_SCRIPT#./}" ;;
  esac
  [[ -f "${publication_path}" ]] || { echo "Missing publication script: ${publication_path}" >&2; exit 1; }
  exec "${MAMBA}" run -n "${ENV_NAME}" python "${publication_path}" "$@"
fi
exec "${MAMBA}" run -n "${ENV_NAME}" python "${SCRIPT_DIR}/embed.py" "$@"
