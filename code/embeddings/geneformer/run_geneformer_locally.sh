#!/usr/bin/env bash
# Run Geneformer embeddings on the host (macOS / Linux): uses APFS/native disk instead
# of Docker bind mounts, which avoids slow HuggingFace Datasets Arrow writes.
#
# Conda installs python/git/git-lfs from conda-forge; deps are pinned in
# geneformer_pip_requirements.txt. Two-step conda + pip avoids mixed
# ``mamba env create`` YAML + embedded ``pip:`` hooks (macOS failures like
# "exec --: invalid option" in temporary install scripts).
#
# Default --dataset (via embed.py): all → dicty and elegans. Pass --dataset dicty or --dataset elegans for one.
#
# Examples:
#   bash code/embeddings/geneformer/run_geneformer_locally.sh --variants tier12 --overwrite
#   bash code/embeddings/geneformer/run_geneformer_locally.sh --dataset dicty --variants tier12 --overwrite --nproc 4
#
# Environment:
#   GENEFORMER_ENV_NAME       Conda env name (default: crossspecies-geneformer)
#   GENEFORMER_REPO_URL       Geneformer git remote (default: Hugging Face repo)
#   GENEFORMER_COMMIT         Geneformer commit to checkout (default: 18a2ca6)
# If a previous run failed halfway, remove the broken env and rerun:
#   mamba env remove -n crossspecies-geneformer -y
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIP_REQ="${SCRIPT_DIR}/geneformer_pip_requirements.txt"

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

ENV_NAME="${GENEFORMER_ENV_NAME:-crossspecies-geneformer}"
GENEFORMER_REPO_URL="${GENEFORMER_REPO_URL:-https://huggingface.co/ctheodoris/Geneformer}"
GENEFORMER_COMMIT="${GENEFORMER_COMMIT:-18a2ca6}"
GENEFORMER_DIR="${REPO_ROOT}/models/Geneformer"
GENEFORMER_MANAGED_DIR="${GENEFORMER_MANAGED_DIR:-${REPO_ROOT}/models/Geneformer-${GENEFORMER_COMMIT}}"
ACTIVE_GENEFORMER_DIR="${GENEFORMER_DIR}"

_env_exists() {
  local want="$1"
  while read -r name; do
    [[ "${name}" == "${want}" ]] && return 0
  done < <("${MAMBA}" env list 2>/dev/null | awk 'NF && $1 !~ /^#/ && $1 !~ /^\$/ {print $1}')
  return 1
}

_pip_install() {
  echo "=== pip install -r geneformer_pip_requirements.txt (pinned) ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install -r "${PIP_REQ}"
}

_ensure_git_lfs() {
  if "${MAMBA}" run -n "${ENV_NAME}" git lfs version >/dev/null 2>&1; then
    return 0
  fi
  echo "=== Installing git-lfs into conda env: ${ENV_NAME} ==="
  "${MAMBA}" install -y -n "${ENV_NAME}" -c conda-forge git-lfs
}

_clone_if_missing() {
  local target="$1"
  mkdir -p "$(dirname "${target}")"
  if [[ ! -d "${target}" ]]; then
    "${MAMBA}" run -n "${ENV_NAME}" git clone "${GENEFORMER_REPO_URL}" "${target}"
  else
    echo "Geneformer directory already exists; reusing ${target}"
  fi
}

_checkout_geneformer_commit() {
  local target="$1"
  if [[ ! -d "${target}/.git" ]]; then
    echo "Geneformer directory exists but is not a git checkout: ${target}" >&2
    return 1
  fi
  "${MAMBA}" run -n "${ENV_NAME}" git -C "${target}" fetch --all
  "${MAMBA}" run -n "${ENV_NAME}" git -C "${target}" checkout "${GENEFORMER_COMMIT}"
  "${MAMBA}" run -n "${ENV_NAME}" git -C "${target}" lfs pull
}

_ensure_geneformer_checkout() {
  _ensure_git_lfs
  echo "=== Ensuring Geneformer checkout: ${GENEFORMER_DIR} @ ${GENEFORMER_COMMIT} ==="
  "${MAMBA}" run -n "${ENV_NAME}" git lfs install

  _clone_if_missing "${GENEFORMER_DIR}"
  if _checkout_geneformer_commit "${GENEFORMER_DIR}"; then
    ACTIVE_GENEFORMER_DIR="${GENEFORMER_DIR}"
  else
    echo "Primary Geneformer checkout could not switch commits cleanly."
    echo "Using managed pinned checkout instead: ${GENEFORMER_MANAGED_DIR}"
    _clone_if_missing "${GENEFORMER_MANAGED_DIR}"
    _checkout_geneformer_commit "${GENEFORMER_MANAGED_DIR}"
    ACTIVE_GENEFORMER_DIR="${GENEFORMER_MANAGED_DIR}"
  fi

  echo "=== pip install -e models/Geneformer ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install -e "${ACTIVE_GENEFORMER_DIR}"
}

if ! _env_exists "${ENV_NAME}"; then
  echo "=== Creating conda env: ${ENV_NAME} (python 3.11 + pip + git + git-lfs from conda-forge) ==="
  "${MAMBA}" create -y -n "${ENV_NAME}" -c conda-forge python=3.11 pip git git-lfs
  _pip_install
elif [[ "${GENEFORMER_LOCAL_UPDATE:-0}" == "1" ]]; then
  echo "=== Refreshing pip packages in: ${ENV_NAME} ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install --upgrade -r "${PIP_REQ}"
else
  echo "=== Using existing conda env: ${ENV_NAME} (GENEFORMER_LOCAL_UPDATE=1 to refresh pip pins) ==="
fi

_ensure_geneformer_checkout

export PYTHONPATH="${ACTIVE_GENEFORMER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

exec "${MAMBA}" run -n "${ENV_NAME}" python "${SCRIPT_DIR}/embed.py" \
  --geneformer-dir "${ACTIVE_GENEFORMER_DIR}" \
  --model-dir "${ACTIVE_GENEFORMER_DIR}/gf-12L-95M-i4096" \
  "$@"
