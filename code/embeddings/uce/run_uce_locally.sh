#!/usr/bin/env bash
# Run UCE embeddings on the host (macOS / Linux).
#
# The script reuses an existing models/UCE checkout when present and clones it
# only when missing. Runtime dependencies are installed into a dedicated conda
# environment from uce_pip_requirements.txt.
#
# Examples:
#   bash code/embeddings/uce/run_uce_locally.sh --dataset elegans --models 4layer --variants full,tier1 --max-cells 100 --overwrite
#   bash code/embeddings/uce/run_uce_locally.sh --dataset dicty --models 4layer,33layer --variants full,tier1,tier12,tier123
#   bash code/embeddings/uce/run_uce_locally.sh --dataset dicty --models 4layer --variants full,tier1,tier12,tier123 --whitening --overwrite
#   bash code/embeddings/uce/run_uce_locally.sh --dataset dicty --models 4layer --variants full,tier1 --whitening --whitening-tiers tier1 --overwrite
#
# Environment:
#   UCE_ENV_NAME       Conda env name (default: crossspecies-uce)
#   UCE_LOCAL_UPDATE   Set to 1 to refresh pip packages in an existing env.
#   UCE_COMMIT         UCE commit to checkout (default: 8227a65)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UCE_DIR="${UCE_DIR:-${REPO_ROOT}/models/UCE}"
UCE_REPO_URL="${UCE_REPO_URL:-https://github.com/snap-stanford/UCE.git}"
UCE_COMMIT="${UCE_COMMIT:-8227a65}"
PIP_REQ="${SCRIPT_DIR}/uce_pip_requirements.txt"

RUN_WHITENING=0
WHITENING_TIERS="global"
DATASET=""
HAS_PROTEIN_ADAPTER=0
PROTEIN_ADAPTER=""
PROTEIN_ADAPTER_TIERS="global"
PUBLICATION_SCRIPT=""
PASSTHROUGH_ARGS=()

if [[ ! -f "${PIP_REQ}" ]]; then
  echo "Missing pip requirements: ${PIP_REQ}" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --publication-script)
      PUBLICATION_SCRIPT="${2:-}"
      shift 2
      ;;
    --whitening)
      RUN_WHITENING=1
      shift
      ;;
    --whitening-tiers)
      WHITENING_TIERS="${2:-}"
      shift 2
      ;;
    --whitening-tiers=*)
      WHITENING_TIERS="${1#--whitening-tiers=}"
      shift
      ;;
    --dataset)
      DATASET="${2:-}"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --dataset=*)
      DATASET="${1#--dataset=}"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --protein-adapter)
      HAS_PROTEIN_ADAPTER=1
      PROTEIN_ADAPTER="${2:-}"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --protein-adapter=*)
      HAS_PROTEIN_ADAPTER=1
      PROTEIN_ADAPTER="${1#--protein-adapter=}"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --protein-adapter-tiers)
      PROTEIN_ADAPTER_TIERS="${2:-}"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --protein-adapter-tiers=*)
      PROTEIN_ADAPTER_TIERS="${1#--protein-adapter-tiers=}"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

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

ENV_NAME="${UCE_ENV_NAME:-crossspecies-uce}"

_env_exists() {
  local want="$1"
  while read -r name; do
    [[ "${name}" == "${want}" ]] && return 0
  done < <("${MAMBA}" env list 2>/dev/null | awk 'NF && $1 !~ /^#/ && $1 !~ /^\$/ {print $1}')
  return 1
}

_pip_install() {
  echo "=== pip install -r uce_pip_requirements.txt (pinned) ==="
  "${MAMBA}" run -n "${ENV_NAME}" python -m pip install -r "${PIP_REQ}"
}

_ensure_uce_checkout() {
  if [[ ! -d "${UCE_DIR}" ]]; then
    echo "=== Cloning UCE into ${UCE_DIR} ==="
    mkdir -p "$(dirname "${UCE_DIR}")"
    "${MAMBA}" run -n "${ENV_NAME}" git clone "${UCE_REPO_URL}" "${UCE_DIR}"
  else
    echo "=== Using existing UCE checkout: ${UCE_DIR} ==="
  fi
  echo "=== Ensuring UCE checkout: ${UCE_DIR} @ ${UCE_COMMIT} ==="
  "${MAMBA}" run -n "${ENV_NAME}" git -C "${UCE_DIR}" fetch --all
  "${MAMBA}" run -n "${ENV_NAME}" git -C "${UCE_DIR}" checkout "${UCE_COMMIT}"
}

if ! _env_exists "${ENV_NAME}"; then
  echo "=== Creating conda env: ${ENV_NAME} (python 3.11 + pip + git from conda-forge) ==="
  "${MAMBA}" create -y -n "${ENV_NAME}" -c conda-forge python=3.11 pip git
  _pip_install
elif [[ "${UCE_LOCAL_UPDATE:-0}" == "1" ]]; then
  echo "=== Refreshing pip packages in: ${ENV_NAME} ==="
  _pip_install
else
  echo "=== Using existing conda env: ${ENV_NAME} (UCE_LOCAL_UPDATE=1 to refresh pip packages) ==="
fi

if [[ -n "${PUBLICATION_SCRIPT}" ]]; then
  case "${PUBLICATION_SCRIPT}" in
    "${REPO_ROOT}/"*) publication_path="${PUBLICATION_SCRIPT}" ;;
    /*) echo "Publication script must live under ${REPO_ROOT}" >&2; exit 1 ;;
    *) publication_path="${REPO_ROOT}/${PUBLICATION_SCRIPT#./}" ;;
  esac
  if [[ ! -f "${publication_path}" ]]; then
    echo "Missing publication script: ${publication_path}" >&2
    exit 1
  fi
  export PYTHONPATH="${UCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  cd "${REPO_ROOT}"
  if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    exec "${MAMBA}" run -n "${ENV_NAME}" python "${publication_path}" "${PASSTHROUGH_ARGS[@]}"
  fi
  exec "${MAMBA}" run -n "${ENV_NAME}" python "${publication_path}"
fi

_ensure_uce_checkout

export PYTHONPATH="${UCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

if [[ "${RUN_WHITENING}" == "1" ]]; then
  if [[ -z "${DATASET}" ]]; then
    echo "--whitening requires --dataset {dicty,elegans,all}" >&2
    exit 1
  fi

  WHITENING_DATASETS=("${DATASET}")
  if [[ "${DATASET}" == "all" ]]; then
    WHITENING_DATASETS=("dicty" "elegans")
  fi

  for whitening_dataset in "${WHITENING_DATASETS[@]}"; do
    echo "=== Fitting ESM2 whitening adapter: ${whitening_dataset} (${WHITENING_TIERS}) ==="
    "${MAMBA}" run -n "${ENV_NAME}" python "${SCRIPT_DIR}/fit_whitening.py" \
      --dataset "${whitening_dataset}" \
      --tiers "${WHITENING_TIERS}" \
      --uce-dir "${UCE_DIR}"
  done

  if [[ "${HAS_PROTEIN_ADAPTER}" == "0" ]]; then
    PASSTHROUGH_ARGS+=(--protein-adapter esm2_whitening --protein-adapter-tiers "${WHITENING_TIERS}")
  fi
fi

if [[ "${PROTEIN_ADAPTER}" == "esm2_macrogene" ]]; then
  if [[ -z "${DATASET}" ]]; then
    echo "--protein-adapter esm2_macrogene requires --dataset {dicty,elegans,all}" >&2
    exit 1
  fi
  if [[ "${PROTEIN_ADAPTER_TIERS}" != "global" ]]; then
    echo "esm2_macrogene currently supports --protein-adapter-tiers global only" >&2
    exit 1
  fi

  MACROGENE_DATASETS=("${DATASET}")
  if [[ "${DATASET}" == "all" ]]; then
    MACROGENE_DATASETS=("dicty" "elegans")
  fi

  for macrogene_dataset in "${MACROGENE_DATASETS[@]}"; do
    adapter_path="${REPO_ROOT}/results/embeddings/uce/esm2_adapter/${macrogene_dataset}_macrogene_global/${macrogene_dataset}_macrogene_global_gene_symbol_to_embedding_ESM2_adapted.pt"
    if [[ -s "${adapter_path}" ]]; then
      echo "=== Reusing ESM2 macrogene adapter: ${adapter_path} ==="
    else
      echo "=== Fitting ESM2 macrogene adapter: ${macrogene_dataset} (global) ==="
      "${MAMBA}" run -n "${ENV_NAME}" python "${SCRIPT_DIR}/fit_whitening.py" \
        --dataset "${macrogene_dataset}" \
        --method macrogene \
        --tiers global \
        --uce-dir "${UCE_DIR}" \
        --n-macrogenes "${UCE_MACROGENE_N:-32}" \
        --macrogene-beta "${UCE_MACROGENE_BETA:-0.1}" \
        --macrogene-temperature "${UCE_MACROGENE_TEMPERATURE:-0.07}" \
        --macrogene-projection-dim "${UCE_MACROGENE_PROJECTION_DIM:-128}"
    fi
  done
fi

exec "${MAMBA}" run -n "${ENV_NAME}" python "${SCRIPT_DIR}/embed.py" --uce-dir "${UCE_DIR}" "${PASSTHROUGH_ARGS[@]}"
