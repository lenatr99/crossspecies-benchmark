#!/usr/bin/env bash
# Build and run the ortholog consensus pipeline in the reproducible Docker image.
#
# Usage:
#   bash code/orthologs/orthologs_run_all_docker.sh [pipeline args]
#
# Examples:
#   bash code/orthologs/orthologs_run_all_docker.sh
#   bash code/orthologs/orthologs_run_all_docker.sh --threads 12 --skip-oma
#
# Environment overrides:
#   ORTHOLOGS_DOCKER_IMAGE=crossspecies-orthologs:2026-04-28
#   ORTHOLOGS_DOCKER_PLATFORM=linux/amd64
#   ORTHOLOGS_DOCKER_BUILD=0      # skip docker build, use existing image
#   ORTHOLOGS_DOCKER_CPUS=16      # number of processors Docker may use (also defaults
#                                   OMA dicty-human xref parallelism unless overridden)
#   ORTHOLOGS_DOCKER_BOOTSTRAP_STATIC=0
#                                 # skip Docker preflight for large/static input files
#   ORTHOLOGS_OMA_WORKERS=16     # optional: override OMA xref thread count (01d only)
#   ORTHOFINDER_THREADS=16        # number of OrthoFinder threads inside Docker

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${ORTHOLOGS_DOCKER_IMAGE:-crossspecies-orthologs:2026-04-28}"
PLATFORM="${ORTHOLOGS_DOCKER_PLATFORM:-linux/amd64}"
BUILD="${ORTHOLOGS_DOCKER_BUILD:-1}"
DOCKER_CPUS="${ORTHOLOGS_DOCKER_CPUS:-16}"
ORTHOFINDER_THREADS_VALUE="${ORTHOFINDER_THREADS:-16}"
BOOTSTRAP_STATIC="${ORTHOLOGS_DOCKER_BOOTSTRAP_STATIC:-${ORTHOLOGS_DOCKER_DOWNLOAD_XREFS:-1}}"
HELP_ARG=0

for arg in "$@"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    HELP_ARG=1
  fi
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for the reproducible ortholog pipeline." >&2
  exit 1
fi

if [[ "${BUILD}" != "0" ]]; then
  echo "=== Building Docker image: ${IMAGE} (${PLATFORM}) ==="
  docker build \
    --platform "${PLATFORM}" \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --tag "${IMAGE}" \
    "${SCRIPT_DIR}"
fi

IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"

mkdir -p "${REPO_ROOT}/results/orthologs"

cat > "${REPO_ROOT}/results/orthologs/docker_runtime_manifest.json" <<EOF
{
  "image": "${IMAGE}",
  "image_id": "${IMAGE_ID}",
  "platform": "${PLATFORM}",
  "dockerfile": "code/orthologs/Dockerfile",
  "environment": "code/orthologs/environment.docker.yml",
  "environment_lock": "code/orthologs/environment.docker.lock.yml",
  "docker_cpus": "${DOCKER_CPUS}",
  "orthofinder_threads": "${ORTHOFINDER_THREADS_VALUE}"
}
EOF

DOCKER_RUN=(docker run --rm --cpus="${DOCKER_CPUS}")

if [[ -t 0 && -t 1 ]]; then
  DOCKER_RUN+=(-it)
fi

if [[ "${BOOTSTRAP_STATIC}" != "0" && "${HELP_ARG}" -eq 0 ]]; then
  mkdir -p "${REPO_ROOT}/data"
  echo "=== Docker preflight: static ortholog inputs ==="
  "${DOCKER_RUN[@]}" \
    --platform "${PLATFORM}" \
    --workdir /workspace/code/orthologs \
    --volume "${REPO_ROOT}:/workspace" \
    --env PYTHONPATH=/workspace/code/orthologs \
    "${IMAGE}" \
    bash -lc '
      set -euo pipefail
      current=/workspace/data/odb12v2_gene_xrefs.tab.gz
      legacy=/workspace/data/odb12v1_gene_xrefs.tab.gz
      url=https://data.orthodb.org/current/download/odb_data_dump/odb12v2_gene_xrefs.tab.gz

      if [[ -s "${current}" || -s "${legacy}" ]]; then
        echo "OrthoDB gene xref dump already cached."
      else
        echo "Downloading OrthoDB gene xref dump (~4.7 GiB) to ${current}"
        tmp="${current}.part"
        if command -v aria2c >/dev/null 2>&1; then
          aria2c --continue=true --max-connection-per-server=4 --split=4 \
            --dir "$(dirname "${current}")" --out "$(basename "${tmp}")" "${url}"
        else
          curl -fL --retry 5 --retry-delay 10 --continue-at - "${url}" -o "${tmp}"
        fi
        mv "${tmp}" "${current}"
      fi

      /opt/conda/bin/python - <<'"'"'PY'"'"'
from _utils import ensure_human_gene_info, ensure_human_ncbi_annotations
from config import (
    HUMAN_GENE_INFO,
    HUMAN_GENE_INFO_URL,
    HUMAN_NCBI_ANNOTATIONS,
    ORTHODB_GENE_XREFS,
)

ensure_human_gene_info(HUMAN_GENE_INFO, HUMAN_GENE_INFO_URL)
ensure_human_ncbi_annotations(HUMAN_NCBI_ANNOTATIONS, ORTHODB_GENE_XREFS)
PY
    '
fi

echo "=== Running pipeline in Docker image: ${IMAGE} ==="
echo "Docker processors: ${DOCKER_CPUS}"
echo "OrthoFinder threads: ${ORTHOFINDER_THREADS_VALUE}"

"${DOCKER_RUN[@]}" \
  --platform "${PLATFORM}" \
  --workdir /workspace/code/orthologs \
  --volume "${REPO_ROOT}:/workspace" \
  --env PYTHON_EXE=/opt/conda/bin/python \
  --env PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --env ORTHOFINDER_THREADS="${ORTHOFINDER_THREADS_VALUE}" \
  --env ORTHOLOGS_DOCKER_IMAGE="${IMAGE}" \
  --env ORTHOLOGS_DOCKER_IMAGE_ID="${IMAGE_ID}" \
  --env ORTHOLOGS_DOCKER_PLATFORM="${PLATFORM}" \
  --env ORTHOLOGS_DOCKER_CPUS="${DOCKER_CPUS}" \
  --env ORTHOLOGS_OMA_WORKERS="${ORTHOLOGS_OMA_WORKERS:-}" \
  "${IMAGE}" \
  bash orthologs_run_all.sh "$@"
