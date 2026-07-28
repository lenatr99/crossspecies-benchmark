#!/usr/bin/env bash
# Build and run evaluation workflows in the reproducible Docker image.
#
# Usage:
#   bash code/evaluation/run_evaluation_docker.sh code/evaluation/compute_avgbio.py --dataset dicty
#   bash code/evaluation/run_evaluation_docker.sh code/evaluation/assign_marker_labels.py --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${EVALUATION_DOCKER_IMAGE:-crossspecies-evaluation:2026-05-22}"
BUILD="${EVALUATION_DOCKER_BUILD:-1}"
if [[ -n "${EVALUATION_DOCKER_CPUS:-}" ]]; then
  DOCKER_CPUS="${EVALUATION_DOCKER_CPUS}"
elif command -v sysctl >/dev/null 2>&1; then
  DOCKER_CPUS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
elif command -v nproc >/dev/null 2>&1; then
  DOCKER_CPUS="$(nproc)"
else
  DOCKER_CPUS="4"
fi
PLATFORM="${EVALUATION_DOCKER_PLATFORM:-}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash code/evaluation/run_evaluation_docker.sh SCRIPT.py [args...]" >&2
  echo "Example: bash code/evaluation/run_evaluation_docker.sh code/evaluation/compute_avgbio.py --dataset dicty --label-keys cell_type" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for the reproducible evaluation runner." >&2
  exit 1
fi

DOCKER_BUILD=(docker build --file "${SCRIPT_DIR}/Dockerfile" --tag "${IMAGE}")
DOCKER_RUN=(docker run --rm --cpus="${DOCKER_CPUS}")

if [[ -n "${PLATFORM}" ]]; then
  DOCKER_BUILD+=(--platform "${PLATFORM}")
  DOCKER_RUN+=(--platform "${PLATFORM}")
fi

if [[ "${BUILD}" != "0" ]]; then
  echo "=== Building Docker image: ${IMAGE}${PLATFORM:+ (${PLATFORM})} ==="
  "${DOCKER_BUILD[@]}" "${SCRIPT_DIR}"
fi

IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
mkdir -p "${REPO_ROOT}/results/evaluation"

cat > "${REPO_ROOT}/results/evaluation/docker_runtime_manifest.json" <<EOF
{
  "image": "${IMAGE}",
  "image_id": "${IMAGE_ID}",
  "platform": "${PLATFORM:-host-default}",
  "dockerfile": "code/evaluation/Dockerfile",
  "environment": "code/evaluation/environment.yml",
  "requirements_lock": "code/evaluation/requirements.lock.txt",
  "docker_cpus": "${DOCKER_CPUS}"
}
EOF

if [[ -t 0 && -t 1 ]]; then
  DOCKER_RUN+=(-it)
fi

echo "=== Running evaluation in Docker image: ${IMAGE} ==="
echo "Docker processors: ${DOCKER_CPUS}"

"${DOCKER_RUN[@]}" \
  --workdir /workspace \
  --volume "${REPO_ROOT}:/workspace" \
  --env PYTHONPATH=/workspace/code/evaluation \
  --env EVALUATION_DOCKER_CPUS="${DOCKER_CPUS}" \
  --env MPLCONFIGDIR=/workspace/data/evaluation/.cache/matplotlib \
  --env NUMBA_CACHE_DIR=/workspace/data/evaluation/.cache/numba \
  --env XDG_CACHE_HOME=/workspace/data/evaluation/.cache/xdg \
  --env LOKY_MAX_CPU_COUNT=1 \
  --env NUMBA_NUM_THREADS=1 \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  "${IMAGE}" \
  python "$@"
