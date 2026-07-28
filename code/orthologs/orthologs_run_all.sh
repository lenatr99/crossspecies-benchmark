#!/usr/bin/env bash
# Multi-evidence ortholog consensus pipeline (Dicty + C. elegans → human).
#
# Order of operations per organism:
#   01a  OrthoDB API pair fetch          (already cached snapshots → CSVs)
#   01b  OrthoDB orthogroup membership   (bulk download → parquet)
#   01c  OrthoFinder phylogenetic run    (DIAMOND-more-sensitive + MSA + FastTree)
#   01d  OMA REST orthologs              (per-protein cached JSON)
#   01e  Curated literature              (OrthoList 2 / dictyBase)
#   02   Consensus tier assignment
#   03   Final {gene: [symbol]} JSON     (--min-tier 2 by default)
# Then once both organisms are done:
#   99   Report tables (tier distribution, source-overlap Venn, legacy diff)
#
# Pass --legacy to also re-run the legacy reciprocal-best-hit pipeline into a
# sibling output directory for the manuscript ablation comparison.
#
# Python runs inside conda env ``orthologs`` by default (OrthoFinder toolchain,
# pandas, requests, pyarrow, …). Override with ``CONDA_ENV=myenv`` or bypass conda entirely with
# ``PYTHON_EXE=/path/to/python``.
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "${SCRIPT_PATH}")"

CONDA_ENV="${CONDA_ENV:-orthologs}"
if [[ -n "${PYTHON_EXE:-}" ]]; then
  PYTHON=( "${PYTHON_EXE}" )
elif command -v conda >/dev/null 2>&1; then
  PYTHON=( conda run -n "${CONDA_ENV}" --no-capture-output python )
else
  echo "Warning: conda not on PATH; using python3. Prefer: conda activate ${CONDA_ENV}" >&2
  PYTHON=( python3 )
fi

ORGANISMS=(dicty elegans)

REFRESH_FLAG=""
LEGACY=0
SKIP_ORTHOFINDER=0
SKIP_OMA=0
ORTHOFINDER_THREADS="${ORTHOFINDER_THREADS:-24}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --refresh-cache) REFRESH_FLAG="--refresh-cache"; shift ;;
    --legacy)        LEGACY=1; shift ;;
    --skip-orthofinder) SKIP_ORTHOFINDER=1; shift ;;
    --skip-oma)      SKIP_OMA=1; shift ;;
    --threads)       ORTHOFINDER_THREADS="$2"; shift 2 ;;
    -h|--help)
      grep -E '^# ' "${SCRIPT_PATH}" | grep -v 'shellcheck' | sed 's/^# //'
      echo ""
      echo "Environment: CONDA_ENV=${CONDA_ENV:-orthologs} (conda run … python)"
      echo "Override Python: PYTHON_EXE=/path/to/python bash ${SCRIPT_PATH}"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${ORTHOLOGS_DOCKER_IMAGE:-}" || -n "${ORTHOLOGS_DOCKER_IMAGE_ID:-}" ]]; then
  mkdir -p "../../results/orthologs"
  "${PYTHON[@]}" - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import os

manifest = {
    "image": os.environ.get("ORTHOLOGS_DOCKER_IMAGE"),
    "image_id": os.environ.get("ORTHOLOGS_DOCKER_IMAGE_ID"),
    "platform": os.environ.get("ORTHOLOGS_DOCKER_PLATFORM"),
    "python_exe": os.environ.get("PYTHON_EXE"),
    "generated_utc": datetime.now(timezone.utc).isoformat(),
}
out = Path("../../results/orthologs/docker_runtime_manifest.json")
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"Docker runtime manifest: {out}")
PY
fi

if [[ "${LEGACY}" -eq 1 ]]; then
  LEGACY_OUT="../../results/orthologs_legacy"
  mkdir -p "${LEGACY_OUT}"
  for organism in "${ORGANISMS[@]}"; do
    echo "=== [legacy] OrthoDB pair fetch: ${organism} ==="
    "${PYTHON[@]}" orthologs_01a_orthodb_pairs.py --organism "${organism}" --save_data ${REFRESH_FLAG} \
      --output-dir "${LEGACY_OUT}"
    if [[ -f legacy/orthologs_03_find_best.py ]]; then
      echo "=== [legacy] reciprocal-best-hit selection: ${organism} ==="
      "${PYTHON[@]}" legacy/orthologs_03_find_best.py --organism "${organism}" \
        --input-dir "${LEGACY_OUT}" --output-dir "${LEGACY_OUT}" \
        | tee "${LEGACY_OUT}/orthologs_${organism}.txt"
    else
      echo "Note: legacy/orthologs_03_find_best.py not found; skipping legacy ablation."
    fi
  done
fi

for organism in "${ORGANISMS[@]}"; do
  echo "=== 01a OrthoDB pair fetch: ${organism} ==="
  "${PYTHON[@]}" orthologs_01a_orthodb_pairs.py --organism "${organism}" --save_data ${REFRESH_FLAG}
done

echo "=== 01b OrthoDB OG membership: all organisms ==="
"${PYTHON[@]}" orthologs_01b_orthodb_og.py --organism all ${REFRESH_FLAG}

for organism in "${ORGANISMS[@]}"; do
  if [[ "${SKIP_ORTHOFINDER}" -eq 0 ]]; then
    echo "=== 01c OrthoFinder run: ${organism} ==="
    "${PYTHON[@]}" orthologs_01c_orthofinder.py --organism "${organism}" \
      --threads "${ORTHOFINDER_THREADS}" --reuse-results
  else
    echo "Skipping 01c OrthoFinder (--skip-orthofinder)."
  fi

  if [[ "${SKIP_OMA}" -eq 0 ]]; then
    echo "=== 01d OMA REST orthologs: ${organism} ==="
    "${PYTHON[@]}" orthologs_01d_oma_rest.py --organism "${organism}" ${REFRESH_FLAG}
  else
    echo "Skipping 01d OMA (--skip-oma)."
  fi

  echo "=== 01e Curated literature: ${organism} ==="
  "${PYTHON[@]}" orthologs_01e_curated.py --organism "${organism}" --auto-download

  echo "=== 02 Consensus: ${organism} ==="
  "${PYTHON[@]}" orthologs_02_consensus.py --organism "${organism}"

  echo "=== 03 Build symbol mapping: ${organism} ==="
  "${PYTHON[@]}" orthologs_03_build_symbol_mapping.py --organism "${organism}" --min-tier 2
  echo ""
done

echo "=== 99 Report tables ==="
LEGACY_INPUT_FLAG=""
if [[ "${LEGACY}" -eq 1 ]]; then
  LEGACY_INPUT_FLAG="--legacy-input-dir ../../results/orthologs_legacy"
fi
# shellcheck disable=SC2086
"${PYTHON[@]}" orthologs_99_report.py --organisms "${ORGANISMS[@]}" ${LEGACY_INPUT_FLAG}

echo "Done."
