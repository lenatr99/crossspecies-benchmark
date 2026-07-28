#!/usr/bin/env bash
# Universal launcher for the cross-species benchmark workflow.
#
# Install once (`pipx install -e .` from the repo root) and drive everything
# with the `xspecies` command, like any other tool. Without installing, the same
# commands work via `bash code/run.sh <command>`.
#
# One command surface for every step. Each step runs in its correct pinned conda
# env automatically (embedding models each have their own env; everything else
# uses the foundation or evaluation env). You never call the per-area
# run_*_locally.sh wrappers directly.
#
# Commands:
#   embed <model> [args]   Create embeddings  (model: pca|scvi|scgpt|geneformer|uce)
#   umap [args]            Compute & plot UMAPs            (plot_umaps.py)
#   annotate [args]        Assign marker-defined cell labels (run_marker_presets.py)
#   avgbio [args]          avgBIO biological scores        (compute_avgbio.py)
#   pseudotime [args]      Pseudotime trajectory scores    (compute_pseudotime.py)
#                          Uncertainty: subsample+recompute by default
#   protein-space [args]    Publication protein-space figure (UCE env)
#   uce-eligibility [args]  Exact chromosome-aware UCE gene eligibility
#   dimension-match [args]  Shared post-embedding 100-PC sensitivity
#   publication-validate    Verify manuscript-facing sensitivity artifacts
#   select [args]          Pin which results run figures use (--run latest|<ts>)
#   runs [args]            List result runs and the pinned one
#   figures [args]         Render manuscript figure panels (make_figures.py)
#   assemble-figures       Assemble and export final figures (Inkscape)
#   prepare-data [args]    Download GEO inputs and build Dicty/Elegans H5ADs
#   ortholog-annotate      Add tracked consensus ortholog tiers to the H5ADs
#   inspect-genes <h5ad>   Inspect gene ID columns before adding a dataset
#   convert-h5ad [args]    Convert processed matrix series + metadata to H5AD
#   workflow               Print the recommended end-to-end command sequence
#   <path/to/script.py>    Escape hatch: run an arbitrary repo script in its env
#
# Append any flags after the command, e.g.:
#   xspecies embed uce --dataset dicty
#   xspecies pseudotime --dataset dicty --embedding-preset main
#
# Set RUN_DRY=1 to print the delegated command instead of running it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"   # .../code
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EMB="${REPO_ROOT}/code/embeddings"
FOUNDATION_WRAP="${EMB}/run_foundation_analysis_locally.sh"
EVAL_WRAP="${REPO_ROOT}/code/evaluation/run_evaluation_locally.sh"
MODELS="pca scvi scgpt geneformer uce"

usage()    { sed -n '2,39p' "$0" | sed 's/^# \{0,1\}//'; }

workflow() {
  cat <<EOF
End-to-end workflow for a dataset (example: dicty)
(Use 'xspecies <command>' if installed, or 'bash code/run.sh <command>'.)

  1. Prepare the clean source matrix:
       xspecies prepare-data --dataset dicty

  2. Add the tracked consensus human-ortholog tiers required by the
     fixed-vocabulary models:
       xspecies ortholog-annotate --dataset dicty

  3. Embeddings — one pinned env per model; run the models you need:
       xspecies embed pca        --dataset dicty
       xspecies embed scvi       --dataset dicty
       xspecies embed scgpt      --dataset dicty
       xspecies embed geneformer --dataset dicty
       xspecies embed uce        --dataset dicty

  4. UMAP coordinates (foundation env) — writes <key>_umap into the H5AD:
       xspecies umap --dataset dicty

  5. Annotate cell labels — marker-based; produces the cell_type / gene-set
     columns that avgBIO and the figures consume:
       xspecies annotate --dataset dicty

  6. Evaluation metrics (evaluation env) — each writes a timestamped run:
       xspecies avgbio     --dataset dicty
       xspecies pseudotime --dataset dicty

     Pseudotime uncertainty defaults to --inference subsample: 30 paired repeats,
     each retaining 90% of cells without replacement, recomputing the neighbor
     graph + DPT per repeat (20 roots). This is the method the manuscript
     describes. Pass --inference bootstrap for the legacy fixed-matrix bootstrap.

  7. Pin which run the figures should use (or a specific --run <timestamp>):
       xspecies select --dataset dicty --run latest
       xspecies runs   --dataset dicty            # inspect available runs

  8. Render manuscript figure panels (foundation env):
       xspecies figures --dataset dicty

  9. Assemble and export the final SVG/PDF/PNG figures (requires Inkscape):
       xspecies assemble-figures --dataset dicty

Which embeddings every step compares is the MAIN set in code/embedding_sets.py
(see code/EMBEDDINGS.md).
EOF
}

# run WRAPPER [args...] — exec, or echo under RUN_DRY=1.
run() {
  if [[ "${RUN_DRY:-0}" == "1" ]]; then
    echo "$*"
  else
    exec "$@"
  fi
}

require_wrapper() {
  [[ -f "$1" ]] || { echo "run.sh: missing env wrapper $1" >&2; exit 1; }
}

[[ $# -ge 1 ]] || { usage; exit 2; }
cmd="$1"; shift

case "${cmd}" in
  -h | --help | help)
    usage ;;

  workflow | list)
    workflow ;;

  embed)
    [[ $# -ge 1 ]] || { echo "run.sh embed: pick a model (${MODELS// /|})" >&2; exit 2; }
    model="$1"; shift
    case " ${MODELS} " in
      *" ${model} "*)
        wrap="${EMB}/${model}/run_${model}_locally.sh"
        require_wrapper "${wrap}"
        run bash "${wrap}" "$@" ;;
      *)
        echo "run.sh embed: unknown model '${model}' (${MODELS// /|})" >&2; exit 2 ;;
    esac ;;

  umap | umaps)
    require_wrapper "${FOUNDATION_WRAP}"
    run bash "${FOUNDATION_WRAP}" code/embeddings/plot_umaps.py "$@" ;;

  figures)
    require_wrapper "${FOUNDATION_WRAP}"
    run bash "${FOUNDATION_WRAP}" code/figures/make_figures.py "$@" ;;

  assemble-figures)
    require_wrapper "${FOUNDATION_WRAP}"
    run bash "${FOUNDATION_WRAP}" code/figures/assemble_figures.py "$@" ;;

  inspect-genes | gene-ids)
    require_wrapper "${FOUNDATION_WRAP}"
    run bash "${FOUNDATION_WRAP}" code/datasets/inspect_gene_ids.py "$@" ;;

  prepare-data | download-data | geo-data)
    require_wrapper "${FOUNDATION_WRAP}"
    run bash "${FOUNDATION_WRAP}" code/datasets/prepare_geo_data.py "$@" ;;

  ortholog-annotate | annotate-orthologs)
    require_wrapper "${FOUNDATION_WRAP}"
    run bash "${FOUNDATION_WRAP}" code/orthologs/orthologs_04_annotate_h5ad.py "$@" ;;

  convert-h5ad | matrix-to-h5ad)
    require_wrapper "${FOUNDATION_WRAP}"
    run bash "${FOUNDATION_WRAP}" code/datasets/convert_matrix_series_to_h5ad.py "$@" ;;

  annotate | markers)
    require_wrapper "${EVAL_WRAP}"
    run bash "${EVAL_WRAP}" code/evaluation/run_marker_presets.py "$@" ;;

  avgbio)
    require_wrapper "${EVAL_WRAP}"
    run bash "${EVAL_WRAP}" code/evaluation/compute_avgbio.py "$@" ;;

  pseudotime)
    require_wrapper "${EVAL_WRAP}"
    run bash "${EVAL_WRAP}" code/evaluation/compute_pseudotime.py "$@" ;;

  protein-space)
    require_wrapper "${EMB}/uce/run_uce_locally.sh"
    run bash "${EMB}/uce/run_uce_locally.sh" \
      --publication-script code/embeddings/uce/plot_protein_space.py "$@" ;;

  uce-eligibility)
    require_wrapper "${EMB}/uce/run_uce_locally.sh"
    run bash "${EMB}/uce/run_uce_locally.sh" \
      --publication-script code/embeddings/uce/export_gene_eligibility.py "$@" ;;

  dimension-match | dimension-matched)
    require_wrapper "${FOUNDATION_WRAP}"
    run bash "${FOUNDATION_WRAP}" code/embeddings/reduce_dimensions.py "$@" ;;

  publication-validate)
    require_wrapper "${EMB}/pca/run_pca_locally.sh"
    run bash "${EMB}/pca/run_pca_locally.sh" \
      --publication-script code/figures/validate_publication_revision.py "$@" ;;

  select)
    # Pin which results run the figures use. Pure stdlib; no conda env needed.
    run python3 "${REPO_ROOT}/code/select_run.py" set "$@" ;;

  runs)
    run python3 "${REPO_ROOT}/code/select_run.py" list "$@" ;;

  *.py | */*)
    # Escape hatch: an arbitrary repo script, routed to its env by path.
    case "${cmd}" in
      "${REPO_ROOT}/"*) rel="${cmd#"${REPO_ROOT}"/}" ;;
      /*) echo "run.sh: script must live under ${REPO_ROOT}" >&2; exit 1 ;;
      *)  rel="${cmd#./}" ;;
    esac
    case "${rel}" in
      code/evaluation/*)
        require_wrapper "${EVAL_WRAP}"; run bash "${EVAL_WRAP}" "${rel}" "$@" ;;
      code/embeddings/pca/* | code/embeddings/scvi/* | code/embeddings/scgpt/* | \
      code/embeddings/geneformer/* | code/embeddings/uce/*)
        model="$(printf '%s' "${rel}" | cut -d/ -f3)"
        wrap="${EMB}/${model}/run_${model}_locally.sh"
        require_wrapper "${wrap}"
        echo "run.sh: '${model}' scripts run via 'bash code/run.sh embed ${model} ...'; forwarding args." >&2
        run bash "${wrap}" "$@" ;;
      code/embeddings/* | code/figures/*)
        require_wrapper "${FOUNDATION_WRAP}"; run bash "${FOUNDATION_WRAP}" "${rel}" "$@" ;;
      code/datasets/*)
        require_wrapper "${FOUNDATION_WRAP}"; run bash "${FOUNDATION_WRAP}" "${rel}" "$@" ;;
      *)
        echo "run.sh: don't know which env runs '${rel}'." >&2
        echo "Try a named command instead — see: bash code/run.sh --help" >&2
        exit 1 ;;
    esac ;;

  *)
    echo "run.sh: unknown command '${cmd}'" >&2
    usage >&2
    exit 2 ;;
esac
