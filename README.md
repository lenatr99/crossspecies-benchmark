# cross-species benchmark

Benchmarking single-cell foundation-model embeddings across species. The whole
workflow is driven by one installable command, `xspecies`.

## Install

```bash
pipx install -e .      # recommended — isolated, puts `xspecies` on your PATH
# or:  pip install -e .
```

Editable (`-e`) install is intended: the scripts, conda-env wrappers, data, and
results all live in this repo, so `xspecies` resolves them relative to the
checkout. (No pipx? Every command also works as `bash code/run.sh <command>`.)

You also need `mamba` or `micromamba` — each step creates/uses its own pinned
conda env on first run.

## Prepare the benchmark data

Do this before running embeddings or evaluations. The repository does not
track the large input H5ADs, so download the archived sources and build both
clean datasets with:

```bash
xspecies prepare-data --dataset all
```

The command retains downloads under `data/sources/`, writes `data/dicty.h5ad`
and `data/elegans.h5ad`, validates their expected dimensions and identifiers,
and updates `data/manifest.json` with source URLs and checksums. Use
`--dataset dicty` or `--dataset elegans` to prepare only one dataset. Existing
output H5ADs are protected; rebuilding them requires an explicit `--overwrite`.
For a comparison build, select one dataset and pass an exact path such as
`--output data/dicty_1.h5ad`.

Useful staged commands:

```bash
xspecies prepare-data --dataset all --download-only
xspecies prepare-data --dataset all --build-only
xspecies prepare-data --dataset all --dry-run
```

Dicty uses the paper's exact filtered AX4 object from
[Zenodo 10.5281/zenodo.18481907](https://doi.org/10.5281/zenodo.18481907);
[GSE305468](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305468)
remains the underlying sequencing-data archive. C. elegans is converted from
the sparse count matrix and cell/gene annotations in
[GSE126954](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126954).

## Usage

```bash
xspecies workflow          # print the full end-to-end command sequence
xspecies <command> --help  # flags for a step (forwarded to the underlying script)
```

| Command | What it does | Env |
|---------|--------------|-----|
| `xspecies prepare-data …` | Download the archived sources and build clean Dicty/Elegans `.h5ad` files | foundation |
| `xspecies embed <model> …` | Create embeddings (`pca`/`scvi`/`scgpt`/`geneformer`/`uce`) | per-model |
| `xspecies umap …` | Compute & write UMAP coordinates | foundation |
| `xspecies annotate …` | Assign marker-defined cell labels | evaluation |
| `xspecies avgbio …` | avgBIO biological-conservation scores | evaluation |
| `xspecies pseudotime …` | Pseudotime trajectory-conservation scores | evaluation |
| `xspecies protein-space …` | Reproducible raw-versus-calibrated protein-space figure | UCE |
| `xspecies uce-eligibility …` | Exact chromosome-aware UCE gene eligibility export | UCE |
| `xspecies dimension-match …` | Reduce the headline representations above 100 dimensions to a common 100-PC output | foundation |
| `xspecies publication-validate …` | Verify PCA hashes and manuscript-facing sensitivity artifacts | PCA |
| `xspecies select …` | Pin which results run the figures use (`--run latest`/`<ts>`) | — |
| `xspecies runs …` | List result runs and show the pinned one | — |
| `xspecies figures …` | Render data-driven manuscript figure panels | foundation |
| `xspecies assemble-figures …` | Assemble and export final SVG/PDF/PNG figures | foundation + Inkscape |
| `xspecies inspect-genes …` | Audit `.h5ad` gene ID columns before adding a dataset | foundation |
| `xspecies convert-h5ad …` | Convert processed matrix-series data plus metadata to `.h5ad` | foundation |

`xspecies` is a thin launcher over [`code/run.sh`](code/run.sh), which routes
each step to its correct pinned conda env. Set `RUN_DRY=1` to print the command
a step would run without running it.

## Typical run for a dataset

```bash
xspecies prepare-data     --dataset dicty
xspecies embed pca        --dataset dicty
xspecies embed scvi       --dataset dicty
xspecies embed scgpt      --dataset dicty
xspecies embed geneformer --dataset dicty
xspecies embed uce        --dataset dicty
xspecies umap             --dataset dicty
xspecies annotate         --dataset dicty
xspecies avgbio           --dataset dicty
xspecies pseudotime       --dataset dicty
xspecies select           --dataset dicty --run latest   # pin runs for the figures
xspecies figures          --dataset dicty
xspecies assemble-figures --dataset dicty
```

`avgbio`/`pseudotime` write a timestamped run under
`results/<area>_<dataset>/runs/<timestamp>/`. The figures read the run you pin
with `xspecies select` (not whatever ran last); `xspecies runs --dataset dicty`
shows what's available and which is pinned.

Figure generation is intentionally split into panel rendering and publication
assembly. Tracked Illustrator masters live in `manuscript/figure_sources/`;
final assets are written to `manuscript/figures/`. See
[`code/figures/README.md`](code/figures/README.md) for prerequisites, output
paths, validation, and optional per-panel exports.

Which embeddings every step compares is the `MAIN` set in
[`code/embedding_sets.py`](code/embedding_sets.py) — the single source of truth.
See [`code/EMBEDDINGS.md`](code/EMBEDDINGS.md).

## Benchmark an additional species

`xspecies` is intended to be an extensible benchmark front-end, not a
zero-configuration upload service. This checkout contains ready-to-run
registrations for `dicty` and `elegans`. The dataset-independent preparation
commands can be used immediately on another atlas:

```bash
xspecies convert-h5ad --help
xspecies inspect-genes data/my_species.h5ad
```

Before `xspecies embed`, `annotate`, `avgbio`, `pseudotime`, `select`, and
`figures` accept a new dataset name, register its paths and biological metadata
and supply the model-specific assets it needs. Fixed-human-vocabulary models
require human ortholog annotations; UCE requires a compatible protein
dictionary and chromosome-position assets; biological evaluation requires
marker labels and, for trajectory evaluation, a temporal coordinate.

The complete onboarding and registration checklist is in
[`docs/add_new_dataset.md`](docs/add_new_dataset.md). It identifies the exact
configuration entry points, distinguishes required benchmark inputs from
optional publication artwork, and provides a planarian example. Once the
dataset key is registered, the same end-to-end `xspecies` commands shown above
apply to that species.
