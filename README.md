# Crossing the species barrier in single-cell foundation models

<p align="center">
  <strong>A reproducible benchmark of zero-shot single-cell representations in evolutionarily distant organisms.</strong>
</p>

<p align="center">
  <a href=".github/workflows/repo-smoke.yml"><img alt="Repository smoke checks" src="https://img.shields.io/badge/checks-GitHub%20Actions-2088ff?logo=githubactions&amp;logoColor=white"></a>
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-%E2%89%A53.9-3776ab">
  <a href="CITATION.cff"><img alt="Citation metadata" src="https://img.shields.io/badge/citation-CFF-4c1"></a>
</p>

This repository implements a reproducible benchmark of pretrained Geneformer,
scGPT and Universal Cell Embedding (UCE) representations against PCA and scVI
in time-resolved *Dictyostelium discoideum* and *Caenorhabditis elegans*
atlases. It evaluates the complete route from species-native genes to model
inputs, cell embeddings, developmental ordering and marker-derived biological
annotations.

The workflow is exposed through one command, `xspecies`, while each model and
evaluation step runs in its own pinned environment.

[See the quick start](#quick-start) ·
[Run the benchmark](#run-the-benchmark) ·
[Add another species](docs/add_new_dataset.md)

## Why this benchmark?

Many single-cell foundation models use a fixed human gene vocabulary. Applying
them to distant organisms therefore requires source genes to be mapped to human
orthologs, with ambiguous and unsupported genes discarded before inference.
UCE instead constructs inputs from target-species protein representations and
genomic positions. This benchmark makes that input-coverage difference explicit
and evaluates whether it accompanies changes in biological representation
quality.

| Atlas | Cells | Measured genes | Developmental reference | Annotation panels |
|---|---:|---:|---|---|
| *D. discoideum* | 48,983 | 12,883 | Observed 0–20 h collection time | Cell type, cAMP-pulse-induced genes, milestones |
| *C. elegans* | 89,701 | 20,222 | Published embryo-time estimate | Cell/tissue class, lineage, developmental markers |
| **Total** | **138,684** | — | Two complementary developmental systems | **Six panels** |

### Headline comparison

| Method | Input strategy | *D. discoideum* genes | *C. elegans* genes | Output dimensions |
|---|---|---:|---:|---:|
| PCA | Species-native counts | 12,883 (100%) | 20,222 (100%) | 100 |
| scVI | Species-native counts | 12,883 (100%) | 20,222 (100%) | 100 |
| Geneformer | Tier 1–2 human orthologs + model vocabulary | 2,297 (17.8%) | 3,506 (17.3%) | 512 |
| scGPT | Tier 1–2 human orthologs + model vocabulary | 2,289 (17.8%) | 3,517 (17.4%) | 512 |
| UCE-4L | Target-species protein and chromosome assets | 12,548 (97.4%) | 19,720 (97.5%) | 1,280 |

The exact headline embedding keys live in
[`code/embedding_sets.py`](code/embedding_sets.py), the single source of truth
used by evaluation and figure generation.

### Main findings

- Fixed human-vocabulary models retained less than one-fifth of measured genes
  after consensus ortholog mapping and model-specific filtering; UCE retained
  about 97% of genes through target-species protein assets.
- UCE-4L was the most consistent method in the headline comparison, ranking
  first on five of six avgBIO annotation panels and leading developmental
  ordering in the branched *C. elegans* atlas.
- In *D. discoideum*, several methods recovered developmental order similarly.
  Conventional PCA preprocessing also changed several annotation rankings,
  showing that the result is endpoint- and baseline-dependent.
- Coverage-aligned UCE controls indicated that gene availability explained the
  large *D. discoideum* cell-type difference, but did not explain all remaining
  annotation and developmental-ordering patterns.

These are benchmark-specific findings, not evidence that one representation is
universally best. The tracked results include coverage controls, stability
analyses and sensitivity comparisons.

## Quick start

### Requirements

- macOS or Linux
- Python 3.9 or newer for the lightweight launcher
- [`mamba`](https://mamba.readthedocs.io/) or `micromamba`
- Git and Git LFS for model checkouts downloaded on first use
- Internet access for the source data, environments and pretrained model assets
- Optional: Docker for rebuilding the ortholog consensus and Inkscape for final
  publication figure assembly

Full-atlas embedding and evaluation runs are computationally substantial.
Model-specific READMEs document smoke-test flags and runtime controls.

### Install the launcher

```bash
git clone https://github.com/lenatr99/crossspecies-benchmark.git
cd crossspecies-benchmark
pipx install -e .
```

An editable install is intentional: `xspecies` resolves scripts, data and
results relative to this checkout. A regular editable install also works:

```bash
python -m pip install -e .
```

Without installing anything, replace `xspecies` with
`bash code/run.sh` in every example.

Check the command surface without starting a model run:

```bash
xspecies --help
xspecies workflow
RUN_DRY=1 xspecies embed uce --dataset dicty
```

### Prepare the benchmark data

```bash
xspecies prepare-data --dataset all
xspecies ortholog-annotate --dataset all
```

The first command downloads retained source files under `data/sources/`, builds
`data/dicty.h5ad` and `data/elegans.h5ad`, validates their expected shapes and
identifiers, and writes checksums to `data/manifest.json`. The second command
adds the tracked Tier 1–3 consensus human-ortholog annotations required by the
fixed-vocabulary and ortholog-control runs.

Existing H5AD outputs are protected. Rebuilding requires `--overwrite`.
Preparation can also be staged:

```bash
xspecies prepare-data --dataset all --download-only
xspecies prepare-data --dataset all --build-only
xspecies prepare-data --dataset all --dry-run
```

See [`data/README.md`](data/README.md) for sources, expected shapes, tracked
metadata and generated files.

### Run a small smoke test

Once a dataset exists, verify the local PCA pipeline without modifying the main
H5AD:

```bash
xspecies embed pca \
  --dataset dicty \
  --variants full \
  --max-cells 256 \
  --overwrite
```

## Run the benchmark

Run the following from the repository root for `dicty`, then repeat with
`--dataset elegans`:

```bash
# 1. Build the five headline representations.
xspecies embed pca        --dataset dicty
xspecies embed scvi       --dataset dicty
xspecies embed scgpt      --dataset dicty
xspecies embed geneformer --dataset dicty
xspecies embed uce        --dataset dicty

# 2. Add shared UMAP coordinates and marker-derived annotations.
xspecies umap     --dataset dicty
xspecies annotate --dataset dicty

# 3. Evaluate annotation preservation and developmental ordering.
xspecies avgbio     --dataset dicty
xspecies pseudotime --dataset dicty

# 4. Pin reviewed result runs before making figures.
xspecies select --dataset dicty --run latest
xspecies runs   --dataset dicty

# 5. Render data-driven panels and assemble the publication figure.
xspecies figures          --dataset dicty
xspecies assemble-figures --dataset dicty
```

`avgbio` and `pseudotime` write timestamped runs beneath
`results/<area>_<dataset>/runs/<timestamp>/`. Figures read the run explicitly
pinned with `xspecies select`, not whichever run happened most recently.

Pseudotime uncertainty defaults to 30 paired 90%-cell subsamples, with the
neighbor graph and diffusion pseudotime recomputed in every repeat. avgBIO is
the unweighted mean of NMI, ARI and label ASW for the marker-derived
annotations.

### Rebuild the ortholog consensus

The repository tracks the consensus tables and compact mappings needed to
reproduce downstream runs. To rebuild them from OrthoFinder, OrthoDB, OMA and
the curated organism resources, use the pinned Docker workflow:

```bash
bash code/orthologs/orthologs_run_all_docker.sh
xspecies ortholog-annotate --dataset all
```

This workflow downloads large reference assets, including the OrthoDB gene-xref
dump. See [`code/orthologs/README.md`](code/orthologs/README.md) before running
it.

## Command reference

| Command | Purpose | Runtime |
|---|---|---|
| `xspecies prepare-data …` | Download sources and build clean H5AD files | Foundation |
| `xspecies ortholog-annotate …` | Add tracked consensus ortholog tiers to `adata.var` | Foundation |
| `xspecies embed <model> …` | Create PCA, scVI, scGPT, Geneformer or UCE embeddings | Per model |
| `xspecies umap …` | Compute and store UMAP coordinates | Foundation |
| `xspecies annotate …` | Assign marker-derived labels | Evaluation |
| `xspecies avgbio …` | Compute annotation-preservation scores | Evaluation |
| `xspecies pseudotime …` | Compute trajectory-conservation scores and stability intervals | Evaluation |
| `xspecies dimension-match …` | Reduce headline representations above 100D to 100 PCs | Foundation |
| `xspecies protein-space …` | Recreate the raw-versus-calibrated protein-space analysis | UCE |
| `xspecies uce-eligibility …` | Export chromosome-aware UCE gene eligibility | UCE |
| `xspecies publication-validate …` | Validate publication-facing sensitivity artifacts | PCA |
| `xspecies select …` / `runs …` | Pin or inspect timestamped evaluation runs | Python standard library |
| `xspecies figures …` | Render data-driven publication panels | Foundation |
| `xspecies assemble-figures …` | Assemble SVG/PDF/PNG publication figures | Foundation + Inkscape |
| `xspecies inspect-genes …` | Audit H5AD gene identifiers | Foundation |
| `xspecies convert-h5ad …` | Convert matrix-series inputs and metadata to H5AD | Foundation |

Use `xspecies <command> --help` for step-specific options. Set `RUN_DRY=1` to
print the delegated command without executing it.

## Repository layout

```text
.
├── code/
│   ├── datasets/       # source download, validation and H5AD conversion
│   ├── embeddings/     # model pipelines and pinned environments
│   ├── evaluation/     # marker labels, avgBIO and pseudotime
│   ├── figures/        # panel rendering and publication assembly
│   └── orthologs/      # multi-source human-ortholog consensus
├── data/               # compact mappings plus ignored local H5AD/model inputs
├── docs/               # extension and contributor-facing guides
├── results/            # manifests, benchmark runs and paper-facing tables
├── crossspecies_cli.py # installed xspecies entry point
└── pyproject.toml
```

More detailed documentation:

- [Embedding conventions and model environments](code/embeddings/README.md)
- [Canonical embedding sets](code/EMBEDDINGS.md)
- [Evaluation workflows](code/evaluation/README.md)
- [Figure pipeline](code/figures/README.md)
- [Ortholog consensus pipeline](code/orthologs/README.md)
- [Adding a new dataset](docs/add_new_dataset.md)

## Outputs and reproducibility

- Embeddings, UMAP coordinates, pseudotime and labels are stored additively in
  the local AnnData objects (`obsm`, `obs` and `var`).
- Model pipelines write JSON manifests under `results/embeddings/`, including
  parameters, package or checkpoint information and input/output provenance.
- Evaluation outputs use timestamped run directories and explicit `CURRENT`
  pointers selected by the user.
- Conda specifications, conda-lock files and pip lock files are kept beside the
  workflow that consumes them.
- Large datasets, model checkpoints, caches and generated work products are
  intentionally excluded from Git; compact mappings, manifests, figure sources
  and publication-facing results are tracked.

## Extending the benchmark

The reusable commands are dataset-independent, but a new atlas must define its
gene identifiers, model assets, biological labels and developmental coordinate.
Start with:

```bash
xspecies inspect-genes data/my_species.h5ad
xspecies convert-h5ad --help
```

The complete registration checklist is in
[`docs/add_new_dataset.md`](docs/add_new_dataset.md).

## Citation

Citation metadata for this software and its authors is provided in
[`CITATION.cff`](CITATION.cff). Please cite the software repository as:

> Lena Trnovec, Mariko Katoh-Kurasawa, Rachel N. Arey, Gad Shaulsky, and Blaž
> Zupan. *Crossing the species barrier in single-cell foundation models*.
> Version 0.1.0. https://github.com/lenatr99/crossspecies-benchmark

## Contributing

Bug reports, reproducibility reports and support for additional species are
welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull
request.

## License

No license is granted for this repository. All rights are reserved.
