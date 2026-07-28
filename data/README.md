# Benchmark data

This directory contains compact, reproducibility-critical mappings and metadata.
The large expression matrices, source downloads, model inputs and caches are
created locally and ignored by Git.

## Prepare the two paper datasets

From the repository root:

```bash
xspecies prepare-data --dataset all
xspecies ortholog-annotate --dataset all
```

The preparation command validates the exact shapes used in the manuscript:

| Dataset key | Organism | Output | Cells × genes | Developmental coordinate |
|---|---|---|---:|---|
| `dicty` | *Dictyostelium discoideum* | `data/dicty.h5ad` | 48,983 × 12,883 | `obs["time"]`, 0–20 h |
| `elegans` | *Caenorhabditis elegans* | `data/elegans.h5ad` | 89,701 × 20,222 | `obs["embryo.time"]` |

`ortholog-annotate` adds `var["ortholog_tier_1"]`,
`var["ortholog_tier_2"]` and `var["ortholog_tier_3"]` from the tracked
consensus tables. These columns are required by Geneformer, scGPT and the
ortholog-restricted controls.

## Sources

### *D. discoideum*

- Underlying sequencing archive:
  [GEO GSE305468](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305468)
- Exact filtered AX4 object used by the paper:
  [Zenodo 10.5281/zenodo.18481907](https://doi.org/10.5281/zenodo.18481907)

The builder copies the filtered expression and metadata, removes source-derived
embeddings and graphs by default, validates cell and gene identifiers, and
ensures a sparse `counts` layer is present.

### *C. elegans*

- Sparse count matrix and cell/gene annotations:
  [GEO GSE126954](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126954)

The GEO gene-by-cell matrix is transposed into a cell-by-gene sparse AnnData
object. WormBase IDs are retained in `var["id"]`, and the published
`embryo.time` estimate is retained in `obs`.

## Generated local layout

```text
data/
├── sources/                 # retained downloads; resumable and ignored
├── dicty.h5ad               # ignored; modified additively by the workflow
├── elegans.h5ad             # ignored; modified additively by the workflow
├── manifest.json            # generated source/output checksums
├── geneformer/              # ignored work and cache files
├── scgpt/                   # ignored work and cache files
├── scvi/                    # ignored work and cache files
├── UCE/                     # ignored UCE inputs, assets and work files
└── foundation_models/       # ignored shared caches
```

Embedding commands append arrays to `adata.obsm`; UMAP commands append
`<embedding>_umap` arrays; annotation and pseudotime commands add `obs`
columns. Keep an untouched copy if you need to compare before/after H5AD
contents.

## Tracked provenance

The repository intentionally tracks:

- compact final human-ortholog mappings and evidence sidecars;
- curated-source metadata and deterministic identifier bridges;
- human symbol annotations required downstream;
- consensus and source-overlap results under `results/orthologs/`;
- JSON manifests produced by paper-facing analyses.

The full ortholog consensus can be rebuilt with:

```bash
bash code/orthologs/orthologs_run_all_docker.sh
xspecies ortholog-annotate --dataset all
```

Read [`code/orthologs/README.md`](../code/orthologs/README.md) first: the
pipeline downloads multi-gigabyte reference files and runs phylogenetic tools.

## Safe preparation options

```bash
xspecies prepare-data --dataset dicty
xspecies prepare-data --dataset elegans
xspecies prepare-data --dataset all --download-only
xspecies prepare-data --dataset all --build-only
xspecies prepare-data --dataset all --dry-run
```

Existing output H5ADs are never replaced unless `--overwrite` is passed.
`--refresh-downloads` re-fetches retained sources, while `--skip-checksums`
skips expensive hashing when a provenance manifest is not required.
