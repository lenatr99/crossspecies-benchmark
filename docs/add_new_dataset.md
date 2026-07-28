# Add a new species

The `xspecies` command is a workflow front-end, not a zero-configuration upload
service. A new atlas must be registered explicitly because gene identifiers,
ortholog evidence, protein assets, biological labels and developmental
coordinates are organism-specific.

This checklist separates the minimum needed to compute embeddings from the
additional metadata needed for the full benchmark and publication figures.

## 1. Create a clean AnnData object

Store the atlas as `data/<dataset>.h5ad` with:

- cells in rows and genes in columns;
- unique `obs_names` and `var_names`;
- raw counts in sparse `X`;
- the same counts in `layers["counts"]`;
- a stable cell identifier in `obs["cell_id"]`;
- stable source-species gene identifiers in a documented `var` column;
- the biological batch or time columns needed by the intended analyses.

Inspect an existing H5AD before registering it:

```bash
xspecies inspect-genes data/my_species.h5ad
```

For a Matrix Market or other matrix-series export, start with:

```bash
xspecies convert-h5ad --help
```

If the source should be downloadable and rebuilt automatically, add a
source-file definition, builder, expected shape and identifier validation to
`code/datasets/prepare_geo_data.py`. Otherwise, document the external
preparation procedure and checksum.

## 2. Choose the benchmark scope

Decide which of these levels the dataset supports:

1. **Species-native baselines:** PCA and scVI need a clean count matrix.
2. **Fixed-vocabulary models:** Geneformer and scGPT additionally need
   source-to-human ortholog tiers.
3. **UCE:** requires target-species protein sequences, gene-to-protein
   resolution, chromosome positions and UCE token/checkpoint assets.
4. **Biological evaluation:** requires marker-derived or independent cell
   labels.
5. **Developmental ordering:** requires an interpretable observed or published
   temporal coordinate.
6. **Publication figures:** requires all selected headline embeddings,
   evaluations, palettes and optional layout artwork.

Document unsupported levels rather than filling them with proxies.

## 3. Register gene mappings and model inputs

### Human-ortholog route

For Geneformer or scGPT:

- add the organism to `ORGANISMS` in `code/orthologs/config.py`;
- define the stable source gene-ID column and output mapping;
- add appropriate OrthoDB, OrthoFinder, OMA and curated-resource inputs;
- extend `DATASETS` in
  `code/orthologs/orthologs_04_annotate_h5ad.py`;
- generate and review `ortholog_tier_1`, `ortholog_tier_2` and
  `ortholog_tier_3` in `adata.var`.

Do not silently reduce a many-to-many relationship to one target. Record the
conflict rule, evidence sources and retained model identifier in a manifest.

### Protein-representation route

For UCE, add a `DatasetConfig` entry in
`code/embeddings/uce/embed.py` that defines:

- target H5AD and managed UCE input paths;
- species key, taxonomy ID and assembly;
- protein FASTA source;
- chromosome-position filename;
- protein-embedding, token and offset asset names.

Verify how transcript/protein identifiers collapse to gene identifiers and
record the number of genes surviving both protein and chromosome filters.

## 4. Register embedding pipelines

Add the dataset key and its path/identifier conventions to the `DATASETS`
registry in each model you intend to run:

- `code/embeddings/pca/embed.py`
- `code/embeddings/scvi/embed.py`
- `code/embeddings/scgpt/embed.py`
- `code/embeddings/geneformer/embed.py`
- `code/embeddings/uce/embed.py`
- `code/embeddings/plot_umaps.py`
- `code/embeddings/reduce_dimensions.py`

Preserve the canonical `obsm` key conventions documented in
[`code/EMBEDDINGS.md`](../code/EMBEDDINGS.md). Do not change the headline
method list unless the scientific comparison itself is changing.

Run non-destructive smoke tests before a full atlas. For example:

```bash
xspecies embed pca \
  --dataset my_species \
  --variants full \
  --max-cells 256 \
  --overwrite
```

Every full run should write a manifest with input/output paths, dimensions,
parameters, software versions or locks, and model/checkpoint provenance.

## 5. Add biological annotations

Create `code/evaluation/src/gene_lists_<dataset>.py` following the existing
Dictyostelium and *C. elegans* modules. Each benchmark label bundle must define:

- marker sets and literature provenance;
- whether labels are mutually exclusive or multi-label;
- unknown/unlabelled-cell handling;
- any expression-matched null or multiple-testing behavior.

Then extend the dataset choices and presets in:

- `code/evaluation/run_marker_presets.py`
- `code/evaluation/compute_avgbio.py`
- `code/evaluation/compute_pseudotime.py`

For pseudotime, specify the exact `obs` key and whether it is an observed
collection time or an inferred/published estimate. Those are different
validation targets and must be described accordingly.

## 6. Register result selection and figures

For timestamped run selection and standard figures, update:

- `DATASETS` in `code/select_run.py`;
- `DATASETS` in `code/figures/figure_presets.py`;
- `DATASETS` in `code/figures/assemble_figures.py`.

The figure preset must define the data path, time key, axis treatment,
categorical panels, category order, colors and avgBIO labels. If publication
assembly is required, add a reviewed SVG master under
`manuscript/figure_sources/`.

Generate and pin reviewed results:

```bash
xspecies avgbio     --dataset my_species
xspecies pseudotime --dataset my_species
xspecies select     --dataset my_species --run latest
xspecies figures    --dataset my_species
```

## 7. Validate the registration

Before proposing the dataset for a headline comparison, verify:

- expected cell/gene shape and sparse-count integrity;
- stable, unique cell and gene identifiers;
- exact usable-gene counts after every model-specific filter;
- all requested `obsm` and `obs` keys exist and have the expected length;
- smoke-test and full-run manifests are complete;
- avgBIO unknown-label handling is intentional;
- pseudotime roots come from the earliest valid state;
- repeated subsampling rebuilds graphs rather than reusing a full-data graph;
- figures read explicitly pinned result runs;
- the top-level README and data provenance documentation are updated.

Finally, run the lightweight repository checks from
[`CONTRIBUTING.md`](../CONTRIBUTING.md).
