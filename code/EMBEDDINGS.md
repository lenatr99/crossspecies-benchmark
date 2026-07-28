# Embedding sets — single source of truth

All embedding-key lists used anywhere in the repo are defined **once**, in
[`code/embedding_sets.py`](embedding_sets.py). Every benchmark and plotting
entrypoint imports from there, so "the main five methods" means exactly the same
keys in avgBIO, the pseudotime benchmark, and the manuscript figures.

**To change which embeddings are compared, edit `code/embedding_sets.py` — and
nothing else.**

## The sets

| Name | Meaning |
|------|---------|
| `main` | The headline comparison: one representative embedding per method (PCA, scVI, Geneformer, scGPT, UCE). **Default everywhere.** |
| `layer_supp` | UCE 4-layer vs 33-layer. |
| `ortholog_supp` | Ortholog tier sweep (tier1/12/123) for Geneformer, scGPT, UCE. |
| `adapter_supp` | UCE adapter / whitening / macrogene variants. |
| `pca_preprocessing_supp` | Raw-count PCA versus normalized, log-transformed, HVG-scaled PCA. |
| `raw` | The full per-embedding grid (`exhaustive` is an alias). |
| `standardized_50` | The full grid re-standardized to 50 dims. |
| `dimension_matched_100_supp` | The three headline embeddings above 100 dimensions independently reduced to 100 PCs; native 100D PCA and scVI are reused unchanged. |

`MAIN` and its display labels (`MAIN_LABELS`) live next to the sets. The labels
name the *method* ("scVI"), not the exact obsm key.

## Who imports it

| Consumer | Re-export shim | Imports |
|----------|----------------|---------|
| avgBIO (`compute_avgbio.py`) | `evaluation/util/avgbio_presets.py` | `AVGBIO_EMBEDDING_PRESETS`, `DEFAULT_AVGBIO_EMBEDDINGS` |
| Pseudotime (`compute_pseudotime.py`) | `evaluation/util/pseudotime_presets.py` | `PSEUDOTIME_EMBEDDING_PRESETS`, `DEFAULT_PSEUDOTIME_EMBEDDINGS` |
| UMAPs (`plot_umaps.py`) | `embeddings/plot_umap_presets.py` | `EMBEDDING_SETS` names as `--model` presets (`main` by default) |
| Figures (`make_figures.py`) | `figures/figure_presets.py` | `METHODS` (= `MAIN`), `METHOD_LABELS` (= `MAIN_LABELS`) |

The shims only rename the canonical names into each area's local vocabulary;
they contain no embedding lists of their own. A consumer adds `code/` to
`sys.path` and does `from embedding_sets import ...`.

## Caveat: figures need per-key UMAP + pseudotime columns

The figures read `obsm["<key>_umap"]` and `obs["<key>_dpt_pseudotime"]` for each
method in `MAIN`. Those columns must already exist in the dataset H5AD. Generate
them for the `main` keys before plotting:

```bash
# UMAP coordinates for the MAIN keys (default --model is 'main')
xspecies umap --dataset dicty

# pseudotime columns (also writes results/pseudotime_<org>/pseudotime.json)
xspecies pseudotime --dataset dicty --embedding-preset main
```

The same preset names are available for supplemental UMAPs:

```bash
xspecies umap --dataset dicty --model layer_supp
xspecies umap --dataset dicty --model ortholog_supp
xspecies umap --dataset dicty --model adapter_supp
```

`xspecies` (installed via `pipx install -e .`) is the program front-end; it
delegates to `code/run.sh`, which runs each step in its correct pinned conda env
(embedding models each have their own; everything else uses the foundation or
evaluation env). Without installing, use `bash code/run.sh <command>`. See the
top-level [`README.md`](../README.md).

`make_figures.py` prints and skips any method whose `_umap` basis is missing, so
it degrades gracefully rather than crashing.
