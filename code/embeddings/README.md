# Foundation-model embeddings

Shared, rerunnable embedding utilities live here. Model-specific pipelines go
in subfolders such as `geneformer/`; shared postprocessing and conventions stay
at this level.

## Standard `obsm` Keys

All cleaned embedding pipelines should write tiered ortholog outputs with the
same naming pattern:

| Model | PCA50 `obsm` keys |
|-------|-------------------|
| Geneformer | `X_geneformer_ortholog_tier1_50`, `X_geneformer_ortholog_tier12_50`, `X_geneformer_ortholog_tier123_50` |
| PCA | `X_pca_ortholog_tier1_50`, `X_pca_ortholog_tier12_50`, `X_pca_ortholog_tier123_50` |
| scVI | `X_scVI_ortholog_tier1_50`, `X_scVI_ortholog_tier12_50`, `X_scVI_ortholog_tier123_50` |
| scGPT | `X_scGPT_ortholog_tier1_50`, `X_scGPT_ortholog_tier12_50`, `X_scGPT_ortholog_tier123_50` |
| UCE | `X_uce_4L_orthoT1_50`, `X_uce_4L_orthoT12_50`, `X_uce_4L_orthoT123_50`; equivalent `33L` keys for the 33-layer model |

Raw/non-PCA embeddings should use the same pattern without `_50`, for example
`X_scGPT_ortholog_tier12`.

## Shared Ortholog Prep

`_utils.py` contains model-agnostic helpers for:

- selecting tiered human-symbol orthologs from `adata.var`
- mapping selected symbols into a model-specific ID space
- collapsing duplicate target IDs by summing expression columns with one sparse
  matrix multiplication

Model-specific scripts should reuse these helpers rather than reimplementing
tier selection or paralog aggregation.

## UCE embeddings (local GPU / CPU run)

Run UCE via the wrapper that creates/reuses the **`crossspecies-uce`** conda env and calls **`uce/embed.py`**:

```bash
bash code/embeddings/uce/run_uce_locally.sh \
  --dataset all \
  --models 4layer,33layer \
  --variants full,tier1,tier12,tier123 \
  --overwrite
```

See **`uce/README.md`** for prerequisites, smoke tests, and environment variables.

## scVI embeddings

Run scVI in its separate environment:

```bash
bash code/embeddings/scvi/run_scvi_locally.sh \
  --dataset dicty \
  --variants tier12 \
  --overwrite
```

The wrapper creates/reuses **`scRNA_scVI`** as a small Python 3.11 + pip conda
env, then installs **`scvi/scvi_pip_requirements.txt`**. The embedding script
trains directly from the source H5AD in memory, disables Lightning
checkpoints/logging, and writes only `obsm` arrays plus a small manifest. The
default is no batch correction; pass `--modes batchcorrected` explicitly for
the `batch_key="time"` version. See **`scvi/README.md`** for full and
batch-corrected key names.

## PCA embeddings

Run the PCA baseline in its separate environment:

```bash
bash code/embeddings/pca/run_pca_locally.sh \
  --dataset dicty \
  --variants tier12 \
  --overwrite
```

The wrapper creates/reuses **`crossspecies-pca`** as a small Python 3.11 + pip
conda env, then installs **`pca/pca_pip_requirements.txt`**. The default recipe
does no preprocessing and writes a 100-component key such as
**`X_pca_ortholog_tier12`** plus the standard comparison key
**`X_pca_ortholog_tier12_50`**. Opt into Scanpy-style processing with
**`--recipes normalized_log1p,hvg_scaled,hvg_scaled_batchcorrected`**, then use
the shared UMAP plotter for visualization.

## Shared Analysis Env

Use a separate analysis environment for UMAPs and comparison figures. This keeps
plotting versions and stochastic UMAP behavior independent of model runtimes.

**Reproducibility:** the wrapper prefers **`conda-lock install`** from **`environment.foundation_analysis.conda-lock.yml`** (exact conda-forge URLs + MD5 for **osx-arm64** and **linux-64**), then **`pip install -r foundation_analysis_pip_lock.txt`** for the Scanpy/stack closure. Editing **`foundation_analysis_pip_requirements.txt`** means regenerate the pip lock (header in `foundation_analysis_pip_lock.txt`); editing **`environment.foundation_analysis.yml`** means re-run **conda-lock** (below). **`FOUNDATION_ANALYSIS_UPDATE=1`** only reapplies the **pip** lock; recreate the conda env if you bump the conda lockfile. **`FOUNDATION_ANALYSIS_SKIP_CONDA_LOCK=1`** skips `conda-lock` and uses the YAML-only solve (fallback if `conda-lock` is not installed).

Install **`conda-lock`** once (for example **`mamba install -n base conda-lock`**).

**Refresh the conda solver lock:**

```bash
cd code/embeddings
conda-lock lock -f environment.foundation_analysis.yml \
  -p osx-arm64 -p linux-64 \
  --lockfile environment.foundation_analysis.conda-lock.yml --conda mamba
```

What can still drift: solver lock is **frozen until you regenerate**, but **`conda-lock`/channel index** evolve over years; numerical UMAP output can still differ slightly by CPU/float semantics even when packages match.

```bash
bash code/embeddings/run_foundation_analysis_locally.sh \
  code/embeddings/plot_umaps.py \
  --dataset dicty \
  --model geneformer \
  --overwrite
```

The wrapper creates/reuses **`crossspecies-foundation-analysis`** using the lockfile. Set **`FOUNDATION_ANALYSIS_UPDATE=1`** to reinstall from **`foundation_analysis_pip_lock.txt`** before running. You do not need to activate the environment; the script runs via **`mamba run`**. That is the safest approach when **`mamba activate`** is not initialized.

Manual equivalent (conda pins + pip pins):

```bash
conda-lock install -n crossspecies-foundation-analysis --conda mamba \
  code/embeddings/environment.foundation_analysis.conda-lock.yml
mamba run -n crossspecies-foundation-analysis pip install -r \
  code/embeddings/foundation_analysis_pip_lock.txt
```

Fallback without **conda-lock** (unpinned conda solves): `mamba env create -f environment.foundation_analysis.yml`. PyPI stacks stay outside the YAML for portability and smaller **conda-lock** payloads.

For direct/manual use without activating:

```bash
mamba run -n crossspecies-foundation-analysis \
  python code/embeddings/plot_umaps.py --dataset dicty --model geneformer --overwrite
```

If you prefer `mamba activate`, initialize your current zsh first:

```bash
eval "$(mamba shell hook --shell zsh)"
mamba activate crossspecies-foundation-analysis
```

## UMAPs

`plot_umaps.py` reads embeddings from `adata.obsm`, computes UMAP coordinates,
writes them back as `{embedding_key}_umap`, and saves figures under:

```text
results/embeddings/umaps/<model>/<dataset>/
```

The UCE preset splits outputs one level deeper by model depth:
`results/embeddings/umaps/uce/<dataset>/4layer/` and
`results/embeddings/umaps/uce/<dataset>/33layer/`.

Examples:

```bash
# All Geneformer tier PCA50 UMAPs
bash code/embeddings/run_foundation_analysis_locally.sh \
  code/embeddings/plot_umaps.py --dataset dicty --model geneformer --overwrite

# One tier only
bash code/embeddings/run_foundation_analysis_locally.sh \
  code/embeddings/plot_umaps.py --dataset dicty --model geneformer --variants tier12 --overwrite

# Future scGPT tier outputs using the same key convention
bash code/embeddings/run_foundation_analysis_locally.sh \
  code/embeddings/plot_umaps.py --dataset dicty --model scgpt --overwrite

# Exact legacy/nonstandard keys
bash code/embeddings/run_foundation_analysis_locally.sh \
  code/embeddings/plot_umaps.py --dataset dicty --embedding-keys X_scGPT_orto_50,X_uce_50 --output-prefix legacy --overwrite

# Dicty UCE full + ortholog PCA50 UMAPs split by model depth
bash code/embeddings/run_foundation_analysis_locally.sh \
  code/embeddings/plot_umaps.py \
  --dataset dicty \
  --model uce \
  --overwrite
```

UCE runs created with `--no-chromosome-aware` write separate `_no_chrom` keys.
The UCE plotting preset includes those PCA50 keys when present; missing keys are
skipped by default.

UCE runs created with `--whitening --whitening-tiers tier1` use Tier 1
ortholog-guided ESM2 whitening and write separate
`_whiteningT1` keys. Here `T1` marks the whitening fit set, not the embedding
input gene-set variant. The UCE plotting preset includes those PCA50 keys when
present.

Default colors are `time` and `cell_type`; override with `--colors`. UMAP
parameters are recorded in each manifest, with `--random-state 42` by default.

## Files

| File | Purpose |
|------|---------|
| `_utils.py` | Shared tier selection, ID mapping, duplicate-ID collapse, temp H5AD prep |
| `plot_umaps.py` | Generic UMAP computation and figure export from `obsm` embeddings |
| `environment.foundation_analysis.yml` | Conda specs for **conda-lock** (constraints only) |
| `environment.foundation_analysis.conda-lock.yml` | Solver-pinned conda packages (**osx-arm64**, **linux-64**) |
| `foundation_analysis_pip_requirements.txt` | Curated top-level pins (maintain here; regenerate lock below) |
| `foundation_analysis_pip_lock.txt` | Full transitive `pip` lock consumed by **`run_foundation_analysis_locally.sh`** |
| `geneformer/` | Geneformer-specific embedding runtime |
| `pca/` | PCA baseline runtime (`run_pca_locally.sh`, `embed.py`; see **`pca/README.md`**) |
| `scvi/` | scVI embedding runtime (`run_scvi_locally.sh`, `embed.py`, `scvi_pip_requirements.txt`; see **`scvi/README.md`**) |
| `uce/` | UCE embedding runtime (`run_uce_locally.sh`, `embed.py`; see **`uce/README.md`**) |
