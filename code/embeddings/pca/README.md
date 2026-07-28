# PCA embeddings

Rerunnable baseline pipeline for **PCA cell embeddings** on full gene sets and
human-ortholog tier gene sets. It writes PCA coordinates into the source H5AD
`obsm` and records a JSON manifest under `results/embeddings/`.

The default recipe is deliberately minimal: **no preprocessing** is applied
before PCA. Optional recipes reproduce the common Scanpy notebook steps for
normalize/log1p, HVG+scale, and HVG+scale with ComBat batch correction.

## Prerequisites

- **AnnData**: `data/dicty.h5ad` and/or `data/elegans.h5ad`.
- Ortholog variants need `ortholog_tier_1`, `ortholog_tier_2`, and
  `ortholog_tier_3` in `adata.var`.
- `hvg_scaled_batchcorrected` needs the chosen `--batch-key` in `adata.obs`.
- Repo root assumed as cwd.

## Embedding variants

| Variant   | Genes used                                             |
|-----------|--------------------------------------------------------|
| `full`    | All measured genes                                     |
| `tier1`   | `ortholog_tier_1`                                      |
| `tier12`  | Tier 1, then fallback to `ortholog_tier_2` where empty |
| `tier123` | Tiers 1 -> 3 in order                                  |

Default run: `full,tier1,tier12,tier123` with the `raw` recipe and 100 PCA
components. When at least 50 components are available, the script also writes a
`_50` view for the shared UMAP/comparison presets.

Default output keys:

- `X_pca`
- `X_pca_50`
- `X_pca_ortholog_tier1`
- `X_pca_ortholog_tier1_50`
- `X_pca_ortholog_tier12`
- `X_pca_ortholog_tier12_50`
- `X_pca_ortholog_tier123`
- `X_pca_ortholog_tier123_50`

Non-default recipes insert the recipe name after `X_pca`, for example
`X_pca_hvg_scaled_ortholog_tier12` and
`X_pca_hvg_scaled_ortholog_tier12_50`.

Manifest: `results/embeddings/pca_{dataset}_manifest.json` (or `_smoke` if
`--max-cells`).

Sensitivity runs should pass `--manifest-tag <tag>` so their provenance is
written to `pca_{dataset}_<tag>[_smoke]_manifest.json` without replacing the
primary raw-PCA manifest.

## Recipes

| Recipe | Processing |
|--------|------------|
| `raw` | No preprocessing; fit PCA directly on selected `adata.X`. |
| `normalized_log1p` | `normalize_total(target_sum=1e4)` then `log1p`. |
| `hvg_scaled` | Copy `X` to `layers["counts"]`, select HVGs, normalize/log1p, subset HVGs, scale. |
| `hvg_scaled_batchcorrected` | Same as `hvg_scaled`, with HVGs selected by `--batch-key` and `sc.pp.combat`. |

## Run locally

From the repo root:

```bash
xspecies embed pca \
  --dataset dicty \
  --variants tier12 \
  --overwrite
```

Smoke test without modifying the source H5AD:

```bash
xspecies embed pca \
  --dataset dicty \
  --variants tier12 \
  --max-cells 256 \
  --overwrite
```

Notebook-style run with all three processed recipes:

```bash
xspecies embed pca \
  --dataset dicty \
  --variants full \
  --recipes normalized_log1p,hvg_scaled,hvg_scaled_batchcorrected \
  --manifest-tag preprocessing_sensitivity \
  --overwrite
```

The wrapper creates/reuses `crossspecies-pca` as a small Python 3.11 + pip conda
env, then installs the fully pinned `pca_pip_lock.txt`. Set `PCA_LOCAL_UPDATE=1` to
refresh pip installs.

Manual equivalent:

```bash
mamba create -y -n crossspecies-pca -c conda-forge python=3.11 pip
mamba run -n crossspecies-pca python -m pip install -r \
  code/embeddings/pca/pca_pip_lock.txt
```

Useful `embed.py` flags: `--variants`, `--recipes`, `--n-comps`,
`--n-top-genes`, `--hvg-flavor`, `--batch-key`, `--max-cells`, `--overwrite`,
`--manifest-tag`, `--results-dir`, and `--work-dir`.

## UMAP plots

The shared plotting preset expects the default 50-component ortholog keys:
`X_pca_ortholog_{variant}_50`.

```bash
xspecies umap \
  --dataset dicty \
  --model pca \
  --variants tier12 \
  --overwrite
```

For recipe-specific or 100-component PCA keys, pass exact names:

```bash
xspecies umap \
  --dataset dicty \
  --embedding-keys X_pca_hvg_scaled_batchcorrected_ortholog_tier12 \
  --output-prefix pca_hvg_scaled_batchcorrected \
  --overwrite
```
