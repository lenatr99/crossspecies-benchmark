# scVI embeddings

Rerunnable pipeline for **scVI cell embeddings** on full gene sets and
human-ortholog tier gene sets. It follows the exploratory notebook pattern:
copy `adata.X` into `adata.layers["counts"]`, call
`scvi.model.SCVI.setup_anndata(..., layer="counts")`, train
`SCVI(n_layers=2, n_latent=100, gene_likelihood="nb")`, and write the latent
representation into the source H5AD `obsm`.

The default mode does **no batch correction**. The optional `batchcorrected`
mode repeats training with `batch_key="time"` by default. Training is in memory
and Lightning checkpoints/loggers are disabled to avoid large temporary files.

## Prerequisites

- **AnnData**: `data/dicty.h5ad` and/or `data/elegans.h5ad`.
- Ortholog variants need `ortholog_tier_1`, `ortholog_tier_2`, and
  `ortholog_tier_3` in `adata.var`.
- Batch-corrected mode needs the chosen `--batch-key` in `adata.obs`.
- Repo root assumed as cwd.

## Embedding variants

| Variant   | Genes used                                             |
|-----------|--------------------------------------------------------|
| `full`    | All measured genes                                     |
| `tier1`   | `ortholog_tier_1`                                      |
| `tier12`  | Tier 1, then fallback to `ortholog_tier_2` where empty |
| `tier123` | Tiers 1 -> 3 in order                                  |

Default run: `full,tier1,tier12,tier123` with no batch correction.

Outputs (non-`--max-cells` runs) patch the base H5AD:

- `X_scVI`, `X_scVI_50`
- `X_scVI_batchcorrected`, `X_scVI_batchcorrected_50`
- `X_scVI_ortholog_{variant}`, `X_scVI_ortholog_{variant}_50`
- `X_scVI_batchcorrected_ortholog_{variant}`,
  `X_scVI_batchcorrected_ortholog_{variant}_50`

Manifest: `results/embeddings/scvi_{dataset}_manifest.json` (or `_smoke` if
`--max-cells`).

Work/cache: no intermediate H5AD is written. Smoke-test arrays live under
`data/scvi/work/`; caches are kept under `data/scvi/.cache/`.

## Run locally

From the repo root:

```bash
bash code/embeddings/scvi/run_scvi_locally.sh \
  --dataset dicty \
  --variants tier12 \
  --overwrite
```

Smoke test without modifying the source H5AD:

```bash
bash code/embeddings/scvi/run_scvi_locally.sh \
  --dataset dicty \
  --variants tier12 \
  --max-cells 256 \
  --max-epochs 2 \
  --overwrite
```

The wrapper creates/reuses `scRNA_scVI` as a small Python 3.11 + pip conda env,
then installs the pinned top-level stack from `scvi_pip_requirements.txt`,
matching the scGPT runtime pattern. Set `SCVI_LOCAL_UPDATE=1` to refresh pip
installs.

Run the batch-corrected variant explicitly:

```bash
bash code/embeddings/scvi/run_scvi_locally.sh \
  --dataset dicty \
  --variants tier12 \
  --modes batchcorrected
```

By default, existing `obsm` outputs or smoke `.npy` arrays are reused. Pass
`--overwrite` to retrain and replace them.

Manual equivalent:

```bash
mamba create -y -n scRNA_scVI -c conda-forge python=3.11 pip
mamba run -n scRNA_scVI python -m pip install -r \
  code/embeddings/scvi/scvi_pip_requirements.txt
```

The root `envs/environment_scVI.yml` is kept as a notebook-friendly conda spec,
but the rerunnable wrapper uses the local requirements file for consistency
with the other embedding runtimes.

Useful `embed.py` flags: `--variants`, `--modes`, `--batch-key`,
`--accelerator`, `--devices`, `--max-epochs`, `--max-cells`, `--n-latent`,
`--overwrite`, `--results-dir`, and `--work-dir`.

## UMAP plots

After embeddings exist in `obsm`, use the shared plotting environment:

```bash
bash code/embeddings/run_foundation_analysis_locally.sh \
  code/embeddings/plot_umaps.py \
  --dataset dicty \
  --model scvi \
  --variants tier12 \
  --overwrite
```

The generic plotter expects the standard cleaned keys:
`X_scVI_ortholog_{variant}_50`.
