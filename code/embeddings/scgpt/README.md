# scGPT embeddings

Rerunnable pipeline for **scGPT cell embeddings** on full gene sets and
human-ortholog tier gene sets. The script follows the exploratory notebook
`code/embeddings_dictyostelium/embed_scGPT.ipynb`: it uses the pretrained
checkpoint in `models/scGPT/`, passes a `Gene Symbol` column to
`scgpt.tasks.embed_data`, PCA-reduces the output to 50 dimensions, and writes
results into the source H5AD `obsm`.

## Prerequisites

- **Pretrained scGPT checkpoint** in `models/scGPT/` with `args.json`,
  `best_model.pt`, and `vocab.json`. The local wrapper downloads the same
  Google Drive folder used by the notebook if the files are missing.
- **AnnData**: `data/dicty.h5ad` and/or `data/elegans.h5ad`.
- Ortholog variants need `ortholog_tier_1`, `ortholog_tier_2`, and
  `ortholog_tier_3` in `adata.var`.
- Repo root assumed as cwd.

## Embedding variants

| Variant   | Genes used                                                  |
|-----------|-------------------------------------------------------------|
| `full`    | Measured gene symbols matched directly to the scGPT vocab   |
| `tier1`   | `ortholog_tier_1`                                           |
| `tier12`  | Tier 1, then fallback to `ortholog_tier_2` where empty      |
| `tier123` | Tiers 1 -> 3 in order                                       |

Default run: `tier1,tier12,tier123`.

Outputs (non-`--max-cells` runs) patch the base H5AD:

- `X_scGPT_full`, `X_scGPT_full_50`
- `X_scGPT_ortholog_{variant}`
- `X_scGPT_ortholog_{variant}_50`

Manifest: `results/embeddings/scgpt_{dataset}_manifest.json` (or
`_smoke` if `--max-cells`).

Work/cache: intermediate H5AD files live under `data/scgpt/work/`; caches under
`data/scgpt/.cache/`.

## Run locally

From the repo root:

```bash
bash code/embeddings/scgpt/run_scgpt_locally.sh \
  --dataset dicty \
  --variants tier12 \
  --overwrite
```

Smoke test without modifying the source H5AD:

```bash
bash code/embeddings/scgpt/run_scgpt_locally.sh \
  --dataset dicty \
  --variants tier12 \
  --max-cells 256 \
  --overwrite
```

The wrapper creates/reuses `crossspecies-scgpt` via `mamba create` plus
`pip install -r scgpt_pip_requirements.txt`, and passes `--download-model`
automatically. Set `SCGPT_LOCAL_UPDATE=1` to refresh pip installs, or
`SCGPT_DOWNLOAD_MODEL=0` to require an already-populated `models/scGPT/`.

Useful `embed.py` flags: `--variants`, `--overwrite`, `--batch-size`,
`--max-cells`, `--download-model`, `--results-dir`, `--work-dir`, `--model-dir`.

## UMAP plots

After embeddings exist in `obsm`, use the shared plotting environment:

```bash
bash code/embeddings/run_foundation_analysis_locally.sh \
  code/embeddings/plot_umaps.py \
  --dataset dicty \
  --model scgpt \
  --variants tier12 \
  --overwrite
```

The generic plotter expects the standard cleaned keys:
`X_scGPT_ortholog_{variant}_50`.
