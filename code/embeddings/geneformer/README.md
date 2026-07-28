# Geneformer ortholog embeddings

Rerunnable pipeline for **CLS cell embeddings** on human-ortholog–mapped datasets: prepares tier-filtered transcriptomes (`prepare_ortholog_h5ad` in [`../_utils.py`](../_utils.py)), maps HGNC symbols to Geneformer Ensembl IDs, runs **`TranscriptomeTokenizer`** → **`EmbExtractor`**, PCA, and writes results into the source H5AD `obsm` plus JSON manifests under `results/embeddings/`.

## Prerequisites

- **Geneformer checkout** at `models/Geneformer/` (expects `geneformer/ensembl_mapping_dict_gc95M.pkl`). The local runner clones `https://huggingface.co/ctheodoris/Geneformer`, checks out commit `18a2ca6`, pulls Git LFS files, and installs it editable when missing.
- **Pretrained checkpoint** directory `models/Geneformer/gf-12L-95M-i4096/` from the Geneformer checkout.
- **AnnData**: `data/dicty.h5ad` and/or `data/elegans.h5ad`, with **`ortholog_tier_1`**, **`ortholog_tier_2`**, **`ortholog_tier_3`** (and tier-specific human-symbol columns consistent with `_utils`).
- Repo root assumed as cwd (scripts resolve paths relative to the repository).

## Embedding variants (`embed.py`)

| Variant   | Columns used                                             |
|-----------|----------------------------------------------------------|
| `tier1`   | `ortholog_tier_1`                                      |
| `tier12`  | Tier 1, then fallback to `ortholog_tier_2` where empty |
| `tier123` | Tiers 1 → 3 in order                                     |

Outputs (non–`--max-cells` runs) patch the base H5AD:

- **`X_geneformer_ortholog_{variant}`** — CLS embedding matrix  
- **`X_geneformer_ortholog_{variant}_50`** — PCA (`min(50, n_cells, n_dim)` components)  

For **`tier12`**, legacy aliases **`X_geneformer_orto`** and **`X_geneformer_orto_50`** are written for compatibility.

**Manifest:** `results/embeddings/geneformer_{dataset}_manifest.json` (or `_smoke` if `--max-cells`).

**Work/cache:** intermediate tokenized Arrow datasets live under `data/geneformer/work/`; caches under `data/geneformer/.cache/`.

## Run locally (recommended)

Run from the host so Hugging Face Datasets can write Arrow shards to local disk at full speed (**`Dataset.map`** during “Creating dataset” is painful over **Docker Desktop bind mounts** on macOS).

From the repo root:

```bash
bash code/embeddings/geneformer/run_geneformer_locally.sh --dataset dicty --variants tier12 --overwrite --nproc 4
```

Creates **`crossspecies-geneformer`** once via `mamba create` + **`pip install -r geneformer_pip_requirements.txt`** (pinned stack; avoids **`mamba env create`** YAML + embedded **`pip:`** hook bugs seen on macOS). It also clones/updates **`models/Geneformer`**, checks out the pinned commit, pulls Git LFS assets, and installs Geneformer editable. If an existing `models/Geneformer` checkout cannot switch commits cleanly, the runner uses a managed pinned checkout at `models/Geneformer-18a2ca6` instead.

Environment variables:

- **`GENEFORMER_ENV_NAME`** — conda env name (default `crossspecies-geneformer`).
- **`GENEFORMER_REPO_URL`** — Geneformer git remote (default `https://huggingface.co/ctheodoris/Geneformer`).
- **`GENEFORMER_COMMIT`** — Geneformer commit to checkout (default `18a2ca6`).
- **`GENEFORMER_MANAGED_DIR`** — fallback managed checkout path (default `models/Geneformer-${GENEFORMER_COMMIT}`).
- **`GENEFORMER_LOCAL_UPDATE=1`** — refresh pip installs from **`geneformer_pip_requirements.txt`** before running.
- If an install failed midway: **`mamba env remove -n crossspecies-geneformer -y`**, then rerun the script.

**Reuse env without reinstall:** if the env already exists and **`GENEFORMER_LOCAL_UPDATE` is unset**, the script skips create/update and only runs **`embed.py`**.

Invoke **`embed.py`** directly:

```bash
cd /path/to/project
export PYTHONPATH="${PWD}/models/Geneformer:${PYTHONPATH:-}"
mamba run -n crossspecies-geneformer python code/embeddings/geneformer/embed.py --help
```

Useful **`embed.py`** flags: `--variants`, `--overwrite`, **`--nproc`**, **`--chunk-size`**, **`--forward-batch-size`**, **`--max-cells`** (smoke test; writes `.npy` instead of patching H5AD), **`--results-dir`**, **`--work-dir`**, **`--geneformer-dir`**, **`--model-dir`**, **`--no-force-cpu`**.

## UMAP plots

UMAPs and figures should run in the shared **foundation-analysis** environment,
not the Geneformer runtime environment. This keeps visualization dependencies
and stochastic UMAP settings pinned across Geneformer, scGPT, UCE, scVI, and
any later foundation-model outputs.

After embeddings exist in **`obsm`**, generate Scanpy PNGs under
**`results/embeddings/umaps/geneformer/`**:

```bash
bash code/embeddings/run_foundation_analysis_locally.sh \
  code/embeddings/plot_umaps.py \
  --dataset dicty \
  --model geneformer \
  --variants tier12 \
  --overwrite
```

The wrapper creates/reuses **`crossspecies-foundation-analysis`** from
[`../foundation_analysis_pip_requirements.txt`](../foundation_analysis_pip_requirements.txt).
Set **`FOUNDATION_ANALYSIS_UPDATE=1`** to refresh those pins. You do not need to
activate the env when using the wrapper; it runs the script via `mamba run`.
For direct/manual plotting without activation:

```bash
mamba run -n crossspecies-foundation-analysis \
  python code/embeddings/plot_umaps.py --dataset dicty --model geneformer --variants tier12 --overwrite
```

If you prefer `mamba activate`, initialize your current zsh first:

```bash
eval "$(mamba shell hook --shell zsh)"
mamba activate crossspecies-foundation-analysis
```

The generic plotter has tiered ortholog presets for **`geneformer`**, **`pca`**,
**`scvi`**, **`scgpt`**, and **`uce`**:

```bash
bash code/embeddings/run_foundation_analysis_locally.sh \
  code/embeddings/plot_umaps.py --dataset dicty --model scgpt --overwrite
```

All presets expect the same `obsm` naming convention:
**`X_{model}_ortholog_{tier}_50`** for **`tier1,tier12,tier123`**. Use
**`--representations embedding`** for non-PCA50 tier embeddings, or
**`--embedding-keys`** for exact legacy `obsm` names such as `X_scGPT_orto_50`
or `X_uce_50`. Coloring uses **`time`** and **`cell_type`** in `obs` unless
**`--colors`** is set. UMAP parameters are written to each manifest; default
stochastic seed is **`--random-state 42`**.

## Files in this folder

| File                       | Purpose                                        |
|---------------------------|------------------------------------------------|
| `embed.py`                | Main embedding pipeline                        |
| `run_geneformer_locally.sh` | conda bootstrap + **embed.py**                |
| `geneformer_pip_requirements.txt` | pinned pip stack for **`run_geneformer_locally.sh`** |
