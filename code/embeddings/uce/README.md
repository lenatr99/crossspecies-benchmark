# UCE embeddings (local)

This folder runs **Universal Cell Embedding (UCE)** on the cleaned datasets and writes embeddings into the target AnnData files (`data/dicty.h5ad`, `data/elegans.h5ad`). Work artifacts go under `data/UCE/work/`; run summaries go under `results/embeddings/` as JSON manifests.

## Prerequisites

- **mamba** or **micromamba** on your PATH (Miniforge/Mambaforge is fine).
- A checkout of UCE at **`models/UCE`** (the wrapper clones [snap-stanford/UCE](https://github.com/snap-stanford/UCE.git) if it is missing).
- Prepared UCE inputs under **`data/UCE/`** (for example `dicty.h5ad` for Dictyostelium and `elegans.h5ad` for *C. elegans*), matching the paths in `embed.py`.
- Target objects **`data/dicty.h5ad`** and **`data/elegans.h5ad`** must exist; embeddings are appended to `obsm`.

## Run from the repository root

Typical full run (both species, 4-layer and 33-layer, full + ortholog tiers):

```bash
bash code/embeddings/uce/run_uce_locally.sh \
  --dataset all \
  --models 4layer,33layer \
  --variants full,tier1,tier12,tier123 \
  --overwrite
```

The wrapper defaults to `--seed 42` and patches the local UCE entrypoint so
stochastic cell-sentence sampling is seeded inside the UCE subprocess
(`PYTHONHASHSEED`, Python `random`, NumPy, and Torch). Use `--seed <int>` to
intentionally generate a different deterministic run.

Dicty only:

```bash
bash code/embeddings/uce/run_uce_locally.sh \
  --dataset dicty \
  --models 4layer,33layer \
  --variants full,tier1,tier12,tier123 \
  --seed 42 \
  --overwrite
```

Dicty without chromosome-aware gene ordering:

```bash
bash code/embeddings/uce/run_uce_locally.sh \
  --dataset dicty \
  --models 33layer \
  --variants full \
  --no-chromosome-aware \
  --overwrite
```

This keeps UCE's protein-token inputs but builds each cell sentence as
`CLS + sampled expressed gene tokens`, without chromosome start/end tokens or
genomic-position sorting. Outputs are written with a `_no_chrom` suffix, for
example `X_uce_33L_no_chrom` and `X_uce_33L_no_chrom_50`, so the
standard chromosome-aware embeddings can coexist in the same AnnData file.

Smoke test (few cells, no writes into the main `data/*.h5ad`):

```bash
bash code/embeddings/uce/run_uce_locally.sh \
  --dataset dicty \
  --models 4layer \
  --variants full \
  --max-cells 100 \
  --overwrite
```

Drop specific genes from the UCE input **before** embedding (comma-separated). Each token matches, in order: current `var_names`, `var["gene_ids"]`, then `var["Gene Symbol"]` (case-insensitive). After dicty normalization, `var_names` are DDB IDs (`DDB_G*`); either **`rad51-1`** or **`DDB_G0273139`** works.

```bash
bash code/embeddings/uce/run_uce_locally.sh \
  --dataset dicty \
  --models 4layer,33layer \
  --variants full,tier1,tier12,tier123 \
  --exclude-genes rad51-1 \
  --overwrite
```

Several genes:

```bash
--exclude-genes rad51-1,DDB_G0273139
```

Run UCE with a fitted global ESM2 whitening adapter:

```bash
bash code/embeddings/uce/run_uce_locally.sh \
  --dataset dicty \
  --models 4layer \
  --variants full,tier1,tier12,tier123 \
  --whitening \
  --exclude-genes rad51-1 \
  --overwrite
```

Whitening outputs are written with a `_whitening` suffix, for example
`X_uce_4L_whitening` and `X_uce_4L_whitening_50`.

Run UCE with Tier 1 ortholog-guided whitening:

```bash
bash code/embeddings/uce/run_uce_locally.sh \
  --dataset dicty \
  --models 4layer \
  --variants full,tier1,tier12,tier123 \
  --whitening \
  --whitening-tiers tier1 \
  --overwrite
```

Tier 1 whitening estimates the source/human ESM2 mean and standard deviation
from matched Tier 1 ortholog pairs, then applies that global transform to all
source genes. Outputs are written with a `_whiteningT1` suffix, for example
`X_uce_4L_whiteningT1_50`. Here `T1` marks the whitening fit set, while
`orthoT1` in other keys marks the embedding input gene-set variant.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `UCE_ENV_NAME` | Conda env name (default: `crossspecies-uce`) |
| `UCE_LOCAL_UPDATE` | Set to `1` to re-run `pip install -r code/embeddings/uce/uce_pip_requirements.txt` in an existing env |
| `UCE_DIR` | Path to UCE checkout (default: `<repo>/models/UCE`) |
| `UCE_REPO_URL` | Git URL if cloning UCE (default: GitHub snap-stanford/UCE) |
| `UCE_COMMIT` | UCE commit to checkout (default: `8227a65`) |

## Direct invocation (after env exists)

If you already created **`crossspecies-uce`** and installed UCE requirements:

```bash
cd /path/to/magistrska
mamba run -n crossspecies-uce python code/embeddings/uce/embed.py \
  --dataset dicty \
  --models 4layer,33layer \
  --variants full,tier1,tier12,tier123 \
  --overwrite
```

## Outputs

- **`obsm`** keys such as `X_uce_4L`, `X_uce_4L_50`, `X_uce_33L`, `X_uce_33L_50`, and ortholog-tier variants such as `X_uce_4L_orthoT1_50` (see `embed.py` → `model_output_keys`).
- With `--no-chromosome-aware`, equivalent keys use `_no_chrom`, such as
  `X_uce_33L_no_chrom_50`.
- **`results/embeddings/uce_<dataset>_manifest.json`** (and `*_smoke_manifest.json` when using `--max-cells`).
- Temporary UCE work `.h5ad` and `.npz` files under `data/UCE/work/` are removed after each variant/run and listed in the manifest.

Each manifest includes a reproducibility block with Python/package/Torch
versions, UCE git state and patch diff hash, before/after target H5AD hashes,
and SHA-256 checksums for model checkpoints, protein embeddings, token files,
FASTA/chromosome assets, and adapter files when used. Whitening adapter
manifests under `results/embeddings/uce/esm2_adapter/` record the same kind of
environment and asset provenance for the adapted ESM2 dictionaries.

For UMAP figures after embeddings exist, use **`plot_umaps.py`** with **`--model uce`** (documented in `code/embeddings/README.md`).
