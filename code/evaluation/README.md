# Evaluation workflows

This folder contains the biological evaluation steps that run after embeddings
are written into `adata.obsm`.

- `assign_marker_labels.py` assigns marker-defined labels to `adata.obs`.
- `compute_avgbio.py` scores embeddings with avgBIO-style biological metrics.

It is kept separate from `code/embeddings/` so the evaluation runtime can be
pinned and reproduced independently.

## Environment

There are two supported runtimes:

- Local conda via `run_evaluation_locally.sh`, best for quick iteration.
- Docker via `run_evaluation_docker.sh`, best for safer reproducible runs
  because Python, system libraries, thread env vars, and package pins are all
  inside the image.

Create and run through the dedicated local environment wrapper:

```bash
bash code/evaluation/run_evaluation_locally.sh \
  code/evaluation/assign_marker_labels.py --help
```

The wrapper creates `crossspecies-evaluation` from `environment.yml`, then
installs the fully pinned Python package closure in `requirements.lock.txt`.
`requirements.txt` is the curated top-level package list; edit that first when
changing the stack, then refresh the lock file.

What the environment contains:

| Package | Why it is here |
|---------|----------------|
| `python`, `pip` | Fixed interpreter and installer baseline |
| `anndata`, `h5py` | Read/write `.h5ad`, including `obs`, `obsm`, and `uns` |
| `scanpy` | Normalization/log transforms, neighbor graphs, Leiden clustering |
| `numpy`, `pandas`, `scipy` | Array, table, sparse matrix, and numeric operations |
| `scib` | Original scIB metric implementations used for avgBIO |
| `scikit-learn` | Gaussian mixtures and supporting metrics used by marker/subsampling utilities |
| `igraph`, `leidenalg` | Leiden clustering backend used by Scanpy/scIB |
| `matplotlib` | Optional diagnostic plots for legacy Otsu marker labeling |
| `tqdm` | Progress bars for longer marker scoring runs |

To refresh an existing environment after editing pins:

```bash
CROSSSPECIES_EVALUATION_UPDATE=1 bash code/evaluation/run_evaluation_locally.sh \
  code/evaluation/assign_marker_labels.py --help
```

Build and run the Docker image:

```bash
bash code/evaluation/run_evaluation_docker.sh \
  code/evaluation/compute_avgbio.py --dataset dicty --label-keys cell_type
```

Docker runtime notes:

- Image tag defaults to `crossspecies-evaluation:2026-05-22`.
- Override with `EVALUATION_DOCKER_IMAGE=...`.
- Skip rebuilds with `EVALUATION_DOCKER_BUILD=0`.
- Limit CPUs with `EVALUATION_DOCKER_CPUS=4`.
- Set `EVALUATION_DOCKER_PLATFORM=linux/amd64` or `linux/arm64` when you want
  to force a platform; otherwise Docker uses the host default.
- Each Docker run writes `results/evaluation/docker_runtime_manifest.json`.

## Marker Labels

### Gene-list modules

Add these files when the marker lists are ready:

```text
code/evaluation/src/gene_lists_dicty.py
code/evaluation/src/gene_lists_elegans.py
```

Each module can expose either:

```python
cell_type = {
    "program_a": ["gene1", "gene2"],
    "program_b": ["gene3", "gene4"],
}
```

or a bundled mapping with the intended mode:

```python
GENE_SETS = {
    "cell_type": (cell_type, "xor"),
    "cell_cycle": (cell_cycle, "or"),
}
```

Use `xor` for mutually exclusive labels and `or` for multi-label marker
programs.

### Standard marker-label runs

Run all default Dictyostelium and C. elegans marker jobs:

```bash
bash code/evaluation/run_evaluation_locally.sh \
  code/evaluation/run_marker_presets.py --progress
```

Run one dataset or one gene-set bundle:

```bash
bash code/evaluation/run_evaluation_locally.sh \
  code/evaluation/run_marker_presets.py --dataset elegans --gene-sets cell_type
```

Optional jobs such as OR-style regulon/cell-cycle bundles are skipped by
default. Include them with:

```bash
bash code/evaluation/run_evaluation_locally.sh \
  code/evaluation/run_marker_presets.py --include-optional --progress
```

### Direct marker-label CLI

Example for Dictyostelium:

```bash
bash code/evaluation/run_evaluation_locally.sh \
  code/evaluation/assign_marker_labels.py \
  --data data/dicty.h5ad \
  --gene-lists code/evaluation/src/gene_lists_dicty.py \
  --gene-sets cell_type \
  --bundle-mode xor \
  --progress \
  --write-h5ad
```

Example for C. elegans:

```bash
bash code/evaluation/run_evaluation_locally.sh \
  code/evaluation/assign_marker_labels.py \
  --data data/elegans.h5ad \
  --gene-lists code/evaluation/src/gene_lists_elegans.py \
  --gene-sets cell_type \
  --bundle-mode xor \
  --var-names-key gene_short_name \
  --gene-id-col "" \
  --progress \
  --write-h5ad
```

Each run writes a manifest under `results/evaluation/`. Passing `--write-h5ad`
updates the chosen `.h5ad` atomically via a temporary file.

## avgBIO

`compute_avgbio.py` computes the same scIB-style benchmark used in the
notebook. The default embedding keys live in `util/avgbio_presets.py` so the
benchmark set is easy to edit.

The official runner uses full-cell silhouette metrics to keep results identical
to the notebook. Because that step is memory-heavy, the preset runs use one
embedding worker by default. Increase `--n-jobs` only when you have enough RAM
for several full silhouette calculations at once.

For each embedding and label key it reports:

| Column | Meaning |
|--------|---------|
| `NMI_cluster/label` | scIB NMI between labels and NMI-optimized Leiden clusters |
| `ARI_cluster/label` | scIB ARI between labels and the same Leiden clusters |
| `ASW_label` | scIB label silhouette on the embedding |
| `ASW_label/batch` | scIB batch silhouette within labels |
| `graph_conn` | scIB graph connectivity |
| `avg_bio` | Mean of `NMI_cluster/label`, `ARI_cluster/label`, and `ASW_label` |
| `avg_batch` | Mean of `graph_conn` and `ASW_label/batch` |
| `overall_score` | `0.6 * avg_bio + 0.4 * avg_batch` |

Run all default Dictyostelium and C. elegans avgBIO jobs:

```bash
bash code/evaluation/run_evaluation_locally.sh \
  code/evaluation/compute_avgbio.py
```

Run one dataset and one label key:

```bash
bash code/evaluation/run_evaluation_locally.sh \
  code/evaluation/compute_avgbio.py \
  --dataset dicty \
  --label-keys cell_type
```

Outputs are written to:

```text
results/biological_benchmarks_dicty/
results/biological_benchmarks_elegans/
```

Each label key gets `<label>.json`, matching the notebook helper behavior.
The runner also writes `results/evaluation/avgBIO_run_summary.json`.
