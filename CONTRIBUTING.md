# Contributing

Thank you for helping improve the cross-species single-cell foundation-model
benchmark. Contributions are especially welcome when they improve
reproducibility, add a well-documented organism, or correct a scientific or
software issue.

## Before opening a change

- Search existing issues and pull requests to avoid duplicate work.
- For a new species or a change to a benchmark definition, open an issue first
  and describe the biological question, source data, expected gene identifiers,
  labels and developmental coordinate.
- Keep generated datasets, checkpoints, caches and other large files out of Git.
  Add compact manifests, checksums and source metadata instead.
- Do not change the headline embedding set in multiple places. Edit
  `code/embedding_sets.py`; its consumers import the canonical definitions.

## Local setup

From the repository root:

```bash
python -m pip install -e .
xspecies --help
xspecies workflow
```

Install `mamba` or `micromamba` before running data, embedding or evaluation
steps. The wrappers create their own pinned environments; do not add the heavy
model dependencies to the top-level package.

## Making a contribution

1. Create a focused branch.
2. Make the smallest change that solves the problem.
3. Update the nearest README and any affected command help.
4. Record data sources, model checkpoints, parameters and checksums in a
   manifest when applicable.
5. Run the relevant smoke checks before opening a pull request.

For a new organism, follow
[`docs/add_new_dataset.md`](docs/add_new_dataset.md).

## Checks

The lightweight repository checks do not download datasets or models:

```bash
python -m compileall -q crossspecies_cli.py code
bash code/run.sh --help
bash code/run.sh workflow
RUN_DRY=1 bash code/run.sh prepare-data --dataset dicty
RUN_DRY=1 bash code/run.sh ortholog-annotate --dataset dicty
```

For a changed embedding pipeline, use its non-destructive smoke-test mode after
preparing the relevant H5AD. For example:

```bash
xspecies embed pca \
  --dataset dicty \
  --variants full \
  --max-cells 256 \
  --overwrite
```

Run full-atlas jobs only when the scientific change requires them. Report the
commands used and point reviewers to the resulting manifest rather than
committing large generated arrays.

## Pull requests

A useful pull request description includes:

- the problem and intended behavior;
- the exact commands used for validation;
- any change to data, model or checkpoint provenance;
- whether existing H5AD keys or result schemas change;
- manuscript figures, tables or claims affected by the change.

Avoid mixing formatting-only edits with scientific or numerical changes.
