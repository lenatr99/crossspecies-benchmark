"""Shared helpers for foundation-model cell embedding pipelines.

Model-agnostic ortholog/tier handling used by Geneformer and any other
foundation model that needs to:

  - select organism genes whose human-symbol ortholog falls into a chosen
    set of tier columns from ``adata.var``,
  - map those human symbols to a target ID space (Ensembl for Geneformer,
    other vocabularies for other models) via a model-specific mapper, and
  - collapse columns sharing a target ID via one sparse matmul, so the
    downstream tokenizer never has to do its own duplicate-collapse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def valid_symbol(value: object) -> bool:
    text = str(value).strip()
    return text not in {"", "nan", "None", "No_ortholog", "<NA>"}


def selected_symbols(var: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Pick the first non-empty tier symbol per organism gene."""
    missing = [c for c in columns if c not in var.columns]
    if missing:
        raise ValueError(f"Missing ortholog tier column(s): {missing}")
    out = pd.Series("", index=var.index, dtype=object)
    for col in columns:
        vals = var[col].astype(str).str.strip()
        fill = out.eq("") & vals.map(valid_symbol)
        out.loc[fill] = vals.loc[fill]
    return out


def collapse_duplicate_columns(
    adata: ad.AnnData, key: str
) -> tuple[ad.AnnData, int, int]:
    """Sum columns sharing the same value in ``adata.var[key]``.

    Performs one sparse matmul ``X @ M`` where ``M`` is a (G x U) binary
    indicator. Returns ``(collapsed_adata, n_unique, n_collapsed_groups)``.
    Bypasses Geneformer's chunked groupby and any equivalent slow path in
    other tokenizers.
    """
    ids = adata.var[key].astype(str).to_numpy()
    unique_ids, first_idx, inverse = np.unique(
        ids, return_index=True, return_inverse=True
    )
    n_genes = ids.shape[0]
    n_unique = unique_ids.shape[0]
    n_collapsed = int(n_genes - n_unique)
    if n_collapsed == 0:
        return adata, n_genes, 0

    indicator = sp.csr_matrix(
        (np.ones(n_genes, dtype=np.float32), (np.arange(n_genes), inverse)),
        shape=(n_genes, n_unique),
    )
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X_collapsed = (X @ indicator).tocsr()

    new_var = adata.var.iloc[first_idx].copy()
    new_var[key] = unique_ids
    # Avoid AnnData write_h5ad conflicts when the source var index name is also
    # present as a column, e.g. C. elegans uses index.name == "gene_short_name".
    new_var.index = pd.Index(unique_ids, name=None)

    return (
        ad.AnnData(X=X_collapsed, obs=adata.obs.copy(), var=new_var),
        n_unique,
        n_collapsed,
    )


def prepare_ortholog_h5ad(
    *,
    source_path: Path,
    output_path: Path,
    tier_columns: tuple[str, ...],
    id_mapper: Callable[[str], str | None],
    id_column: str = "ensembl_id",
    max_cells: int | None = None,
) -> dict[str, int]:
    """Build a temp H5AD restricted to ortholog-mapped, ID-collapsed genes.

    ``id_mapper`` is the model-specific symbol -> ID lookup (e.g.
    ``symbol_to_ensembl.get`` for Geneformer). The returned H5AD has unique
    values in ``var[id_column]`` so downstream tokenizers stay on their fast
    path.
    """
    adata = ad.read_h5ad(source_path)
    if max_cells is not None:
        adata = adata[:max_cells, :].copy()

    symbols = selected_symbols(adata.var, tier_columns)
    ids = symbols.map(
        lambda x: id_mapper(str(x).strip()) if valid_symbol(x) else None
    )
    usable = ids.notna()
    if int(usable.sum()) == 0:
        raise ValueError(f"No mappable genes for columns {tier_columns}")

    tmp = adata[:, usable.to_numpy()].copy()
    tmp.var[id_column] = ids.loc[usable].astype(str).to_numpy()
    collapsed_col = f"{id_column}_collapsed"
    if collapsed_col in tmp.var.columns:
        tmp.var = tmp.var.drop(columns=[collapsed_col])

    if "n_counts" not in tmp.obs.columns:
        if "total_counts" not in tmp.obs.columns:
            raise ValueError(
                f"{source_path} missing obs['n_counts'] and obs['total_counts']"
            )
        tmp.obs["n_counts"] = tmp.obs["total_counts"]
    if "cell_id" not in tmp.obs.columns:
        tmp.obs["cell_id"] = tmp.obs_names.astype(str)

    n_pre_collapse = int(tmp.n_vars)
    tmp, n_unique_ids, n_collapsed = collapse_duplicate_columns(tmp, id_column)
    tmp.var.index.name = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    tmp.write_h5ad(output_path)

    return {
        "n_cells": int(tmp.n_obs),
        "n_source_genes": int(adata.n_vars),
        "n_selected_ortholog_symbols": int(symbols.map(valid_symbol).sum()),
        "n_mapped_ids": n_pre_collapse,
        "n_unique_ids": n_unique_ids,
        "n_collapsed_paralog_rows": n_collapsed,
    }
