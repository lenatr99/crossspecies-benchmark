"""Pseudotime trajectory conservation metrics.

Benchmarks how well a diffusion pseudotime (DPT) ordering computed on each
embedding recovers the known ground-truth developmental time. Mirrors the
design of :mod:`util.benchmark_embeddings`:

- per-root and root-consensus point estimates (Spearman, Lin's CCC, isotonic
  Lin's CCC),
- a **paired bootstrap** (same cell indices + root draws for every embedding
  per iteration) for confidence intervals, and
- pairwise significance via bootstrap CIs on the paired differences.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from tqdm.auto import tqdm


@contextmanager
def suppress_output():
    """Suppress stdout, stderr, and warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr


def lins_concordance(y_true, y_pred):
    """Lin's concordance correlation coefficient (ρ_c). Returns [-1, 1]; 1 = perfect."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(ddof=1), y_pred.var(ddof=1)
    cov = np.cov(y_true, y_pred, ddof=1)[0, 1]
    return 2 * cov / (var_t + var_p + (mean_t - mean_p) ** 2)


def _spearman(a, b):
    r, _ = spearmanr(a, b, nan_policy="omit")
    return r


def _lins_isotonic(y, pt):
    """Lin's concordance after monotonic isotonic calibration."""
    y, pt = np.asarray(y, float), np.asarray(pt, float)
    if pt.size == 0 or np.nanmax(pt) == np.nanmin(pt):
        return np.nan
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    return lins_concordance(y, ir.fit_transform(pt, y))


def _flip_if_needed(y, pt):
    """Flip pseudotime if negatively correlated with ground truth."""
    if _spearman(y, pt) < 0:
        pt = pt.max() - pt
    return pt


def _stratified_sample(groups, rng):
    """Bootstrap indices stratified by groups."""
    groups = np.asarray(groups)
    idx = np.arange(groups.size)
    return np.concatenate([
        rng.choice(idx[groups == g], size=(groups == g).sum(), replace=True)
        for g in np.unique(groups)
    ])


def _normalize_probs(scores):
    """Normalize scores to probabilities."""
    s = np.asarray(scores, float)
    s = s - s.min() + 1e-6
    p = s / s.sum()
    return p if np.isfinite(p).all() and p.sum() > 0 else np.ones_like(s) / s.size


def _nanaverage(values, weights=None):
    """Weighted mean that ignores non-finite entries.

    A single degenerate root (disconnected diffusion graph -> constant
    pseudotime -> NaN Spearman) would otherwise poison the whole average
    and turn an embedding's score into NaN. Masking keeps the estimate
    over the roots that did produce a finite metric.
    """
    values = np.asarray(values, float)
    mask = np.isfinite(values)
    if not mask.any():
        return np.nan
    if weights is None:
        return float(np.mean(values[mask]))
    weights = np.asarray(weights, float)[mask]
    if weights.sum() <= 0:
        return float(np.mean(values[mask]))
    return float(np.average(values[mask], weights=weights))


def _consensus_pseudotime(pt_matrix, y_min, y_max, root_weights=None):
    """Compute the final consensus pseudotime stored for plotting/downstream use."""
    pt_matrix = np.asarray(pt_matrix, float)
    if root_weights is None:
        pt_consensus = pt_matrix.mean(axis=0)
    else:
        pt_consensus = np.average(pt_matrix, axis=0, weights=root_weights)

    pt_min, pt_max = pt_consensus.min(), pt_consensus.max()
    if pt_max > pt_min:
        pt_consensus = (pt_consensus - pt_min) / (pt_max - pt_min)
    else:
        pt_consensus = np.zeros_like(pt_consensus)

    pt_scaled = y_min + pt_consensus * (y_max - y_min)
    return pt_consensus, pt_scaled


def _precompute_pseudotimes(adata_method, cell_subset, candidate_roots, verbose=True):
    """Compute pseudotime for all candidate roots. Returns (n_roots, n_cells) matrix."""
    idx_subset = adata_method.obs_names.get_indexer(cell_subset)
    y_all = adata_method.obs.loc[cell_subset, "real_time"].to_numpy()

    if "X_diffmap" not in adata_method.obsm:
        sc.tl.diffmap(adata_method)

    pts = np.empty((len(candidate_roots), idx_subset.size), dtype=np.float32)

    iterator = enumerate(candidate_roots)
    if verbose:
        iterator = tqdm(list(iterator), desc="  DPT roots", leave=False)

    for i, rp in iterator:
        adata_method.uns["iroot"] = int(rp)
        sc.tl.dpt(adata_method)
        pt = adata_method.obs["dpt_pseudotime"].to_numpy()[idx_subset]
        pt = np.where(np.isinf(pt), np.nan, pt)
        pt = np.nan_to_num(pt, nan=np.nanmax(pt))
        pt = _flip_if_needed(y_all, pt)
        pt_min, pt_max = pt.min(), pt.max()
        pts[i] = (pt - pt_min) / (pt_max - pt_min) if pt_max > pt_min else np.zeros_like(pt)
    return pts


def _bootstrap_ci(
    y,
    pt_matrix,
    groups,
    root_probs,
    n_boot=1000,
    k_roots=50,
    alpha=0.05,
    seed=0,
    verbose=True,
    use_isotonic=True,
    use_consensus=True,
    bootstrap_consensus_roots=True,
    consensus_pt_matrix=None,
    consensus_pt_scaled=None,
):
    """Bootstrap CI sampling both cells and roots.  Returns CIs AND per-iteration values."""
    n_roots, n_cells = pt_matrix.shape
    rng = np.random.default_rng(seed)
    stats_s, stats_c = np.empty(n_boot), np.empty(n_boot)
    if use_consensus:
        stats_s_cons = np.empty(n_boot)
    if use_isotonic:
        stats_c_iso = np.empty(n_boot)
    if use_consensus:
        stats_c_cons = np.empty(n_boot)

    y_min, y_max = np.nanmin(y), np.nanmax(y)

    iterator = range(n_boot)
    if verbose:
        iterator = tqdm(iterator, desc="  Bootstrap", leave=False)

    for i in iterator:
        idx = _stratified_sample(groups, rng) if groups is not None else rng.integers(0, n_cells, n_cells)
        rdraw = rng.choice(n_roots, size=k_roots, replace=True, p=root_probs)
        stats_s[i] = np.nanmean([_spearman(y[idx], pt_matrix[r, idx]) for r in rdraw])
        stats_c[i] = np.nanmean([lins_concordance(y[idx], pt_matrix[r, idx]) for r in rdraw])
        if use_isotonic:
            stats_c_iso[i] = np.nanmean([_lins_isotonic(y[idx], pt_matrix[r, idx]) for r in rdraw])
        if use_consensus:
            if bootstrap_consensus_roots:
                if consensus_pt_matrix is None:
                    raise ValueError("consensus_pt_matrix is required when bootstrap_consensus_roots=True")
                _, pt_cons_scaled = _consensus_pseudotime(consensus_pt_matrix[rdraw], y_min, y_max)
            else:
                if consensus_pt_scaled is None:
                    raise ValueError("consensus_pt_scaled is required when bootstrap_consensus_roots=False")
                pt_cons_scaled = consensus_pt_scaled
            stats_s_cons[i] = _spearman(y[idx], pt_cons_scaled[idx])
            stats_c_cons[i] = lins_concordance(y[idx], pt_cons_scaled[idx])

    cis = (
        (np.quantile(stats_s, alpha / 2), np.quantile(stats_s, 1 - alpha / 2)),
        (np.quantile(stats_c, alpha / 2), np.quantile(stats_c, 1 - alpha / 2)),
    )
    boot_values = {"spearman": stats_s, "lins_ccc": stats_c}
    if use_isotonic:
        cis = cis + ((np.quantile(stats_c_iso, alpha / 2), np.quantile(stats_c_iso, 1 - alpha / 2)),)
        boot_values["lins_ccc_isotonic"] = stats_c_iso
    if use_consensus:
        cis = cis + (
            (np.quantile(stats_s_cons, alpha / 2), np.quantile(stats_s_cons, 1 - alpha / 2)),
            (np.quantile(stats_c_cons, alpha / 2), np.quantile(stats_c_cons, 1 - alpha / 2)),
        )
        boot_values["spearman_consensus"] = stats_s_cons
        boot_values["lins_ccc_consensus"] = stats_c_cons
    return cis, boot_values


def _paired_bootstrap_multi(
    y,
    pt_matrices,
    groups,
    root_probs,
    consensus_pt_matrices=None,
    consensus_pt_scaled=None,
    n_boot=1000, k_roots=50, seed=0, verbose=True,
    bootstrap_consensus_roots=True,
):
    """
    Paired bootstrap across multiple embeddings.

    All embeddings are evaluated on the SAME cell indices and root draws
    per iteration, enabling valid pairwise comparison.

    Parameters
    ----------
    y : np.ndarray
        Ground truth time (n_cells,).
    pt_matrices : dict[str, np.ndarray]
        {embed_name: scaled per-root pseudotime matrix of shape (n_roots, n_cells)}.
    groups : np.ndarray
        Stratification groups for bootstrap.
    root_probs : np.ndarray
        Weights for root sampling.
    n_boot : int
        Number of bootstrap iterations.
    k_roots : int
        Roots per bootstrap replicate.
    seed : int
        Random seed.
    verbose : bool
        Show progress bar.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        {embed: {"spearman": array(n_boot), "lins_ccc": ..., "lins_ccc_isotonic": ..., "lins_ccc_consensus": ...}}
    """
    emb_names = list(pt_matrices.keys())
    n_roots = next(iter(pt_matrices.values())).shape[0]
    n_cells = y.size
    rng = np.random.default_rng(seed)

    boot_values = {
        emb: {
            "spearman": np.empty(n_boot),
            "lins_ccc": np.empty(n_boot),
            "lins_ccc_isotonic": np.empty(n_boot),
            "spearman_consensus": np.empty(n_boot),
            "lins_ccc_consensus": np.empty(n_boot),
        }
        for emb in emb_names
    }

    iterator = range(n_boot)
    if verbose:
        iterator = tqdm(iterator, desc="Paired bootstrap", leave=True)

    for i in iterator:
        # SAME cell indices and root draws for all embeddings
        idx = _stratified_sample(groups, rng) if groups is not None else rng.integers(0, n_cells, n_cells)
        rdraw = rng.choice(n_roots, size=k_roots, replace=True, p=root_probs)

        for emb in emb_names:
            pt_m = pt_matrices[emb]
            boot_values[emb]["spearman"][i] = np.nanmean([
                _spearman(y[idx], pt_m[r, idx]) for r in rdraw
            ])
            boot_values[emb]["lins_ccc"][i] = np.nanmean([
                lins_concordance(y[idx], pt_m[r, idx]) for r in rdraw
            ])
            boot_values[emb]["lins_ccc_isotonic"][i] = np.nanmean([
                _lins_isotonic(y[idx], pt_m[r, idx]) for r in rdraw
            ])
            if bootstrap_consensus_roots:
                if consensus_pt_matrices is None:
                    raise ValueError("consensus_pt_matrices is required when bootstrap_consensus_roots=True")
                _, pt_cons_scaled = _consensus_pseudotime(
                    consensus_pt_matrices[emb][rdraw], np.nanmin(y), np.nanmax(y)
                )
            else:
                if consensus_pt_scaled is None:
                    raise ValueError("consensus_pt_scaled is required when bootstrap_consensus_roots=False")
                pt_cons_scaled = consensus_pt_scaled[emb]
            boot_values[emb]["spearman_consensus"][i] = _spearman(
                y[idx], pt_cons_scaled[idx]
            )
            boot_values[emb]["lins_ccc_consensus"][i] = lins_concordance(
                y[idx], pt_cons_scaled[idx]
            )

    return boot_values


def compute_pseudotime_significance(
    boot_values: dict[str, dict[str, np.ndarray]],
    metric: str = "spearman",
    alpha: float = 0.05,
) -> dict:
    """
    Compute pairwise significance from paired bootstrap values.

    Uses bootstrap confidence interval on the difference: if the
    (1-alpha) CI on (A - B) excludes zero, the difference is significant.
    This is more interpretable than a Wilcoxon test on bootstrap pairs,
    which is massively overpowered and gives absurdly small p-values
    even for meaningless differences.

    Parameters
    ----------
    boot_values : dict
        Output of _paired_bootstrap_multi.
    metric : str
        Which metric to compare ("spearman", "lins_ccc", "lins_ccc_isotonic").
    alpha : float
        Significance level for CIs.

    Returns
    -------
    dict with "pairs" and "metric".
    """
    emb_names = list(boot_values.keys())
    pairs = []
    for i in range(len(emb_names)):
        for j in range(i + 1, len(emb_names)):
            a, b = emb_names[i], emb_names[j]
            vals_a = boot_values[a][metric]
            vals_b = boot_values[b][metric]
            diff = vals_a - vals_b
            n_boot = len(diff)
            mean_diff = float(np.mean(diff))
            ci_lo = float(np.quantile(diff, alpha / 2))
            ci_hi = float(np.quantile(diff, 1 - alpha / 2))

            # Bootstrap p-value: fraction of bootstrap diffs on the wrong
            # side of zero (two-sided).  When 0 out of n_boot cross zero,
            # report the conservative upper bound 2/n_boot rather than 0.
            if mean_diff > 0:
                p_value = float(2 * np.mean(diff <= 0))
            elif mean_diff < 0:
                p_value = float(2 * np.mean(diff >= 0))
            else:
                p_value = 1.0
            p_value = min(p_value, 1.0)
            if p_value == 0.0:
                p_value = 2.0 / n_boot  # conservative upper bound

            significant = (ci_lo > 0) or (ci_hi < 0)

            pairs.append({
                "emb_a": a,
                "emb_b": b,
                "mean_diff": mean_diff,
                "diff_ci": [ci_lo, ci_hi],
                "p_value": p_value,
                "significant": significant,
            })

    return {"metric": metric, "pairs": pairs}


def sample_roots(adata, time_key: str, n_candidates: int = 500, method: str = "uniform", seed: int = 0):
    """
    Sample candidate root cells from earliest timepoint.

    Returns
    -------
    tuple: (candidate_roots, candidate_scores)
    """
    rng = np.random.default_rng(seed)
    time_vals = adata.obs[time_key]

    # Handle categorical/string time values
    if hasattr(time_vals, 'cat') or time_vals.dtype == object:
        # Try to extract numeric values from strings like "00hr", "0", etc.
        try:
            numeric_times = time_vals.astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
            earliest_numeric = numeric_times.min()
            zero_idx = np.where(numeric_times == earliest_numeric)[0]
        except (ValueError, TypeError):
            # Fall back to first category or first unique value
            if hasattr(time_vals, 'cat'):
                earliest = time_vals.cat.categories[0]
            else:
                earliest = sorted(time_vals.unique())[0]
            zero_idx = np.where(time_vals == earliest)[0]
    else:
        earliest = time_vals.min()
        zero_idx = np.where(time_vals == earliest)[0]

    if len(zero_idx) == 0:
        raise ValueError("No cells found at earliest timepoint")

    if method == "uniform":
        n = min(n_candidates, zero_idx.size)
        roots = rng.choice(zero_idx, size=n, replace=False).tolist()
    elif method == "stratified":
        adata_temp = adata[zero_idx].copy()
        sc.pp.normalize_total(adata_temp)
        sc.pp.log1p(adata_temp)
        sc.pp.scale(adata_temp)
        sc.tl.pca(adata_temp)
        sc.pp.neighbors(adata_temp, use_rep="X_pca")
        sc.tl.leiden(adata_temp, resolution=1.0)
        roots = []
        for _, sub in adata_temp.obs.groupby("leiden"):
            n = max(1, int(len(sub) / len(adata_temp) * n_candidates))
            sampled = rng.choice(sub.index, size=min(n, len(sub)), replace=False)
            roots.extend([adata.obs_names.get_loc(c) for c in sampled])
    else:
        raise ValueError(f"Unknown method: {method}")

    return roots, np.ones(len(roots))


def pseudotime_eval(
    adata,
    embed_key: str,
    time_key: str,
    candidate_roots: list[int],
    candidate_scores: np.ndarray,
    n_boot: int = 1000,
    k_roots: int = 50,
    alpha: float = 0.05,
    verbose: bool = True,
    store_pseudotime: bool = True,
    bootstrap_consensus_roots: bool = True,
) -> dict:
    """
    Evaluate pseudotime trajectory conservation for a single embedding.

    Parameters
    ----------
    adata : AnnData
        Must have time_key column in obs (numeric ground truth time).
    embed_key : str
        Key in adata.obsm for the embedding.
    time_key : str
        Column in adata.obs with ground truth time (numeric).
    candidate_roots : list[int]
        Indices of candidate root cells.
    candidate_scores : np.ndarray
        Weights for each root.
    n_boot : int
        Number of bootstrap iterations.
    k_roots : int
        Roots to sample per bootstrap.
    alpha : float
        Significance level for CIs.
    verbose : bool
        Show internal scanpy output.
    store_pseudotime : bool
        If True, store computed pseudotime in adata.obs.
    bootstrap_consensus_roots : bool
        If True, consensus Lin CCC uncertainty also resamples roots.
        If False, bootstrap only cells while keeping the final consensus
        pseudotime fixed.

    Returns
    -------
    dict with per-root and consensus pseudotime metrics and their CIs.
    """
    adata_method = adata.copy()

    # Convert time to numeric
    time_vals = adata_method.obs[time_key]
    if hasattr(time_vals, 'cat') or time_vals.dtype == object:
        # Extract numeric from strings like "00hr", "0h", etc.
        adata_method.obs["real_time"] = (
            time_vals.astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
        )
    else:
        adata_method.obs["real_time"] = time_vals.astype(float)

    if verbose:
        sc.pp.neighbors(adata_method, use_rep=embed_key)
    else:
        with suppress_output():
            sc.pp.neighbors(adata_method, use_rep=embed_key)

    cell_subset = adata_method.obs.index[adata_method.obs["real_time"].notnull()]
    root_probs = _normalize_probs(candidate_scores)

    if verbose:
        pt_matrix = _precompute_pseudotimes(adata_method, cell_subset, candidate_roots, verbose=True)
    else:
        with suppress_output():
            pt_matrix = _precompute_pseudotimes(adata_method, cell_subset, candidate_roots, verbose=False)

    y = adata_method.obs.loc[cell_subset, "real_time"].to_numpy()
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    pt_scaled = y_min + pt_matrix * (y_max - y_min)

    # Point estimates (nan-aware so a single degenerate root can't poison them)
    spearman_val = _nanaverage([_spearman(y, pt) for pt in pt_scaled], weights=root_probs)
    lins_val = _nanaverage([lins_concordance(y, pt) for pt in pt_scaled], weights=root_probs)
    lins_iso_val = _nanaverage([_lins_isotonic(y, pt) for pt in pt_matrix], weights=root_probs)
    pt_weighted, pt_scaled_weighted = _consensus_pseudotime(pt_matrix, y_min, y_max, root_probs)
    spearman_consensus_val = _spearman(y, pt_scaled_weighted)
    lins_consensus_val = lins_concordance(y, pt_scaled_weighted)

    # Store weighted-average pseudotime in original adata
    if store_pseudotime:
        adata.obs[f"{embed_key}_dpt_pseudotime"] = np.nan
        adata.obs.loc[cell_subset, f"{embed_key}_dpt_pseudotime"] = pt_scaled_weighted

        # Also store isotonic-calibrated pseudotime
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        pt_isotonic = ir.fit_transform(pt_weighted, y)
        adata.obs[f"{embed_key}_dpt_pseudotime_isotonic"] = np.nan
        adata.obs.loc[cell_subset, f"{embed_key}_dpt_pseudotime_isotonic"] = pt_isotonic

    # Bootstrap CIs - use raw time if discrete (<20 unique), otherwise bin
    time_vals = adata_method.obs.loc[cell_subset, "real_time"]
    if time_vals.nunique() <= 20:
        # Discrete time - use raw values
        groups = time_vals.to_numpy()
    else:
        # Continuous time - bin into quantiles
        groups = pd.qcut(time_vals, q=min(15, time_vals.nunique()), duplicates="drop").cat.codes.to_numpy()

    (
        (s_lo, s_hi),
        (c_lo, c_hi),
        (c_iso_lo, c_iso_hi),
        (s_cons_lo, s_cons_hi),
        (c_cons_lo, c_cons_hi),
    ), boot_values = _bootstrap_ci(
        y,
        pt_scaled,
        groups,
        root_probs,
        n_boot,
        min(k_roots, len(candidate_roots)),
        alpha,
        verbose=verbose,
        use_isotonic=True,
        use_consensus=True,
        bootstrap_consensus_roots=bootstrap_consensus_roots,
        consensus_pt_matrix=pt_matrix,
        consensus_pt_scaled=pt_scaled_weighted,
    )

    return {
        "spearman": np.clip(spearman_val, -1, 1),
        "spearman_ci": (np.clip(s_lo, -1, 1), np.clip(s_hi, -1, 1)),
        "spearman_consensus": np.clip(spearman_consensus_val, -1, 1),
        "spearman_consensus_ci": (np.clip(s_cons_lo, -1, 1), np.clip(s_cons_hi, -1, 1)),
        "lins_ccc": np.clip(lins_val, -1, 1),
        "lins_ccc_ci": (np.clip(c_lo, -1, 1), np.clip(c_hi, -1, 1)),
        "lins_ccc_isotonic": np.clip(lins_iso_val, -1, 1),
        "lins_ccc_isotonic_ci": (np.clip(c_iso_lo, -1, 1), np.clip(c_iso_hi, -1, 1)),
        "lins_ccc_consensus": np.clip(lins_consensus_val, -1, 1),
        "lins_ccc_consensus_ci": (np.clip(c_cons_lo, -1, 1), np.clip(c_cons_hi, -1, 1)),
        "boot_values": boot_values,
    }


def benchmark_pseudotime(
    data: str | Path,
    embeddings: list[str],
    time_key: str,
    n_candidates: int = 500,
    root_method: str = "uniform",
    n_boot: int = 1000,
    k_roots: int = 50,
    alpha: float = 0.05,
    verbose: bool = True,
    save_dir: str | Path | None = None,
    write_h5ad: bool = False,
    bootstrap_consensus_roots: bool = True,
) -> dict[str, dict]:
    """
    Benchmark pseudotime trajectory conservation across multiple embeddings.

    Uses a **paired bootstrap** design: all embeddings are evaluated on the
    same cell indices and root draws per iteration, enabling valid pairwise
    significance testing via bootstrap confidence intervals on differences.

    Parameters
    ----------
    data : str | Path
        Path to .h5ad file.
    embeddings : list[str]
        List of embedding keys in adata.obsm.
    time_key : str
        Column in adata.obs with numeric ground truth time.
    n_candidates : int
        Number of candidate roots to sample.
    root_method : str
        "uniform" or "stratified" root sampling.
    n_boot : int
        Bootstrap iterations.
    k_roots : int
        Roots per bootstrap replicate.
    alpha : float
        CI alpha level.
    verbose : bool
        Show progress and internal output.
    save_dir : str | Path | None
        If provided, save results to {save_dir}/pseudotime.json.
    write_h5ad : bool
        If True, write updated adata back to the same path.
    bootstrap_consensus_roots : bool
        If True, consensus Lin CCC uncertainty and significance also
        resample roots. If False, they bootstrap only cells while keeping
        each embedding's final consensus pseudotime fixed.

    Returns
    -------
    dict[str, dict]
        Results keyed by embedding name, including point estimates, CIs,
        bootstrap values, and pairwise significance.
    """
    print(f"Loading {data}...")
    adata = sc.read(data)
    print(f"Sampling {n_candidates} candidate roots ({root_method})...")
    candidate_roots, candidate_scores = sample_roots(
        adata, time_key, n_candidates, root_method
    )
    print(f"  Found {len(candidate_roots)} roots from earliest timepoint")
    root_probs = _normalize_probs(candidate_scores)

    # ---- Ground-truth time (identical for every embedding) ----
    # Computed once on `adata` and reused everywhere so the per-embedding
    # loop and the paired bootstrap operate on the exact same cells in the
    # same order. (Previously these were recomputed per embedding and the
    # bootstrap silently relied on the last loop iteration's values.)
    time_vals = adata.obs[time_key]
    if hasattr(time_vals, "cat") or time_vals.dtype == object:
        adata.obs["real_time"] = (
            time_vals.astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
        )
    else:
        adata.obs["real_time"] = time_vals.astype(float)

    cell_subset = adata.obs.index[adata.obs["real_time"].notnull()]
    y = adata.obs.loc[cell_subset, "real_time"].to_numpy()
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    print(f"  {len(cell_subset)}/{adata.n_obs} cells have a numeric {time_key!r} value")
    if len(cell_subset) == 0:
        raise ValueError(f"No cells have a numeric {time_key!r} value")

    # Bootstrap stratification groups: raw values if discrete, else quantile bins.
    y_series = pd.Series(y)
    if y_series.nunique() <= 20:
        groups = y
    else:
        groups = pd.qcut(
            y_series, q=min(15, y_series.nunique()), duplicates="drop"
        ).cat.codes.to_numpy()

    # ---- Phase 1: Precompute pseudotime matrices for all embeddings ----
    # (expensive — one neighbor graph + DPT per root per embedding)
    pt_matrices = {}     # {emb: (n_roots, n_cells) array}
    pt_scaled_all = {}   # scaled versions for per-root metrics
    pt_consensus_scaled = {}  # final stored consensus pseudotime per embedding
    results = {}

    for i, emb in enumerate(embeddings, 1):
        print(f"\n[{i}/{len(embeddings)}] {emb}: computing DPT for {len(candidate_roots)} roots...")
        adata_method = adata.copy()  # inherits the shared "real_time" column

        if verbose:
            sc.pp.neighbors(adata_method, use_rep=emb)
        else:
            with suppress_output():
                sc.pp.neighbors(adata_method, use_rep=emb)

        if verbose:
            pt_matrix = _precompute_pseudotimes(adata_method, cell_subset, candidate_roots, verbose=True)
        else:
            with suppress_output():
                pt_matrix = _precompute_pseudotimes(adata_method, cell_subset, candidate_roots, verbose=False)

        pt_scaled = y_min + pt_matrix * (y_max - y_min)

        pt_matrices[emb] = pt_matrix
        pt_scaled_all[emb] = pt_scaled

        # Point estimates (nan-aware so a single degenerate root can't poison them)
        spearman_val = _nanaverage([_spearman(y, pt) for pt in pt_scaled], weights=root_probs)
        lins_val = _nanaverage([lins_concordance(y, pt) for pt in pt_scaled], weights=root_probs)
        lins_iso_val = _nanaverage([_lins_isotonic(y, pt) for pt in pt_matrix], weights=root_probs)
        pt_weighted, pt_scaled_weighted = _consensus_pseudotime(pt_matrix, y_min, y_max, root_probs)
        pt_consensus_scaled[emb] = pt_scaled_weighted
        spearman_consensus_val = _spearman(y, pt_scaled_weighted)
        lins_consensus_val = lins_concordance(y, pt_scaled_weighted)

        results[emb] = {
            "spearman": float(np.clip(spearman_val, -1, 1)),
            "spearman_consensus": float(np.clip(spearman_consensus_val, -1, 1)),
            "lins_ccc": float(np.clip(lins_val, -1, 1)),
            "lins_ccc_isotonic": float(np.clip(lins_iso_val, -1, 1)),
            "lins_ccc_consensus": float(np.clip(lins_consensus_val, -1, 1)),
        }
        print(
            f"  spearman={spearman_val:.4f}  spearman_consensus={spearman_consensus_val:.4f}  "
            f"lins={lins_val:.4f}  "
            f"lins_consensus={lins_consensus_val:.4f}  lins_iso={lins_iso_val:.4f}"
        )

        # Store pseudotime in adata
        adata.obs[f"{emb}_dpt_pseudotime"] = np.nan
        adata.obs.loc[cell_subset, f"{emb}_dpt_pseudotime"] = pt_scaled_weighted

        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        pt_isotonic = ir.fit_transform(pt_weighted, y)
        adata.obs[f"{emb}_dpt_pseudotime_isotonic"] = np.nan
        adata.obs.loc[cell_subset, f"{emb}_dpt_pseudotime_isotonic"] = pt_isotonic

    # ---- Phase 2: Paired bootstrap across ALL embeddings ----
    # Same cell indices + root draws for every embedding in each iteration
    print(f"\nRunning paired bootstrap ({n_boot} iterations) across {len(embeddings)} embeddings...")
    boot_values = _paired_bootstrap_multi(
        y,
        pt_scaled_all,
        groups,
        root_probs,
        consensus_pt_matrices=pt_matrices,
        consensus_pt_scaled=pt_consensus_scaled,
        n_boot=n_boot, k_roots=min(k_roots, len(candidate_roots)),
        seed=0, verbose=verbose,
        bootstrap_consensus_roots=bootstrap_consensus_roots,
    )

    # Attach CIs and boot values to results
    for emb in embeddings:
        bv = boot_values[emb]
        for metric in ("spearman", "spearman_consensus", "lins_ccc", "lins_ccc_isotonic", "lins_ccc_consensus"):
            lo = float(np.clip(np.quantile(bv[metric], alpha / 2), -1, 1))
            hi = float(np.clip(np.quantile(bv[metric], 1 - alpha / 2), -1, 1))
            results[emb][f"{metric}_ci"] = [lo, hi]
            results[emb][f"{metric}_std"] = float(np.std(bv[metric]))
        results[emb]["boot_values"] = {k: v.tolist() for k, v in bv.items()}

    significance = {}

    # ---- Phase 3: Pairwise significance ----
    # Uses bootstrap CI on paired differences (not Wilcoxon, which is
    # massively overpowered on bootstrap samples and gives ~0 p-values)
    print("\nComputing pairwise significance (bootstrap CI on differences)...")

    for metric in ("spearman", "spearman_consensus", "lins_ccc", "lins_ccc_isotonic", "lins_ccc_consensus"):
        sig = compute_pseudotime_significance(boot_values, metric=metric, alpha=alpha)
        significance[metric] = sig

    # ---- Summary ----
    ranked = sorted(results.items(), key=lambda x: x[1]["spearman"], reverse=True)
    print(
        f"\n{'Rank':<5}{'Embedding':<40}{'Spearman':<16}"
        f"{'Sp Cons':<16}"
        f"{'Lin CCC':<16}{'Lin Cons':<16}{'Lin ISO':<16}"
    )
    print("-" * 122)
    for i, (name, r) in enumerate(ranked, 1):
        print(f"{i:<5}{name:<40}{r['spearman']:.4f}±{r['spearman_std']:.3f}  "
              f"{r['spearman_consensus']:.4f}±{r['spearman_consensus_std']:.3f}  "
              f"{r['lins_ccc']:.4f}±{r['lins_ccc_std']:.3f}  "
              f"{r['lins_ccc_consensus']:.4f}±{r['lins_ccc_consensus_std']:.3f}  "
              f"{r['lins_ccc_isotonic']:.4f}±{r['lins_ccc_isotonic_std']:.3f}")

    # Print significant pairwise differences for main metric
    sig_spearman = significance["spearman"]["pairs"]
    n_sig = sum(1 for p in sig_spearman if p["significant"])
    n_total = len(sig_spearman)
    print(f"\nPairwise comparisons (Spearman): {n_sig}/{n_total} significant at alpha={alpha}")
    # Show non-significant pairs (these are the interesting ones — embeddings that are NOT distinguishable)
    nonsig = [p for p in sig_spearman if not p["significant"]]
    if nonsig:
        print("Non-significant pairs (indistinguishable):")
        for p in nonsig:
            print(f"  {p['emb_a']} vs {p['emb_b']}: diff={p['mean_diff']:.4f}, "
                  f"CI=[{p['diff_ci'][0]:.4f}, {p['diff_ci'][1]:.4f}]")
    else:
        print("All pairs are significantly different.")

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        # Save results (without boot_values to keep JSON small)
        json_results = {}
        for emb, r in results.items():
            json_results[emb] = {k: v for k, v in r.items() if k != "boot_values"}
        # Add significance
        json_results["_significance"] = significance
        with open(save_path / "pseudotime.json", "w") as f:
            json.dump(json_results, f, indent=4)
        print(f"\nSaved to {save_path / 'pseudotime.json'}")

    # Write the pseudotime columns back only after every metric, the
    # bootstrap, and the JSON have succeeded — and do it atomically (write a
    # temp file, then os.replace) so a crash or interrupt can never leave the
    # source H5AD half-written.
    if write_h5ad:
        data_path = Path(data)
        tmp_path = data_path.with_name(data_path.name + ".tmp")
        adata.write(tmp_path)
        os.replace(tmp_path, data_path)
        print(f"Wrote {data_path}")

    return results, significance


# ===========================================================================
# Subsampling-without-replacement + recompute inference (Design A)
# ===========================================================================
#
# The bootstrap path above resamples cells/roots from a pseudotime matrix that
# was computed ONCE on the full data, so it cannot see graph/DPT-reconstruction
# instability. For trajectory methods the field-standard robustness check is to
# subsample cells WITHOUT replacement and RE-RUN the pipeline (neighbor graph +
# diffusion pseudotime) on each subset (Saelens et al. 2019, "stability after
# subsampling"). Subsampling without replacement avoids the duplicate-cell
# pathologies that make bootstrap-with-replacement inappropriate for kNN-graph
# metrics. We report percentile stability intervals across repeats and describe
# methods as "distinguishable" when their paired per-repeat differences exclude
# zero, rather than as a formal Wilcoxon p-value (the ~90%-overlapping repeats
# are pseudo-replicated, so a formal test would be anti-conservative).


def _stratified_subsample(groups, frac, rng):
    """Stratified subsample WITHOUT replacement. Returns sorted integer positions."""
    groups = np.asarray(groups)
    idx = np.arange(groups.size)
    out = []
    for g in np.unique(groups):
        gi = idx[groups == g]
        n = max(1, int(round(frac * gi.size)))
        out.append(rng.choice(gi, size=min(n, gi.size), replace=False))
    return np.sort(np.concatenate(out))


def _metrics_on_subset(adata_sub, embed_key, roots_positions, y_sub, verbose=False):
    """Recompute neighbors + DPT on a cell subset and score against y_sub.

    Returns per-root-weighted Spearman/Lin CCC and their root-consensus
    variants, all as raw values in [-1, 1].
    """
    # Force a fresh graph + diffusion map on this subset (drop anything carried
    # over in obsm from the parent object).
    adata_sub.obsm.pop("X_diffmap", None)
    adata_sub.uns.pop("neighbors", None)

    if verbose:
        sc.pp.neighbors(adata_sub, use_rep=embed_key)
        sc.tl.diffmap(adata_sub)
    else:
        with suppress_output():
            sc.pp.neighbors(adata_sub, use_rep=embed_key)
            sc.tl.diffmap(adata_sub)

    y_min, y_max = np.nanmin(y_sub), np.nanmax(y_sub)
    pts_scaled = np.empty((len(roots_positions), y_sub.size), dtype=np.float64)
    sp_list, cc_list = [], []
    for i, rp in enumerate(roots_positions):
        adata_sub.uns["iroot"] = int(rp)
        if verbose:
            sc.tl.dpt(adata_sub)
        else:
            with suppress_output():
                sc.tl.dpt(adata_sub)
        pt = adata_sub.obs["dpt_pseudotime"].to_numpy()
        pt = np.where(np.isinf(pt), np.nan, pt)
        pt = np.nan_to_num(pt, nan=np.nanmax(pt))
        pt = _flip_if_needed(y_sub, pt)
        pt_min, pt_max = pt.min(), pt.max()
        pt01 = (pt - pt_min) / (pt_max - pt_min) if pt_max > pt_min else np.zeros_like(pt)
        pt_scaled = y_min + pt01 * (y_max - y_min)
        pts_scaled[i] = pt_scaled
        sp_list.append(_spearman(y_sub, pt_scaled))
        cc_list.append(lins_concordance(y_sub, pt_scaled))

    # Root-consensus pseudotime for this subset (weighted mean over roots; equal
    # weights here since roots are drawn uniformly from the earliest state).
    _, pt_cons_scaled = _consensus_pseudotime(pts_scaled, y_min, y_max)
    return {
        "spearman": _nanaverage(sp_list),
        "lins_ccc": _nanaverage(cc_list),
        "spearman_consensus": _spearman(y_sub, pt_cons_scaled),
        "lins_ccc_consensus": lins_concordance(y_sub, pt_cons_scaled),
    }


METRIC_KEYS = ("spearman", "lins_ccc", "spearman_consensus", "lins_ccc_consensus")


def benchmark_pseudotime_subsampling(
    data: str | Path,
    embeddings: list[str],
    time_key: str,
    n_repeats: int = 30,
    subsample_frac: float = 0.9,
    k_roots: int = 20,
    n_candidates: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
    verbose: bool = False,
    save_dir: str | Path | None = None,
) -> dict[str, dict]:
    """Subsampling-without-replacement + recompute inference for pseudotime.

    Point estimates are computed once on the full data (identical design to
    :func:`benchmark_pseudotime`), and robustness is quantified by ``n_repeats``
    paired subsampling repeats that recompute the neighbor graph and DPT on each
    ``subsample_frac`` subset. All embeddings share the same cell subset and root
    draws within a repeat (paired design), enabling per-repeat paired
    differences.
    """
    print(f"Loading {data}...")
    adata = sc.read(data)

    # ---- numeric ground-truth time (shared across embeddings) ----
    time_vals = adata.obs[time_key]
    if hasattr(time_vals, "cat") or time_vals.dtype == object:
        adata.obs["real_time"] = (
            time_vals.astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
        )
    else:
        adata.obs["real_time"] = time_vals.astype(float)

    # Keep only timed cells so positions are stable and shared.
    timed = adata.obs["real_time"].notnull().to_numpy()
    adata = adata[timed].copy()
    y_all = adata.obs["real_time"].to_numpy()
    print(f"  {adata.n_obs} timed cells; earliest state = root pool")

    earliest = np.nanmin(y_all)
    earliest_mask = y_all == earliest

    # Stratification: raw labels if discrete (<=20 unique), else quantile bins.
    y_series = pd.Series(y_all)
    if y_series.nunique() <= 20:
        groups = y_all
    else:
        groups = pd.qcut(
            y_series, q=min(15, y_series.nunique()), duplicates="drop"
        ).cat.codes.to_numpy()

    # ---- Point estimates on FULL data (one graph + DPT per embedding) ----
    print("Computing full-data point estimates...")
    cell_subset = adata.obs_names
    root_pool = np.where(earliest_mask)[0]
    rng = np.random.default_rng(seed)
    full_roots = rng.choice(
        root_pool, size=min(n_candidates, root_pool.size), replace=False
    ).tolist()

    results: dict[str, dict] = {}
    for i, emb in enumerate(embeddings, 1):
        ad = adata.copy()
        if verbose:
            sc.pp.neighbors(ad, use_rep=emb)
        else:
            with suppress_output():
                sc.pp.neighbors(ad, use_rep=emb)
        pt_matrix = _precompute_pseudotimes(ad, cell_subset, full_roots, verbose=verbose)
        y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
        pt_scaled = y_min + pt_matrix * (y_max - y_min)
        _, pt_cons_scaled = _consensus_pseudotime(pt_matrix, y_min, y_max)
        results[emb] = {
            "spearman": float(np.clip(_nanaverage([_spearman(y_all, pt) for pt in pt_scaled]), -1, 1)),
            "lins_ccc": float(np.clip(_nanaverage([lins_concordance(y_all, pt) for pt in pt_scaled]), -1, 1)),
            "spearman_consensus": float(np.clip(_spearman(y_all, pt_cons_scaled), -1, 1)),
            "lins_ccc_consensus": float(np.clip(lins_concordance(y_all, pt_cons_scaled), -1, 1)),
        }
        print(f"  [{i}/{len(embeddings)}] {emb}: spearman={results[emb]['spearman']:.4f}")

    # ---- Subsampling repeats (paired across embeddings) ----
    print(f"\nRunning {n_repeats} subsampling repeats "
          f"(frac={subsample_frac}, {k_roots} roots/repeat, recomputing DPT)...")
    per_repeat = {emb: {m: [] for m in METRIC_KEYS} for emb in embeddings}

    for rep in tqdm(range(n_repeats), desc="Subsample repeats"):
        sub_pos = _stratified_subsample(groups, subsample_frac, rng)
        y_sub = y_all[sub_pos]
        adata_sub = adata[sub_pos]
        # earliest-state roots that fell into this subset (positions within subset)
        sub_earliest = np.where(y_sub == np.nanmin(y_sub))[0]
        if sub_earliest.size == 0:
            continue
        k = min(k_roots, sub_earliest.size)
        roots = rng.choice(sub_earliest, size=k, replace=False)
        for emb in embeddings:
            m = _metrics_on_subset(adata_sub.copy(), emb, roots, y_sub, verbose=verbose)
            for key in METRIC_KEYS:
                per_repeat[emb][key].append(m[key])

    # ---- Aggregate: stability intervals ----
    for emb in embeddings:
        for key in METRIC_KEYS:
            vals = np.asarray(per_repeat[emb][key], float)
            vals = vals[np.isfinite(vals)]
            lo = float(np.clip(np.quantile(vals, alpha / 2), -1, 1)) if vals.size else np.nan
            hi = float(np.clip(np.quantile(vals, 1 - alpha / 2), -1, 1)) if vals.size else np.nan
            results[emb][f"{key}_ci"] = [lo, hi]
            results[emb][f"{key}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            results[emb][f"{key}_std"] = float(np.std(vals)) if vals.size else np.nan
        results[emb]["repeat_values"] = {k: per_repeat[emb][k] for k in METRIC_KEYS}

    # ---- Pairwise distinguishability (paired per-repeat differences) ----
    distinguishability = {}
    emb_names = list(embeddings)
    for key in METRIC_KEYS:
        pairs = []
        for a_i in range(len(emb_names)):
            for b_i in range(a_i + 1, len(emb_names)):
                a, b = emb_names[a_i], emb_names[b_i]
                va = np.asarray(per_repeat[a][key], float)
                vb = np.asarray(per_repeat[b][key], float)
                n = min(va.size, vb.size)
                diff = va[:n] - vb[:n]
                diff = diff[np.isfinite(diff)]
                if diff.size == 0:
                    continue
                ci_lo = float(np.quantile(diff, alpha / 2))
                ci_hi = float(np.quantile(diff, 1 - alpha / 2))
                pairs.append({
                    "emb_a": a,
                    "emb_b": b,
                    "mean_diff": float(np.mean(diff)),
                    "diff_ci": [ci_lo, ci_hi],
                    "distinguishable": bool(ci_lo > 0 or ci_hi < 0),
                })
        distinguishability[key] = {"metric": key, "pairs": pairs}

    # ---- Summary ----
    ranked = sorted(results.items(), key=lambda x: x[1]["spearman"], reverse=True)
    print(f"\n{'Rank':<5}{'Embedding':<40}{'Spearman':<24}{'Lin CCC':<20}")
    print("-" * 90)
    for i, (name, r) in enumerate(ranked, 1):
        s_ci = r["spearman_ci"]
        print(f"{i:<5}{name:<40}{r['spearman']:.4f} [{s_ci[0]:.3f},{s_ci[1]:.3f}]   "
              f"{r['lins_ccc']:.4f}")

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        json_results = {emb: {k: v for k, v in r.items() if k != "repeat_values"}
                        for emb, r in results.items()}
        json_results["_distinguishability"] = distinguishability
        json_results["_inference"] = {
            "method": "subsampling_without_replacement_recompute",
            "n_repeats": n_repeats,
            "subsample_frac": subsample_frac,
            "k_roots": k_roots,
            "n_candidates": n_candidates,
            "seed": seed,
        }
        with open(save_path / "pseudotime.json", "w") as f:
            json.dump(json_results, f, indent=4)
        print(f"\nSaved to {save_path / 'pseudotime.json'}")

    return results, distinguishability
