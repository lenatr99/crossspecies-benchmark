"""Assign marker-defined cell states to AnnData objects.

The default method uses a rank-based gene-set score with expression-matched
random gene sets for thresholds. It is designed for cross-dataset/species
comparisons where raw expression scales are not directly comparable.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results" / "evaluation"
DEFAULT_CACHE_DIR = ROOT / "data" / "evaluation" / ".cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR / "xdg"))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm.auto import tqdm

try:
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover
    GaussianMixture = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

warnings.simplefilter("ignore", ResourceWarning)


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_write_h5ad(adata, out_path: Path) -> None:
    """Write an AnnData file via a same-directory temp file, then replace."""
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".h5ad", dir=out_path.parent)
    os.close(fd)
    tmp = Path(tmp_path)
    try:
        adata.write_h5ad(tmp)
        shutil.move(str(tmp), str(out_path))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _h5ad_safe(value: Any) -> Any:
    """Convert nested metadata to AnnData/HDF5-friendly scalar containers."""
    if value is None:
        return "NA"
    if isinstance(value, dict):
        return {str(k): _h5ad_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_h5ad_safe(v) for v in value]
    return value


def sanitize_uns_for_h5ad(adata) -> None:
    adata.uns = _h5ad_safe(dict(adata.uns))


def load_module(module_or_path: str) -> ModuleType:
    """Load gene lists by module name or Python file path."""
    value = str(module_or_path).strip()
    if not value:
        raise ValueError("Empty --gene-lists value.")

    if value.endswith(".py") or Path(value).exists():
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Gene-list file not found: {path}\n"
                "Add your marker module there, or pass --gene-lists as an importable module."
            )
        name = f"gene_lists__{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import gene-list module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module

    return importlib.import_module(value)


def get_program_dict_and_mode(
    module: ModuleType, gene_sets_key: str, bundle_mode: str
) -> tuple[dict[str, list[str]], str]:
    """Return marker programs and xor/or bundle mode from a gene-list module."""
    if hasattr(module, "GENE_SETS"):
        gene_sets = getattr(module, "GENE_SETS")
        if not isinstance(gene_sets, dict) or gene_sets_key not in gene_sets:
            available = list(gene_sets) if isinstance(gene_sets, dict) else type(gene_sets)
            raise KeyError(f"{gene_sets_key!r} not found in {module.__name__}.GENE_SETS. Available: {available}")
        program_dict, inferred_mode = gene_sets[gene_sets_key]
        return validate_program_dict(program_dict), bundle_mode if bundle_mode != "auto" else inferred_mode

    if not hasattr(module, gene_sets_key):
        candidates = [k for k in dir(module) if not k.startswith("_")]
        raise KeyError(f"Gene set {gene_sets_key!r} not found in {module.__name__}. Available examples: {candidates[:50]}")
    return validate_program_dict(getattr(module, gene_sets_key)), bundle_mode if bundle_mode != "auto" else "xor"


def validate_program_dict(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dict of program -> gene list, got {type(value)}")
    out: dict[str, list[str]] = {}
    for program, genes in value.items():
        if not isinstance(genes, (list, tuple, set, pd.Index, np.ndarray)):
            raise TypeError(f"Expected gene list for {program!r}, got {type(genes)}")
        out[str(program)] = [str(g) for g in genes if str(g).strip()]
    return out


def _as_1d(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x).reshape(-1)


def _get_gene_mask(adata, gene_list: list[str], gene_id_col: str | None) -> np.ndarray:
    if gene_id_col:
        if gene_id_col not in adata.var:
            raise KeyError(
                f"adata.var[{gene_id_col!r}] is required for gene matching. "
                f"Available var columns: {list(adata.var.columns)}"
            )
        return adata.var[gene_id_col].astype(str).isin(gene_list).to_numpy()
    return pd.Index(adata.var_names.astype(str)).isin(gene_list)


def compute_program_scores(adata, gene_sets: dict[str, list[str]], score_prefix: str, gene_id_col: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for program_name, gene_list in gene_sets.items():
        genes_mask = _get_gene_mask(adata, gene_list, gene_id_col)
        n_found = int(genes_mask.sum())
        if n_found == 0:
            print(f"  [scores] Skipping {program_name!r}: 0/{len(gene_list)} genes found")
            continue
        scores = _as_1d(adata[:, genes_mask].X.mean(axis=1))
        col = f"{score_prefix}__{program_name}__score__diag"
        adata.obs[col] = scores
        out[program_name] = col
        print(f"  [scores] {program_name}: found {n_found}/{len(gene_list)} genes; range {scores.min():.3f}-{scores.max():.3f}")
    return out


def _topk_hit_weight_score(X, gene_set_mask: np.ndarray, topk: int = 1500) -> np.ndarray:
    n_cells, n_genes = X.shape
    marker_count = int(gene_set_mask.sum())
    if marker_count == 0:
        return np.zeros(n_cells, dtype=float)
    k = int(min(max(10, topk), n_genes))
    denom = max(k - 1, 1)
    scores = np.zeros(n_cells, dtype=float)

    if sp.issparse(X):
        csr = X.tocsr()
        for i in range(n_cells):
            row = csr.indices[csr.indptr[i] : csr.indptr[i + 1]]
            vals = csr.data[csr.indptr[i] : csr.indptr[i + 1]]
            if row.size == 0:
                continue
            if row.size > k:
                part = np.argpartition(vals, -k)[-k:]
                row = row[part]
                vals = vals[part]
            top_genes = row[np.argsort(-vals)]
            weights = 1.0 - (np.arange(top_genes.size) / denom)
            hits = gene_set_mask[top_genes]
            if hits.any():
                scores[i] = float(weights[hits].sum()) / marker_count
        return scores

    dense = np.asarray(X)
    for i in range(n_cells):
        vals = dense[i]
        if np.all(vals == 0):
            continue
        top_idx = np.argpartition(vals, -k)[-k:]
        top_genes = top_idx[np.argsort(-vals[top_idx])]
        weights = 1.0 - (np.arange(top_genes.size) / denom)
        hits = gene_set_mask[top_genes]
        if hits.any():
            scores[i] = float(weights[hits].sum()) / marker_count
    return scores


def _make_expression_bins(gene_means: np.ndarray, n_bins: int = 20) -> np.ndarray:
    qs = np.quantile(np.asarray(gene_means, dtype=float), np.linspace(0, 1, n_bins + 1))
    return np.digitize(gene_means, qs[1:-1], right=True)


def _benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    pvalues = np.asarray(pvalues, dtype=float)
    shape = pvalues.shape
    flat = pvalues.reshape(-1)
    n = len(flat)
    if n == 0:
        return pvalues, np.zeros(shape, dtype=bool)
    order = np.argsort(flat)
    sorted_p = flat[order]
    ranks = np.arange(1, n + 1)
    reject_sorted = sorted_p <= (ranks / n) * alpha
    if reject_sorted.any():
        reject_sorted[: np.where(reject_sorted)[0][-1] + 1] = True
    q_sorted = np.minimum.accumulate((n / ranks[::-1] * sorted_p[::-1]))[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n, dtype=float)
    reject = np.empty(n, dtype=bool)
    q[order] = q_sorted
    reject[order] = reject_sorted
    return q.reshape(shape), reject.reshape(shape)


def _sample_matched_random_sets(
    rng: np.random.Generator,
    gene_set_gene_indices: np.ndarray,
    bins: np.ndarray,
    n_sets: int = 50,
) -> list[np.ndarray]:
    gene_set_gene_indices = np.asarray(gene_set_gene_indices, dtype=int)
    target_size = len(gene_set_gene_indices)
    by_bin: dict[int, np.ndarray] = {}
    for b in np.unique(bins):
        pool = np.where(bins == b)[0]
        by_bin[int(b)] = pool[~np.isin(pool, gene_set_gene_indices)]

    bin_counts = Counter(int(b) for b in bins[gene_set_gene_indices])
    all_non_target = np.concatenate([p for p in by_bin.values() if len(p)]) if by_bin else np.array([], dtype=int)
    out: list[np.ndarray] = []
    for _ in range(n_sets):
        chosen: list[int] = []
        for b_val, count in bin_counts.items():
            pool = by_bin.get(b_val, np.array([], dtype=int))
            if len(pool) == 0:
                continue
            chosen.extend(rng.choice(pool, size=count, replace=len(pool) < count).tolist())
        rand = np.array(chosen, dtype=int)
        if len(rand) < target_size and len(all_non_target) > 0:
            fill = rng.choice(all_non_target, size=target_size - len(rand), replace=True)
            rand = np.concatenate([rand, fill])
        out.append(rand[:target_size])
    return out


def _gaussian_posterior_boundary_1d(mu_low, s_low, w_low, mu_high, s_high, w_high, p_high=0.5) -> list[float]:
    s_low = float(max(s_low, 1e-8))
    s_high = float(max(s_high, 1e-8))
    w_low = float(max(w_low, 1e-12))
    w_high = float(max(w_high, 1e-12))
    p_high = min(max(float(p_high), 1e-6), 1 - 1e-6)
    ratio = p_high / (1 - p_high)
    a = 1 / (2 * s_high**2) - 1 / (2 * s_low**2)
    b = mu_low / s_low**2 - mu_high / s_high**2
    c = (mu_high**2) / (2 * s_high**2) - (mu_low**2) / (2 * s_low**2) + np.log((s_high * ratio * w_low) / (s_low * w_high))
    if abs(a) < 1e-12:
        return [] if abs(b) < 1e-12 else [float(-c / b)]
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    root = np.sqrt(disc)
    return [float((-b + root) / (2 * a)), float((-b - root) / (2 * a))]


def _gmm_threshold(
    scores: np.ndarray,
    *,
    seed: int = 0,
    bic_delta: float = 10.0,
    max_cells: int = 20000,
    posterior_cut: float = 0.9,
    min_weight: float = 0.2,
    max_std_ratio: float = 10.0,
) -> tuple[bool, np.ndarray | None, dict[str, Any], np.ndarray | None]:
    x = np.asarray(scores, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size < 200 or GaussianMixture is None:
        return False, None, {"reason": "too_few_points_or_no_sklearn"}, None

    rng = np.random.default_rng(seed)
    fit = rng.choice(finite, size=max_cells, replace=False) if finite.size > max_cells else finite
    X_fit = fit.reshape(-1, 1)
    g1 = GaussianMixture(n_components=1, random_state=seed).fit(X_fit)
    g2 = GaussianMixture(n_components=2, random_state=seed).fit(X_fit)

    means = g2.means_.reshape(-1)
    stds = np.sqrt(g2.covariances_.reshape(-1))
    weights = g2.weights_.reshape(-1)
    order = np.argsort(means)
    means, stds, weights = means[order], stds[order], weights[order]
    p_high = g2.predict_proba(x.reshape(-1, 1))[:, int(order[1])]

    bic1 = float(g1.bic(X_fit))
    bic2 = float(g2.bic(X_fit))
    delta = bic1 - bic2
    info: dict[str, Any] = {
        "bic1": bic1,
        "bic2": bic2,
        "delta_bic": delta,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "posterior_cut": float(posterior_cut),
    }
    if delta < bic_delta:
        info["reason"] = "delta_bic_small"
        return False, None, info, p_high
    sep = abs(means[1] - means[0]) / max(float(stds.mean()), 1e-8)
    info["sep"] = float(sep)
    if sep < 1.0:
        info["reason"] = "means_not_separated"
        return False, None, info, p_high
    if float(min(weights)) < min_weight:
        info["reason"] = "min_weight_too_small"
        info["min_weight"] = float(min_weight)
        return False, None, info, p_high
    std_ratio = float(max(stds) / max(min(stds), 1e-12))
    if std_ratio > max_std_ratio:
        info["reason"] = "std_ratio_too_large"
        info["std_ratio"] = std_ratio
        return False, None, info, p_high

    active = p_high >= posterior_cut
    active_frac = float(np.mean(active)) if active.size else 0.0
    info["active_frac"] = active_frac
    if active_frac <= 0.0 or active_frac >= 1.0:
        info["reason"] = "degenerate_active_fraction"
        return False, None, info, p_high

    roots = _gaussian_posterior_boundary_1d(means[0], stds[0], weights[0], means[1], stds[1], weights[1], posterior_cut)
    if roots:
        mid = float((means[0] + means[1]) / 2)
        between = [r for r in roots if min(means) <= r <= max(means)]
        choices = between or roots
        info["thr_equiv"] = float(choices[int(np.argmin([abs(r - mid) for r in choices]))])
    else:
        info["thr_equiv"] = None
    return True, active, info, p_high


def discretize_rank_null(
    score_adata,
    gene_sets: dict[str, list[str]],
    gene_sets_name: str,
    *,
    gene_id_col: str | None,
    topk: int = 1500,
    n_null_sets: int = 50,
    null_cells: int = 5000,
    null_q: float = 0.99,
    threshold_method: str = "auto",
    gmm_bic_delta: float = 10.0,
    gmm_max_cells: int = 20000,
    gmm_posterior_cut: float = 0.9,
    gmm_min_weight: float = 0.2,
    gmm_max_std_ratio: float = 10.0,
    fdr: float = 0.05,
    seed: int = 0,
    allow_ambiguous: bool = False,
    make_single_label: bool = True,
    xor_pick: str = "qvalue",
    progress: bool = False,
) -> tuple[Any, dict[str, float]]:
    rng = np.random.default_rng(seed)
    X = score_adata.X
    n_cells = score_adata.n_obs
    null_cells = int(min(max(200, null_cells), n_cells))
    null_idx = rng.choice(np.arange(n_cells), size=null_cells, replace=False)
    bins = _make_expression_bins(_as_1d(score_adata.X.mean(axis=0)), n_bins=20)

    program_scores: dict[str, np.ndarray] = {}
    program_thresholds: dict[str, float] = {}
    program_active: dict[str, np.ndarray] = {}
    program_threshold_info: dict[str, Any] = {}
    program_null_stds: dict[str, float] = {}
    program_pvalues: dict[str, np.ndarray] = {}

    items = list(gene_sets.items())
    iterator = tqdm(items, desc="Programs (rank-null)", leave=True) if progress else items
    for program_name, gene_list in iterator:
        genes_mask = np.asarray(_get_gene_mask(score_adata, gene_list, gene_id_col), dtype=bool)
        if int(genes_mask.sum()) == 0:
            print(f"  [rank-null] Skipping {program_name!r}: 0/{len(gene_list)} genes found")
            continue

        observed = _topk_hit_weight_score(X, genes_mask, topk=topk)
        program_scores[program_name] = observed
        score_adata.obs[f"{gene_sets_name}__{program_name}__rank_score__diag"] = observed

        rand_sets = _sample_matched_random_sets(rng, np.where(genes_mask)[0], bins, n_sets=n_null_sets)
        X_null = score_adata.X[null_idx]
        null_scores = []
        for rand_set in rand_sets:
            rand_mask = np.zeros(score_adata.n_vars, dtype=bool)
            rand_mask[rand_set] = True
            null_scores.append(_topk_hit_weight_score(X_null, rand_mask, topk=topk))
        null_scores_all = np.concatenate(null_scores, axis=0)
        thr_null = float(np.quantile(null_scores_all, null_q))
        program_null_stds[program_name] = float(np.std(null_scores_all))

        use_gmm, gmm_active, gmm_info, _ = _gmm_threshold(
            observed,
            seed=seed,
            bic_delta=gmm_bic_delta,
            max_cells=gmm_max_cells,
            posterior_cut=gmm_posterior_cut,
            min_weight=gmm_min_weight,
            max_std_ratio=gmm_max_std_ratio,
        )

        null_sorted = np.sort(null_scores_all.astype(float))
        n_null = len(null_sorted)
        count_ge = n_null - np.searchsorted(null_sorted, observed.astype(float), side="left")
        pvalue_empirical = (count_ge + 1) / (n_null + 1)
        program_pvalues[program_name] = pvalue_empirical

        chosen = "null"
        threshold = thr_null
        if threshold_method in {"auto", "gmm"} and use_gmm and gmm_active is not None:
            chosen = "gmm"
            threshold = float(gmm_info.get("thr_equiv") if gmm_info.get("thr_equiv") is not None else thr_null)

        program_thresholds[program_name] = threshold
        program_threshold_info[program_name] = {
            "chosen": chosen,
            "thr": threshold,
            "thr_null": thr_null,
            "thr_gmm": None if gmm_info.get("thr_equiv") is None else float(gmm_info.get("thr_equiv")),
            "gmm": gmm_info,
        }
        active = np.asarray(gmm_active, dtype=bool) if chosen == "gmm" and gmm_active is not None else observed >= thr_null
        program_active[program_name] = active
        score_adata.obs[f"{gene_sets_name}__{program_name}__active__diag"] = pd.Categorical(
            np.where(active, "active", "inactive"),
            categories=["inactive", "active"],
        )

    if len(program_scores) >= 2:
        program_names = list(program_scores)
        pval_matrix = np.column_stack([program_pvalues[p] for p in program_names])
        qval_matrix = np.zeros_like(pval_matrix)
        for i in range(score_adata.n_obs):
            qval_matrix[i, :], _ = _benjamini_hochberg(pval_matrix[i, :], alpha=fdr)
        for k, program in enumerate(program_names):
            score_adata.obs[f"{gene_sets_name}__{program}__pvalue_null__diag"] = pval_matrix[:, k]
            score_adata.obs[f"{gene_sets_name}__{program}__qvalue_bh__diag"] = qval_matrix[:, k]
            active_bh = qval_matrix[:, k] < fdr
            score_adata.obs[f"{gene_sets_name}__{program}__active_bh__diag"] = pd.Categorical(
                np.where(active_bh, "active", "inactive"),
                categories=["inactive", "active"],
            )
    elif len(program_scores) == 1:
        program = next(iter(program_scores))
        score_adata.obs[f"{gene_sets_name}__{program}__pvalue_null__diag"] = program_pvalues[program]

    if len(program_scores) >= 2 and program_null_stds:
        tau = max(float(np.sqrt(np.mean([s**2 for s in program_null_stds.values()]))), 1e-8)
        program_names = list(program_scores)
        score_matrix = np.column_stack([program_scores[p] for p in program_names])
        logits = score_matrix / tau
        logits -= logits.max(axis=1, keepdims=True)
        class_probs = np.exp(logits)
        class_probs /= class_probs.sum(axis=1, keepdims=True)
        for k, program in enumerate(program_names):
            score_adata.obs[f"{gene_sets_name}__{program}__class_prob__diag"] = class_probs[:, k]
        score_adata.uns[f"{gene_sets_name}__softmax_tau"] = float(tau)

    if len(program_scores) == 0:
        if make_single_label:
            score_adata.obs[gene_sets_name] = None
        return score_adata, program_thresholds

    if make_single_label:
        scores_df = pd.DataFrame(program_scores, index=score_adata.obs_names)
        active_df = pd.DataFrame(
            {
                program: (
                    score_adata.obs[f"{gene_sets_name}__{program}__active_bh__diag"] == "active"
                    if f"{gene_sets_name}__{program}__active_bh__diag" in score_adata.obs
                    else program_active[program]
                )
                for program in program_scores
            },
            index=score_adata.obs_names,
        )
        any_active = active_df.any(axis=1)
        xor_pick = str(xor_pick or "score").strip().lower()
        if xor_pick == "score":
            best = scores_df.where(active_df, other=-np.inf).idxmax(axis=1)
        elif xor_pick == "qvalue" and all(f"{gene_sets_name}__{p}__qvalue_bh__diag" in score_adata.obs for p in program_scores):
            qvals = pd.DataFrame(
                {p: score_adata.obs[f"{gene_sets_name}__{p}__qvalue_bh__diag"].to_numpy(dtype=float) for p in program_scores},
                index=score_adata.obs_names,
            )
            best = qvals.where(active_df, other=np.inf).idxmin(axis=1)
        else:
            pvals = pd.DataFrame(program_pvalues, index=score_adata.obs_names)
            best = pvals.where(active_df, other=np.inf).idxmin(axis=1)
        label = best.where(any_active, other=None)
        if allow_ambiguous and active_df.shape[1] >= 2:
            label = label.where(active_df.sum(axis=1) <= 1, other=None)
        score_adata.obs[gene_sets_name] = label

    score_adata.uns[f"{gene_sets_name}__rank_threshold_info"] = program_threshold_info
    return score_adata, program_thresholds


def classify_cells_by_gene_signature(adata, gene_sets: dict[str, list[str]], gene_sets_name: str, gene_id_col: str | None, show_plots: bool = False):
    """Legacy mean-expression + Otsu thresholding."""
    assignment_score_df = pd.DataFrame(index=adata.obs_names)
    for cell_type, gene_list in gene_sets.items():
        print(f"\nProcessing {cell_type}...")
        genes_mask = _get_gene_mask(adata, gene_list, gene_id_col)
        n_found = int(genes_mask.sum())
        if n_found == 0:
            print(f"  No genes found for {cell_type}")
            continue
        scores = _as_1d(adata[:, genes_mask].X.mean(axis=1))
        if len(np.unique(scores)) <= 1:
            threshold = 0.0
        else:
            hist, bin_edges = np.histogram(scores, bins=256)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            total_weight = np.sum(hist)
            total_mean = np.sum(bin_centers * hist) / max(total_weight, 1)
            max_variance, threshold = 0.0, float(bin_centers[0])
            for i in range(len(hist)):
                w0 = np.sum(hist[: i + 1])
                if w0 == 0 or w0 == total_weight:
                    continue
                mu0 = np.sum(bin_centers[: i + 1] * hist[: i + 1]) / w0
                w1 = total_weight - w0
                mu1 = (total_mean * total_weight - mu0 * w0) / w1
                variance = w0 * w1 * (mu0 - mu1) ** 2
                if variance > max_variance:
                    max_variance, threshold = variance, float(bin_centers[i])
            if show_plots and plt is not None:
                plt.hist(scores, bins=256)
                plt.axvline(threshold, color="red")
                plt.show()
        high = scores >= threshold
        assignment_score_df[cell_type] = np.maximum(0, (scores - threshold) / max(threshold, 1e-8) if threshold > 0 else scores - threshold)
        print(f"  Found {n_found}/{len(gene_list)} genes; threshold {threshold:.4f}; selected {high.sum()} cells ({high.mean()*100:.1f}%)")

    best = assignment_score_df.idxmax(axis=1)
    max_score = assignment_score_df.max(axis=1)
    adata.obs[gene_sets_name] = best.where(max_score > 0, other=None)
    return adata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "dicty.h5ad", help="Input AnnData .h5ad.")
    parser.add_argument("--gene-lists", "--gene_lists", dest="gene_lists", default=str(SCRIPT_DIR / "src" / "gene_lists_dicty.py"), help="Gene-list module name or .py file path.")
    parser.add_argument("--gene-sets", "--gene_sets", dest="gene_sets", default="cell_type", help="Gene-set dictionary name.")
    parser.add_argument("--gene-sets-name", "--gene_sets_name", dest="gene_sets_name", default="", help="Output obs prefix/name; defaults to --gene-sets.")
    parser.add_argument("--bundle-mode", "--bundle_mode", dest="bundle_mode", choices=["auto", "xor", "or"], default="auto", help="Whether programs are mutually exclusive or multi-label.")
    parser.add_argument("--var-names-key", "--var_names_key", dest="var_names_key", default="", help="Set adata.var_names from this var column before matching.")
    parser.add_argument("--gene-id-col", "--gene_id_col", dest="gene_id_col", default="gene_ids", help="adata.var column for gene matching. Empty string uses var_names.")
    parser.add_argument("--mode", choices=["discretize_otsu", "discretize_rank_null", "scores_only"], default="discretize_rank_null")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--topk", type=int, default=1500)
    parser.add_argument("--n-null-sets", "--n_null_sets", dest="n_null_sets", type=int, default=50)
    parser.add_argument("--null-cells", "--null_cells", dest="null_cells", type=int, default=5000)
    parser.add_argument("--null-q", "--null_q", dest="null_q", type=float, default=0.99)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--threshold-method", "--threshold_method", dest="threshold_method", choices=["auto", "null", "gmm"], default="auto")
    parser.add_argument("--gmm-bic-delta", "--gmm_bic_delta", dest="gmm_bic_delta", type=float, default=10.0)
    parser.add_argument("--gmm-max-cells", "--gmm_max_cells", dest="gmm_max_cells", type=int, default=20000)
    parser.add_argument("--gmm-posterior-cut", "--gmm_posterior_cut", dest="gmm_posterior_cut", type=float, default=0.9)
    parser.add_argument("--gmm-min-weight", "--gmm_min_weight", dest="gmm_min_weight", type=float, default=0.2)
    parser.add_argument("--gmm-max-std-ratio", "--gmm_max_std_ratio", dest="gmm_max_std_ratio", type=float, default=10.0)
    parser.add_argument("--allow-ambiguous", "--allow_ambiguous", dest="allow_ambiguous", action="store_true", default=False)
    parser.add_argument("--xor-pick", "--xor_pick", dest="xor_pick", choices=["score", "pvalue", "qvalue"], default="qvalue")
    parser.add_argument("--save-intermediate", "--save_intermediate", dest="save_intermediate", action="store_true", default=False, help="Kept for compatibility; diagnostic columns are always saved for rank-null.")
    parser.add_argument("--show-plots", "--show_plots", dest="show_plots", action="store_true", default=False)
    parser.add_argument("--write-h5ad", "--write_h5ad", dest="write_h5ad", action="store_true", default=False)
    parser.add_argument("--out-h5ad", "--out_h5ad", dest="out_h5ad", default="", help="Output .h5ad path. Defaults to overwriting --data when writing.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for JSON manifests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    gene_id_col = args.gene_id_col.strip() or None
    gene_sets_name = args.gene_sets_name.strip() or args.gene_sets

    data_path = args.data.expanduser()
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    module = load_module(args.gene_lists)
    program_dict, bundle_mode = get_program_dict_and_mode(module, args.gene_sets, args.bundle_mode)
    if bundle_mode not in {"xor", "or"}:
        raise ValueError(f"bundle_mode must resolve to 'xor' or 'or', got {bundle_mode!r}")
    is_xor = bundle_mode == "xor"

    print(f"[load] Reading {data_path}")
    adata = sc.read_h5ad(data_path)
    if args.var_names_key:
        if args.var_names_key not in adata.var:
            raise KeyError(f"--var-names-key {args.var_names_key!r} not in adata.var. Available: {list(adata.var.columns)}")
        adata.var_names = adata.var[args.var_names_key].astype(str)
    org_adata = adata.copy()

    score_adata = adata.copy()
    sc.pp.normalize_total(score_adata, target_sum=1e4)
    sc.pp.log1p(score_adata)

    score_cols = compute_program_scores(score_adata, program_dict, gene_sets_name, gene_id_col)
    if args.save_intermediate:
        for col in score_cols.values():
            org_adata.obs[col] = score_adata.obs[col]

    thresholds: dict[str, float] = {}
    if args.mode == "scores_only":
        pass
    elif args.mode == "discretize_rank_null":
        print(
            f"[rank-null] {gene_sets_name} bundle_mode={bundle_mode} topk={args.topk} "
            f"n_null_sets={args.n_null_sets} null_cells={args.null_cells} null_q={args.null_q} seed={args.seed}"
            + (f" xor_pick={args.xor_pick}" if is_xor else "")
        )
        score_adata, thresholds = discretize_rank_null(
            score_adata,
            program_dict,
            gene_sets_name,
            gene_id_col=gene_id_col,
            topk=args.topk,
            n_null_sets=args.n_null_sets,
            null_cells=args.null_cells,
            null_q=args.null_q,
            threshold_method=args.threshold_method,
            gmm_bic_delta=args.gmm_bic_delta,
            gmm_max_cells=args.gmm_max_cells,
            gmm_posterior_cut=args.gmm_posterior_cut,
            gmm_min_weight=args.gmm_min_weight,
            gmm_max_std_ratio=args.gmm_max_std_ratio,
            fdr=args.fdr,
            seed=args.seed,
            allow_ambiguous=args.allow_ambiguous,
            make_single_label=is_xor,
            xor_pick=args.xor_pick,
            progress=args.progress,
        )
        prefix = f"{gene_sets_name}__"
        if is_xor:
            org_adata.obs[gene_sets_name] = score_adata.obs[gene_sets_name]
        else:
            for col in score_adata.obs.columns:
                if col.startswith(prefix) and col.endswith("__active__diag"):
                    clean_name = col.replace("__active__diag", "")
                    values = score_adata.obs[col].astype(str)
                    org_adata.obs[clean_name] = pd.Categorical(
                        np.where(values == "active", clean_name, None),
                        categories=[clean_name],
                    )
        for col in score_adata.obs.columns:
            if col.startswith(prefix):
                org_adata.obs[col] = score_adata.obs[col]
        org_adata.uns[f"{gene_sets_name}__rank_null_thresholds"] = thresholds
        if f"{gene_sets_name}__rank_threshold_info" in score_adata.uns:
            org_adata.uns[f"{gene_sets_name}__rank_threshold_info"] = score_adata.uns[f"{gene_sets_name}__rank_threshold_info"]
        if f"{gene_sets_name}__softmax_tau" in score_adata.uns:
            org_adata.uns[f"{gene_sets_name}__softmax_tau"] = score_adata.uns[f"{gene_sets_name}__softmax_tau"]
    elif args.mode == "discretize_otsu":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
        print(f"[otsu] {gene_sets_name} bundle_mode={bundle_mode}")
        if is_xor:
            adata = classify_cells_by_gene_signature(adata, program_dict, gene_sets_name, gene_id_col, args.show_plots)
            org_adata.obs[gene_sets_name] = adata.obs[gene_sets_name]
        else:
            for program, genes in program_dict.items():
                if not genes:
                    continue
                adata = classify_cells_by_gene_signature(adata, {program: genes}, program, gene_id_col, args.show_plots)
                org_adata.obs[program] = adata.obs[program]

    out_path = Path(args.out_h5ad).expanduser() if args.out_h5ad.strip() else data_path
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    h5ad_written = False
    if args.write_h5ad:
        sanitize_uns_for_h5ad(org_adata)
        _safe_write_h5ad(org_adata, out_path)
        h5ad_written = True
        print(f"[done] Wrote {out_path}")
    else:
        print("[done] Not writing .h5ad (pass --write-h5ad to persist).")

    manifest = {
        "step": "assign_marker_labels",
        "created_utc": utc_now(),
        "data": str(data_path),
        "out_h5ad": str(out_path) if h5ad_written else None,
        "h5ad_written": h5ad_written,
        "gene_lists": args.gene_lists,
        "gene_sets": args.gene_sets,
        "gene_sets_name": gene_sets_name,
        "bundle_mode": bundle_mode,
        "mode": args.mode,
        "seed": args.seed,
        "parameters": {
            "topk": args.topk,
            "n_null_sets": args.n_null_sets,
            "null_cells": args.null_cells,
            "null_q": args.null_q,
            "fdr": args.fdr,
            "threshold_method": args.threshold_method,
            "xor_pick": args.xor_pick,
            "allow_ambiguous": args.allow_ambiguous,
            "gene_id_col": gene_id_col,
            "var_names_key": args.var_names_key or None,
        },
        "programs": {name: len(genes) for name, genes in program_dict.items()},
        "thresholds": thresholds,
        "obs_columns_written": [c for c in org_adata.obs.columns if c == gene_sets_name or c.startswith(f"{gene_sets_name}__")],
    }
    manifest_path = args.results_dir / f"{data_path.stem}_{gene_sets_name}_{args.mode}_manifest.json"
    write_json(manifest_path, manifest)
    print(f"[manifest] Wrote {manifest_path}")


if __name__ == "__main__":
    main()
