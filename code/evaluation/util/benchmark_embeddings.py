"""SCIB integration benchmark utilities.

Supports paired repeated subsampling for stability intervals and pairwise
significance testing (Wilcoxon signed-rank).

Why paired subsampling (not bootstrap)?
-----------------------------------------
Standard bootstrap (sampling with replacement) is inappropriate for scIB metrics:

1. **Graph-based metrics are not i.i.d.**: scIB metrics (NMI, ARI, ASW) are computed
   on a kNN graph that is recomputed for each sample. Changing which cells are present
   changes the graph structure for *all* cells, not just the resampled ones. This
   violates the assumption that observations are independent.

2. **Duplicates distort neighborhoods**: Bootstrap creates duplicate cells (same point
   appears multiple times) while omitting ~37% of original cells. Duplicates create
   artificial micro-clusters in the kNN graph, distorting distance structure and
   density-based metrics like silhouette width.

3. **Clustering is non-linear and unstable**: NMI/ARI depend on Leiden clustering with
   resolution optimization. Small perturbations in the graph can cause large discrete
   jumps in cluster assignments, making bootstrap variance dominated by clustering
   instability rather than biological signal variability.

4. **Systematically biases scores downward**: Removing cells breaks graph connectivity
   and creates smaller, sparser neighborhoods. Metrics like graph_connectivity and
   cluster separation systematically decrease, so bootstrap "confidence intervals"
   don't reflect sampling uncertainty--they reflect graph degradation.

What we do instead:
--------------------
- **Subsampling without replacement** (default 90% of cells, stratified by cell type)
  avoids duplicates and preserves more of the original graph structure.
- **Paired design**: All methods evaluated on identical cell subsets per repeat,
  enabling powerful paired tests (Wilcoxon signed-rank) that control for graph/
  clustering noise.
- **Interpretation**: Intervals reflect *robustness* of the metric to dataset
  perturbation, not classical confidence intervals of a population parameter.

This matches how the original scIB paper (Luecken et al., 2022) reports uncertainty:
across different datasets/tasks, not via resampling within one dataset.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = ROOT / "data" / "evaluation" / ".cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR / "xdg"))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import t as t_dist
from scipy.stats import wilcoxon
from scib.metrics.ari import ari
from scib.metrics.clustering import cluster_optimal_resolution
from scib.metrics.graph_connectivity import graph_connectivity
from scib.metrics.nmi import nmi
from scib.metrics.silhouette import silhouette, silhouette_batch
from tqdm import tqdm


def _to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


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


def _compute_metrics_with_neighbors(
    adata_int,
    batch_key: str,
    label_key: str,
    embed_key: str,
    silhouette_subsample: int | None = None,
    seed: int = 0,
    resolutions: list[float] | None = None,
    fixed_resolution: float | None = None,
) -> dict:
    """Compute all scIB metrics on an adata that already has neighbors computed.

    Assumes adata_int.uns["neighbors"] exists and matches embed_key.
    """
    cluster_key = "cluster"

    if fixed_resolution is not None:
        used_resolution = float(fixed_resolution)
        sc.tl.leiden(
            adata_int,
            resolution=used_resolution,
            key_added=cluster_key,
            random_state=seed,
        )
    else:
        kwargs = dict(
            label_key=label_key,
            cluster_key=cluster_key,
            metric=nmi,
            use_rep=embed_key,
            force=True,
            verbose=False,
            return_all=True,
            random_state=seed,
        )
        if resolutions is not None:
            kwargs["resolutions"] = resolutions

        res_max, nmi_max, nmi_all = cluster_optimal_resolution(adata_int, **kwargs)
        used_resolution = float(res_max)

    n_clusters = int(adata_int.obs[cluster_key].nunique())

    nmi_score = nmi(adata_int, cluster_key=cluster_key, label_key=label_key)
    ari_score = ari(adata_int, cluster_key=cluster_key, label_key=label_key)

    if silhouette_subsample and adata_int.n_obs > silhouette_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata_int.n_obs, silhouette_subsample, replace=False)
        ad_sil = adata_int[idx]
    else:
        ad_sil = adata_int

    asw_label = silhouette(ad_sil, label_key=label_key, embed=embed_key)
    asw_batch = silhouette_batch(
        ad_sil,
        batch_key=batch_key,
        label_key=label_key,
        embed=embed_key,
        return_all=False,
        verbose=False,
    )

    graph_conn_score = graph_connectivity(adata_int, label_key=label_key)

    result_dict = {
        "NMI_cluster/label": nmi_score,
        "ARI_cluster/label": ari_score,
        "ASW_label": asw_label,
        "ASW_label/batch": asw_batch,
        "graph_conn": graph_conn_score,
        "leiden_resolution": used_resolution,
        "n_clusters": n_clusters,
    }

    result_dict["avg_bio"] = np.mean([
        result_dict["NMI_cluster/label"],
        result_dict["ARI_cluster/label"],
        result_dict["ASW_label"],
    ])
    result_dict["avg_batch"] = np.mean([
        result_dict["graph_conn"],
        result_dict["ASW_label/batch"],
    ])
    result_dict["overall_score"] = 0.6 * result_dict["avg_bio"] + 0.4 * result_dict["avg_batch"]
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict


def _eval_single_embedding(
    adata_prep,
    emb,
    batch_key,
    cell_type_key,
    silhouette_subsample,
    seed: int,
    resolutions: list[float] | None = None,
    fixed_resolution: float | None = None,
    neighbor_metric: str = "euclidean",
):
    """Worker: compute neighbors + metrics for one embedding (runs in subprocess)."""
    ad = adata_prep.copy()
    with suppress_output():
        sc.pp.neighbors(ad, use_rep=emb, random_state=seed, metric=neighbor_metric)
        r = _compute_metrics_with_neighbors(
            ad, batch_key, cell_type_key, emb,
            silhouette_subsample=silhouette_subsample,
            seed=seed,
            resolutions=resolutions,
            fixed_resolution=fixed_resolution,
        )
    return emb, r


def _find_optimal_resolution(
    adata,
    label_key: str,
    embed_key: str,
    seed: int = 0,
    resolutions: list[float] | None = None,
    neighbor_metric: str = "euclidean",
) -> float:
    """Compute optimal Leiden resolution (NMI-optimized) for one embedding."""
    from scib.metrics.clustering import get_resolutions

    ad = adata.copy()
    sc.pp.neighbors(ad, use_rep=embed_key, random_state=seed, metric=neighbor_metric)

    if resolutions is None:
        resolutions = get_resolutions(n=10, max=2)

    best_res = resolutions[0]
    best_nmi = -1.0
    cluster_key = "_opt_cluster"

    for res in resolutions:
        sc.tl.leiden(ad, resolution=res, key_added=cluster_key, random_state=seed)
        n_clusters = ad.obs[cluster_key].nunique()
        if n_clusters < 2:
            continue
        score = nmi(ad, cluster_key=cluster_key, label_key=label_key)
        if score > best_nmi:
            best_nmi = score
            best_res = res

    if best_nmi < 0:
        warnings.warn(
            f"{embed_key}: no resolution produced >=2 clusters; "
            f"falling back to highest resolution ({resolutions[-1]:.4f})",
            stacklevel=2,
        )
        best_res = resolutions[-1]

    return float(best_res)


UNKNOWN_LABELS = frozenset(
    {
        "",
        "unknown",
        "Unknown",
        "UNKNOWN",
        "nan",
        "NaN",
        "None",
        "none",
        "NA",
        "N/A",
        "<NA>",
        "unassigned",
    }
)


def _clean_labels(adata, key: str, unknown_handling: str):
    """Normalize label values before scIB sees them.

    The original notebook expected missing labels to be real missing values.
    AnnData files can round-trip those as string/category-like placeholders, so
    we collapse those placeholders here while leaving all metric logic unchanged.
    """
    if key not in adata.obs:
        raise KeyError(f"Missing label key in adata.obs: {key}")
    if unknown_handling not in {"filter", "category"}:
        raise ValueError("unknown_handling must be 'filter' or 'category'")

    labels = adata.obs[key]
    as_object = labels.astype("object")
    as_string = as_object.astype(str).str.strip()
    missing = labels.isna() | as_string.isin(UNKNOWN_LABELS)

    cleaned = as_string.copy()
    cleaned.loc[missing] = "unknown"

    if unknown_handling == "filter":
        ad = adata[~missing.to_numpy()].copy()
        ad.obs[key] = cleaned.loc[ad.obs_names].astype(str)
    else:
        ad = adata.copy()
        ad.obs[key] = cleaned.astype(str)

    ad.obs[key] = ad.obs[key].astype("category").cat.remove_unused_categories()
    return ad


def scib_eval(
    adata,
    batch_key: str,
    cell_type_key: str,
    embed_key: str,
    unknown_handling: str = "filter",
    verbose: bool = True,
    seed: int = 0,
    resolutions: list[float] | None = None,
    l2_normalize: bool = False,
    neighbor_metric: str = "euclidean",
) -> dict:
    """Evaluate integration quality using scIB metrics."""
    adata_int = _clean_labels(adata, cell_type_key, unknown_handling)

    if l2_normalize:
        _l2_normalize_embeddings(adata_int, [embed_key], verbose=verbose)

    with suppress_output():
        sc.pp.neighbors(adata_int, use_rep=embed_key, random_state=seed, metric=neighbor_metric)
        return _compute_metrics_with_neighbors(
            adata_int, batch_key, cell_type_key, embed_key,
            silhouette_subsample=None,
            seed=seed,
            resolutions=resolutions,
        )


def _l2_normalize_embeddings(adata, embeddings, verbose=False):
    """L2-normalize embedding matrices in-place. Skips already-normalized ones."""
    for key in embeddings:
        X = adata.obsm[key]
        norms = np.linalg.norm(X, axis=1)
        already_normed = np.allclose(norms[norms > 1e-12], 1.0, atol=1e-3)
        if already_normed:
            if verbose:
                print(f"  {key}: already L2-normalized, skipping")
            continue
        safe_norms = np.maximum(norms, 1e-12)
        adata.obsm[key] = X / safe_norms[:, None]
        if verbose:
            print(f"  {key}: L2-normalized (norms were {norms.min():.2f}-{norms.max():.2f})")


def _prepare_adata(adata, cell_type_key, unknown_handling, embeddings=None, l2_normalize=False, verbose=False):
    """Handle unknowns, clean up categories, optionally L2-normalize. Returns a copy."""
    adata_prep = _clean_labels(adata, cell_type_key, unknown_handling)

    if embeddings is not None:
        missing = [emb for emb in embeddings if emb not in adata_prep.obsm]
        if missing:
            raise KeyError(
                "Missing embedding key(s) in adata.obsm: "
                + ", ".join(missing)
                + f". Available keys include: {list(adata_prep.obsm.keys())[:50]}"
            )

    if l2_normalize and embeddings is not None:
        if verbose:
            print("L2-normalizing embeddings...")
        _l2_normalize_embeddings(adata_prep, embeddings, verbose=verbose)

    return adata_prep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def benchmark_embeddings(
    adata,
    embeddings: list[str],
    batch_key: str,
    cell_type_key: str,
    unknown_handling: str = "filter",
    verbose: bool = True,
    save_dir: str | Path | None = None,
    n_jobs: int = 1,
    silhouette_subsample: int | None = None,
    seed: int = 0,
    resolutions: list[float] | None = None,
    l2_normalize: bool = False,
    neighbor_metric: str = "euclidean",
) -> dict[str, dict]:
    """Benchmark multiple embeddings and return ranked scIB results."""
    adata_prep = _prepare_adata(
        adata,
        cell_type_key,
        unknown_handling,
        embeddings=embeddings,
        l2_normalize=l2_normalize,
        verbose=verbose,
    )

    if verbose:
        sil_msg = f", silhouette on {silhouette_subsample} cells" if silhouette_subsample else ""
        par_msg = f", {n_jobs} workers" if n_jobs > 1 else ""
        print(f"Computing metrics for {len(embeddings)} embeddings ({adata_prep.n_obs} cells{sil_msg}{par_msg})...")

    if n_jobs > 1:
        results = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {
                ex.submit(
                    _eval_single_embedding,
                    adata_prep, emb, batch_key, cell_type_key, silhouette_subsample, seed, resolutions,
                    None, neighbor_metric,
                ): emb
                for emb in embeddings
            }
            pbar = tqdm(total=len(embeddings), desc="Embeddings (parallel)", disable=not verbose)
            for fut in as_completed(futures):
                emb, r = fut.result()
                results[emb] = r
                pbar.update(1)
                if verbose:
                    tqdm.write(
                        f"  {emb}: bio={r['avg_bio']:.4f}  "
                        f"batch={r['avg_batch']:.4f}  "
                        f"overall={r['overall_score']:.4f}  "
                        f"clusters={r.get('n_clusters', '?')}  "
                        f"res={r.get('leiden_resolution', 0):.4f}"
                    )
            pbar.close()
    else:
        results = {}
        for emb in tqdm(embeddings, desc="Embeddings", disable=not verbose):
            _, r = _eval_single_embedding(
                adata_prep, emb, batch_key, cell_type_key, silhouette_subsample, seed, resolutions,
                None, neighbor_metric,
            )
            results[emb] = r
            if verbose:
                tqdm.write(
                    f"  {emb}: bio={r['avg_bio']:.4f}  batch={r['avg_batch']:.4f}  "
                    f"overall={r['overall_score']:.4f}  "
                    f"clusters={r.get('n_clusters', '?')}  res={r.get('leiden_resolution', 0):.4f}"
                )

    if verbose:
        ranked = sorted(results.items(), key=lambda x: x[1]["avg_bio"], reverse=True)
        print(f"\n{'Rank':<5}{'Embedding':<35}{'Bio':<10}{'Batch':<10}{'Overall':<10}")
        print("-" * 70)
        for i, (name, r) in enumerate(ranked, 1):
            print(f"{i:<5}{name:<35}{r['avg_bio']:<10.4f}{r['avg_batch']:<10.4f}{r['overall_score']:<10.4f}")

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / f"{cell_type_key}.json", "w") as f:
            json.dump(_to_serializable(results), f, indent=4)
        if verbose:
            print(f"\nSaved to {save_path / f'{cell_type_key}.json'}")

    return results


def _eval_multi_single(adata, emb, key, batch_key, unknown_handling, silhouette_subsample, seed: int, resolutions: list[float] | None = None, neighbor_metric: str = "euclidean"):
    """Worker: compute neighbors + metrics for one (embedding, key) pair."""
    ad = _clean_labels(adata, key, unknown_handling)

    with suppress_output():
        sc.pp.neighbors(ad, use_rep=emb, random_state=seed, metric=neighbor_metric)
        r = _compute_metrics_with_neighbors(
            ad, batch_key, key, emb,
            silhouette_subsample=silhouette_subsample,
            seed=seed,
            resolutions=resolutions,
        )
    return emb, key, r


def benchmark_embeddings_multi(
    adata,
    embeddings: list[str],
    cell_type_keys: list[str],
    batch_key: str,
    unknown_handling: str = "filter",
    verbose: bool = True,
    save_dir: str | Path | None = None,
    n_jobs: int = 1,
    silhouette_subsample: int | None = None,
    seed: int = 0,
    resolutions: list[float] | None = None,
    l2_normalize: bool = False,
    neighbor_metric: str = "euclidean",
) -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    """Benchmark embeddings across multiple cell type keys."""
    if l2_normalize:
        adata = adata.copy()
        if verbose:
            print("L2-normalizing embeddings...")
        _l2_normalize_embeddings(adata, embeddings, verbose=verbose)

    missing = [emb for emb in embeddings if emb not in adata.obsm]
    if missing:
        raise KeyError("Missing embedding key(s) in adata.obsm: " + ", ".join(missing))

    n_total = len(embeddings) * len(cell_type_keys)
    if verbose:
        sil_msg = f", silhouette on {silhouette_subsample} cells" if silhouette_subsample else ""
        par_msg = f", {n_jobs} workers" if n_jobs > 1 else ""
        print(f"\nBenchmarking {len(embeddings)} embeddings x {len(cell_type_keys)} keys = {n_total} evaluations ({adata.n_obs} cells{sil_msg}{par_msg})")

    per_key_results = {key: {} for key in cell_type_keys}

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {
                ex.submit(
                    _eval_multi_single,
                    adata, emb, key, batch_key, unknown_handling, silhouette_subsample, seed, resolutions, neighbor_metric,
                ): (emb, key)
                for emb in embeddings
                for key in cell_type_keys
            }
            pbar = tqdm(total=n_total, desc="Evaluations (parallel)", disable=not verbose)
            for fut in as_completed(futures):
                emb, key, r = fut.result()
                per_key_results[key][emb] = r
                pbar.update(1)
                if verbose:
                    tqdm.write(f"    {emb} | {key}: bio={r['avg_bio']:.4f}  batch={r['avg_batch']:.4f}")
            pbar.close()
    else:
        pbar = tqdm(total=n_total, desc="Total progress", disable=not verbose)
        for i_emb, emb in enumerate(embeddings, 1):
            if verbose:
                tqdm.write(f"\n  [{i_emb}/{len(embeddings)}] {emb}: computing neighbors...")
            ad_base = adata.copy()
            with suppress_output():
                sc.pp.neighbors(ad_base, use_rep=emb, random_state=seed, metric=neighbor_metric)
            if verbose:
                tqdm.write(f"  [{i_emb}/{len(embeddings)}] {emb}: neighbors done, evaluating {len(cell_type_keys)} keys...")

            for key in cell_type_keys:
                ad = _clean_labels(ad_base, key, unknown_handling)
                if unknown_handling == "filter":
                    with suppress_output():
                        sc.pp.neighbors(ad, use_rep=emb, random_state=seed, metric=neighbor_metric)

                with suppress_output():
                    r = _compute_metrics_with_neighbors(
                        ad, batch_key, key, emb,
                        silhouette_subsample=silhouette_subsample,
                        seed=seed,
                        resolutions=resolutions,
                    )
                per_key_results[key][emb] = r
                pbar.update(1)
                if verbose:
                    tqdm.write(f"    {emb} | {key}: bio={r['avg_bio']:.4f}  batch={r['avg_batch']:.4f}")

            del ad_base
        pbar.close()

    aggregated = {}
    for emb in embeddings:
        scores = {"overall_score": [], "avg_bio": [], "avg_batch": []}
        for key in cell_type_keys:
            if emb in per_key_results[key]:
                r = per_key_results[key][emb]
                scores["overall_score"].append(r["overall_score"])
                scores["avg_bio"].append(r["avg_bio"])
                scores["avg_batch"].append(r["avg_batch"])

        if scores["overall_score"]:
            aggregated[emb] = {
                "avg_bio": np.mean(scores["avg_bio"]),
                "avg_batch": np.mean(scores["avg_batch"]),
                "overall_score": np.mean(scores["overall_score"]),
            }

    if verbose:
        for key in cell_type_keys:
            print(f"\n{'='*60}")
            print(f"Results: {key}")
            print(f"{'='*60}")
            ranked = sorted(per_key_results[key].items(), key=lambda x: x[1]["avg_bio"], reverse=True)
            print(f"{'Rank':<5}{'Embedding':<35}{'Bio':<10}{'Batch':<10}{'Overall':<10}")
            print("-" * 70)
            for i, (name, r) in enumerate(ranked, 1):
                print(f"{i:<5}{name:<35}{r['avg_bio']:<10.4f}{r['avg_batch']:<10.4f}{r['overall_score']:<10.4f}")

        print(f"\n{'='*60}")
        print(f"AGGREGATED RESULTS (mean across {len(cell_type_keys)} keys)")
        print(f"{'='*60}")
        ranked = sorted(aggregated.items(), key=lambda x: x[1]["avg_bio"], reverse=True)
        print(f"\n{'Rank':<5}{'Embedding':<35}{'Bio':<10}{'Batch':<10}{'Overall':<10}")
        print("-" * 70)
        for i, (name, r) in enumerate(ranked, 1):
            print(f"{i:<5}{name:<35}{r['avg_bio']:<10.4f}{r['avg_batch']:<10.4f}{r['overall_score']:<10.4f}")

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        out = {"per_key": per_key_results, "aggregated": aggregated}
        fname = f"multi_{'_'.join(cell_type_keys[:3])}.json"
        with open(save_path / fname, "w") as f:
            json.dump(_to_serializable(out), f, indent=4)
        if verbose:
            print(f"\nSaved to {save_path / fname}")

    return per_key_results, aggregated


# ---------------------------------------------------------------------------
# Subsampling-based robustness intervals
# ---------------------------------------------------------------------------

def _stratified_subsample(adata, label_key, frac, rng):
    """Stratified subsample without replacement. Returns sorted index array."""
    labels = adata.obs[label_key]
    idx = np.arange(adata.n_obs)
    keep = []
    for lab in labels.cat.categories:
        mask = (labels == lab).values
        lab_idx = idx[mask]
        n = max(1, int(len(lab_idx) * frac))
        keep.append(rng.choice(lab_idx, size=n, replace=False))
    result = np.concatenate(keep)
    result.sort()
    return result


def benchmark_embeddings_with_subsampling(
    adata,
    embeddings: list[str],
    batch_key: str,
    cell_type_key: str,
    unknown_handling: str = "filter",
    n_repeats: int = 10,
    subsample_frac: float = 0.8,
    n_jobs: int = 1,
    silhouette_subsample: int | None = None,
    verbose: bool = True,
    save_dir: str | Path | None = None,
    base_seed: int = 0,
    resolutions: list[float] | None = None,
    fixed_resolutions: bool = False,
    l2_normalize: bool = False,
    neighbor_metric: str = "euclidean",
) -> dict:
    """Paired repeated subsampling benchmark for robustness intervals."""
    adata_prep = _prepare_adata(
        adata,
        cell_type_key,
        unknown_handling,
        embeddings=embeddings,
        l2_normalize=l2_normalize,
        verbose=verbose,
    )

    per_repeat = []
    metrics_of_interest = [
        "avg_bio", "avg_batch", "overall_score",
        "NMI_cluster/label", "ARI_cluster/label", "ASW_label",
        "ASW_label/batch", "graph_conn",
    ]

    fixed_res = None
    if fixed_resolutions:
        if verbose:
            print("\nPrecomputing optimal Leiden resolution per embedding (full data)...")
        fixed_res = {}
        for emb in tqdm(embeddings, desc="  Resolutions", disable=not verbose):
            fixed_res[emb] = _find_optimal_resolution(
                adata_prep, label_key=cell_type_key, embed_key=emb,
                seed=base_seed, resolutions=resolutions,
                neighbor_metric=neighbor_metric,
            )
        if verbose:
            for emb, r in fixed_res.items():
                print(f"  {emb}: fixed_resolution={r:.4f}")

    for i in range(n_repeats):
        seed_i = base_seed + i
        rng = np.random.default_rng(seed_i)
        idx = _stratified_subsample(adata_prep, cell_type_key, subsample_frac, rng)
        ad_sub = adata_prep[idx].copy()

        if verbose:
            print(f"\n--- Repeat {i+1}/{n_repeats} (seed={seed_i}, {ad_sub.n_obs}/{adata_prep.n_obs} cells) ---")

        results_i = {}
        algo_seed = base_seed
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = {
                    ex.submit(
                        _eval_single_embedding,
                        ad_sub, emb, batch_key, cell_type_key,
                        silhouette_subsample, algo_seed, resolutions,
                        (fixed_res[emb] if fixed_res is not None else None),
                        neighbor_metric,
                    ): emb
                    for emb in embeddings
                }
                pbar = tqdm(total=len(embeddings), desc=f"  Repeat {i+1}", disable=not verbose)
                for fut in as_completed(futures):
                    emb, r = fut.result()
                    results_i[emb] = r
                    pbar.update(1)
                    if verbose:
                        tqdm.write(
                            f"    {emb}: bio={r['avg_bio']:.4f}  batch={r['avg_batch']:.4f}  "
                            f"clusters={r.get('n_clusters', '?')}  res={r.get('leiden_resolution', 0):.4f}  "
                            f"NMI={r.get('NMI_cluster/label', 0):.4f}  "
                            f"ARI={r.get('ARI_cluster/label', 0):.4f}  ASW={r.get('ASW_label', 0):.4f}"
                        )
                pbar.close()
        else:
            for emb in embeddings:
                _, r = _eval_single_embedding(
                    ad_sub, emb, batch_key, cell_type_key,
                    silhouette_subsample, algo_seed, resolutions,
                    (fixed_res[emb] if fixed_res is not None else None),
                    neighbor_metric,
                )
                results_i[emb] = r
                if verbose:
                    print(
                        f"    {emb}: bio={r['avg_bio']:.4f}  batch={r['avg_batch']:.4f}  "
                        f"clusters={r.get('n_clusters', '?')}  res={r.get('leiden_resolution', 0):.4f}  "
                        f"NMI={r.get('NMI_cluster/label', 0):.4f}  "
                        f"ARI={r.get('ARI_cluster/label', 0):.4f}  ASW={r.get('ASW_label', 0):.4f}"
                    )
        per_repeat.append(results_i)

    summary = {}
    for emb in embeddings:
        summary[emb] = {}
        for metric in metrics_of_interest:
            vals = [rep[emb].get(metric, np.nan) for rep in per_repeat]
            vals = np.array(vals)
            valid = vals[~np.isnan(vals)]
            n = len(valid)
            mean = float(np.mean(valid))
            median = float(np.median(valid))
            std = float(np.std(valid, ddof=1)) if n > 1 else 0.0
            se = std / np.sqrt(n) if n > 1 else 0.0
            t_crit = float(t_dist.ppf(0.975, df=n - 1)) if n > 1 else 0.0
            margin = t_crit * se
            summary[emb][metric] = {
                "mean": mean,
                "median": median,
                "std": std,
                "ci_lo": mean - margin,
                "ci_hi": mean + margin,
                "q_lo": float(np.quantile(valid, 0.025)),
                "q_hi": float(np.quantile(valid, 0.975)),
                "values": vals.tolist(),
            }

    significance = {}
    for metric in ["avg_bio", "avg_batch", "overall_score"]:
        pairs = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                a, b = embeddings[i], embeddings[j]
                va = np.array(summary[a][metric]["values"])
                vb = np.array(summary[b][metric]["values"])
                diff = va - vb
                mean_diff = float(np.mean(diff))
                if np.all(diff == 0):
                    p = 1.0
                else:
                    _, p = wilcoxon(va, vb, alternative="two-sided")
                pairs.append({
                    "emb_a": a,
                    "emb_b": b,
                    "mean_diff": mean_diff,
                    "p_value": float(p),
                    "significant": p < 0.05,
                })
        significance[metric] = pairs

    if verbose:
        print(f"\n{'='*80}")
        print(f"Summary ({n_repeats} repeats, {subsample_frac:.0%} subsample)")
        print(f"{'='*80}")
        ranked = sorted(summary.items(), key=lambda x: x[1]["avg_bio"]["mean"], reverse=True)
        print(f"\n{'Rank':<5}{'Embedding':<35}{'avgBIO':<20}{'avgBatch':<20}{'Overall':<20}")
        print("-" * 100)
        for rank, (emb, s) in enumerate(ranked, 1):
            bio = s["avg_bio"]
            bat = s["avg_batch"]
            ov = s["overall_score"]
            print(
                f"{rank:<5}{emb:<35}"
                f"{bio['mean']:.4f}+/-{bio['std']:.4f}  "
                f"{bat['mean']:.4f}+/-{bat['std']:.4f}  "
                f"{ov['mean']:.4f}+/-{ov['std']:.4f}"
            )

        sig_bio = significance["avg_bio"]
        n_sig = sum(1 for p in sig_bio if p["significant"])
        print(f"\nPairwise significance (avg_bio, Wilcoxon): {n_sig}/{len(sig_bio)} pairs significant at p<0.05")
        nonsig = [p for p in sig_bio if not p["significant"]]
        if nonsig:
            print("Non-significant pairs (indistinguishable):")
            for p in nonsig:
                print(f"  {p['emb_a']} vs {p['emb_b']}: diff={p['mean_diff']:.4f}, p={p['p_value']:.4f}")

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        out = {
            "summary": summary,
            "significance": significance,
            "n_repeats": n_repeats,
            "subsample_frac": subsample_frac,
        }
        fname = f"{cell_type_key}_subsampled.json"
        with open(save_path / fname, "w") as f:
            json.dump(_to_serializable(out), f, indent=4)
        if verbose:
            print(f"\nSaved to {save_path / fname}")

    return {"per_repeat": per_repeat, "summary": summary, "significance": significance}
