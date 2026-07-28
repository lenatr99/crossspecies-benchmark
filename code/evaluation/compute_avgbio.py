"""Compute avgBIO-style biological benchmark scores for embedding keys."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CACHE_DIR = ROOT / "data" / "evaluation" / ".cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR / "xdg"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import scanpy as sc

from util.avgbio_presets import AVGBIO_EMBEDDING_PRESETS, DEFAULT_AVGBIO_EMBEDDINGS

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

PRESETS = {
    "dicty": {
        "data": ROOT / "data" / "dicty.h5ad",
        "batch_key": "time",
        "label_jobs": (
            ("cell_type", "filter", 0),
            ("cAMP_genes", "category", 0),
            ("milestones", "filter", 0),
        ),
        "save_dir": ROOT / "results" / "biological_benchmarks_dicty",
    },
    "elegans": {
        "data": ROOT / "data" / "elegans.h5ad",
        "batch_key": "batch",
        "label_jobs": (
            ("cell_type", "filter", 0),
            ("developmental_markers", "filter", 0),
            ("lineage_markers", "filter", 0),
        ),
        "save_dir": ROOT / "results" / "biological_benchmarks_elegans",
    },
}


def parse_csv(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def parse_resolutions(raw: str) -> list[float] | None:
    raw = raw.strip()
    if not raw:
        return None
    return [float(x) for x in parse_csv(raw)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "dicty", "elegans"], default="all")
    parser.add_argument("--data", type=Path, default=None, help="Override input H5AD for a single dataset run.")
    parser.add_argument("--batch-key", default="", help="Override batch key for a single dataset run.")
    parser.add_argument("--label-keys", default="", help="Comma-separated label keys. Defaults to preset labels.")
    parser.add_argument("--unknown-handling", choices=["preset", "filter", "category"], default="preset")
    parser.add_argument(
        "--embedding-preset",
        choices=sorted(AVGBIO_EMBEDDING_PRESETS),
        default="main",
        help="Named embedding-key preset. Ignored when --embeddings is provided.",
    )
    parser.add_argument("--embeddings", default="", help="Comma-separated obsm keys to score. Empty uses --embedding-preset and skips preset keys missing from the current H5AD.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Override output directory for a single dataset run.")
    parser.add_argument(
        "--timestamped-save-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write results under <save-dir>/runs/<timestamp> to avoid "
            "overwriting previous avgBIO JSONs. Use --no-timestamped-save-dir "
            "to restore legacy overwrite behavior."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=0, help="Override process workers. 0 picks the largest exact-mode worker count that should fit in memory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolutions", default="", help="Comma-separated Leiden resolutions. Empty uses scIB default.")
    parser.add_argument("--l2-normalize", action="store_true", default=False)
    parser.add_argument("--neighbor-metric", default="euclidean")
    parser.add_argument("--max-cells", type=int, default=0, help="Optional smoke-test subset using the first N cells.")
    parser.add_argument(
        "--subsample-repeats",
        type=int,
        default=0,
        help=(
            "If >0, additionally run paired repeated-subsampling avgBIO and write "
            "<label>_subsampled.json (per-embedding mean/std avgBIO across repeats "
            "plus pairwise Wilcoxon significance). 0 disables; the point-estimate "
            "<label>.json is always written."
        ),
    )
    parser.add_argument(
        "--subsample-frac",
        type=float,
        default=0.9,
        help="Fraction of cells retained per subsampling repeat (matches the pseudotime benchmark's 0.9).",
    )
    parser.add_argument(
        "--subsample-fixed-resolutions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, reuse the Leiden resolution optimized once on full data instead of "
            "re-optimizing every repeat. DEFAULT OFF: fixing the resolution biases methods "
            "sitting on a cluster-count boundary (e.g. UCE on the binary dicty cell_type "
            "label collapses from 2 to 3 clusters), so each repeat re-optimizes the "
            "resolution over --subsample-resolutions, matching the exact point-estimate estimator."
        ),
    )
    parser.add_argument(
        "--subsample-resolutions",
        default="0.05,0.1,0.2,0.3,0.4,0.5,0.7,1.0",
        help=(
            "Bounded Leiden resolution grid re-optimized within each subsampling repeat "
            "(keeps re-optimization affordable). Empty uses the full scIB default grid. "
            "Only affects the interval JSON; the point-estimate <label>.json keeps --resolutions."
        ),
    )
    parser.add_argument(
        "--subsample-silhouette",
        type=int,
        default=10000,
        help=(
            "Cells to subsample for the silhouette (ASW) term inside subsampling repeats. "
            "0 uses exact ASW. Only affects the interval width, not the exact point-estimate JSON."
        ),
    )
    parser.add_argument(
        "--subsample-max-cells",
        type=int,
        default=20000,
        help=(
            "Cap labeled cells per subsampling repeat: effective frac = min(--subsample-frac, "
            "cap/n_labeled). Keeps large panels (49k/90k cells) as fast as small ones; "
            "NMI/ARI/ASW are stable at this size. 0 disables the cap."
        ),
    )
    parser.add_argument(
        "--subsample-only",
        action="store_true",
        help=(
            "Skip the exact point-estimate <label>.json (reuse the pinned point estimates) and "
            "compute only the subsampling intervals. Big speedup: avoids exact full-data silhouette."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def selected_presets(args: argparse.Namespace) -> list[str]:
    return ["dicty", "elegans"] if args.dataset == "all" else [args.dataset]


def run_timestamp(args: argparse.Namespace) -> str:
    value = getattr(args, "_run_timestamp", "")
    if not value:
        value = datetime.now().strftime("%Y%m%d_%H%M%S")
        setattr(args, "_run_timestamp", value)
    return value


def label_jobs_for(dataset: str, args: argparse.Namespace) -> list[tuple[str, str, int]]:
    preset_jobs = list(PRESETS[dataset]["label_jobs"])
    requested = parse_csv(args.label_keys)
    if requested:
        preset_by_key = {key: (key, handling, n_jobs) for key, handling, n_jobs in preset_jobs}
        jobs = [preset_by_key.get(key, (key, "filter", 6)) for key in requested]
    else:
        jobs = preset_jobs
    if args.unknown_handling != "preset":
        jobs = [(key, args.unknown_handling, n_jobs) for key, _, n_jobs in jobs]
    if args.n_jobs > 0:
        jobs = [(key, handling, args.n_jobs) for key, handling, _ in jobs]
    return jobs


def _visible_cpu_count() -> int:
    raw = os.environ.get("EVALUATION_DOCKER_CPUS", "").strip()
    if raw:
        try:
            return max(1, int(float(raw)))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def _visible_memory_bytes() -> int | None:
    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for path in candidates:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw and raw != "max":
            value = int(raw)
            if 0 < value < 1 << 60:
                return value
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def _label_cell_count(adata, label_key: str, unknown_handling: str) -> int:
    if unknown_handling == "category":
        return int(adata.n_obs)
    labels = adata.obs[label_key]
    as_string = labels.astype("object").astype(str).str.strip()
    missing = labels.isna() | as_string.isin(UNKNOWN_LABELS)
    return int((~missing).sum())


def _format_gib(n_bytes: int | None) -> str:
    if n_bytes is None:
        return "unknown"
    return f"{n_bytes / (1024 ** 3):.1f} GiB"


def auto_n_jobs(adata, label_key: str, unknown_handling: str, n_embeddings: int) -> int:
    """Pick aggressive-but-bounded parallelism for exact full-silhouette runs."""
    cpus = _visible_cpu_count()
    mem = _visible_memory_bytes()
    n_cells = max(1, _label_cell_count(adata, label_key, unknown_handling))

    if mem is None:
        return max(1, min(n_embeddings, cpus // 2))

    distance_bytes = n_cells * n_cells * 8
    worker_bytes = max(10 * 1024**3, int(distance_bytes * 2.5))
    usable_bytes = int(mem * 0.75)
    by_mem = max(1, usable_bytes // worker_bytes)
    return max(1, min(n_embeddings, cpus, int(by_mem)))


def run_dataset(dataset: str, args: argparse.Namespace) -> list[dict[str, object]]:
    preset = PRESETS[dataset]
    data_path = args.data if args.data is not None else preset["data"]
    base_save_dir = args.save_dir if args.save_dir is not None else preset["save_dir"]
    save_dir = (
        base_save_dir / "runs" / run_timestamp(args)
        if args.timestamped_save_dir
        else base_save_dir
    )
    batch_key = args.batch_key.strip() or str(preset["batch_key"])
    resolutions = parse_resolutions(args.resolutions)

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    print(f"\n# {dataset}: reading {data_path}")
    adata = sc.read_h5ad(data_path)
    if args.embeddings.strip():
        embeddings = parse_csv(args.embeddings)
    else:
        preset_embeddings = AVGBIO_EMBEDDING_PRESETS.get(
            args.embedding_preset,
            DEFAULT_AVGBIO_EMBEDDINGS,
        )
        available = set(adata.obsm.keys())
        embeddings = [key for key in preset_embeddings if key in available]
        skipped = [key for key in preset_embeddings if key not in available]
        if skipped:
            print(
                "[avgBIO] skipping preset embedding(s) absent from this H5AD: "
                + ", ".join(skipped)
            )
        if not embeddings:
            raise KeyError(
                "None of the default avgBIO embeddings are present in adata.obsm. "
                f"Available keys include: {list(adata.obsm.keys())[:50]}"
            )
        print(f"[avgBIO] embedding preset={args.embedding_preset} ({len(embeddings)} present)")
    if args.max_cells > 0:
        adata = adata[: args.max_cells].copy()
        print(f"[avgBIO] using first {adata.n_obs} cells (--max-cells)")
    summaries = []
    for label_key, unknown_handling, n_jobs in label_jobs_for(dataset, args):
        if n_jobs <= 0:
            n_jobs = auto_n_jobs(adata, label_key, unknown_handling, len(embeddings))
            print(
                f"[avgBIO] auto workers for exact silhouette: n_jobs={n_jobs} "
                f"(cells={_label_cell_count(adata, label_key, unknown_handling)}, "
                f"cpus={_visible_cpu_count()}, memory={_format_gib(_visible_memory_bytes())})"
            )
        start = time.time()
        print(f"\n[avgBIO] dataset={dataset} label={label_key} batch={batch_key} unknown={unknown_handling}")
        from util.benchmark_embeddings import benchmark_embeddings

        if args.subsample_only:
            results = embeddings  # reuse pinned point estimates; only intervals are recomputed
            print(f"[avgBIO] {dataset}/{label_key} skipping exact point estimate (--subsample-only)")
        else:
            results = benchmark_embeddings(
                adata,
                embeddings,
                batch_key,
                label_key,
                unknown_handling=unknown_handling,
                verbose=not args.quiet,
                save_dir=save_dir,
                n_jobs=n_jobs,
                seed=args.seed,
                resolutions=resolutions,
                silhouette_subsample=None,
                l2_normalize=args.l2_normalize,
                neighbor_metric=args.neighbor_metric,
            )
        elapsed = time.time() - start
        print(f"[avgBIO] {dataset}/{label_key} finished in {elapsed:.1f}s")

        if args.subsample_repeats > 0:
            from util.benchmark_embeddings import benchmark_embeddings_with_subsampling

            sub_start = time.time()
            n_labeled = _label_cell_count(adata, label_key, unknown_handling)
            sub_frac = args.subsample_frac
            if args.subsample_max_cells > 0 and n_labeled > 0:
                sub_frac = min(sub_frac, args.subsample_max_cells / n_labeled)
            print(
                f"[avgBIO] dataset={dataset} label={label_key} "
                f"subsampling: {args.subsample_repeats} repeats @ {sub_frac:.0%} "
                f"(~{int(sub_frac * n_labeled)} of {n_labeled} labeled cells)"
            )
            sub_resolutions = parse_resolutions(args.subsample_resolutions) or resolutions
            benchmark_embeddings_with_subsampling(
                adata,
                embeddings,
                batch_key,
                label_key,
                unknown_handling=unknown_handling,
                n_repeats=args.subsample_repeats,
                subsample_frac=sub_frac,
                n_jobs=n_jobs,
                silhouette_subsample=(args.subsample_silhouette or None),
                verbose=not args.quiet,
                save_dir=save_dir,
                base_seed=args.seed,
                resolutions=sub_resolutions,
                fixed_resolutions=args.subsample_fixed_resolutions,
                l2_normalize=args.l2_normalize,
                neighbor_metric=args.neighbor_metric,
            )
            print(
                f"[avgBIO] {dataset}/{label_key} subsampling finished in "
                f"{time.time() - sub_start:.1f}s"
            )

        summaries.append(
            {
                "dataset": dataset,
                "label_key": label_key,
                "batch_key": batch_key,
                "base_save_dir": str(base_save_dir),
                "save_dir": str(save_dir),
                "timestamped_save_dir": bool(args.timestamped_save_dir),
                "elapsed_seconds": elapsed,
                "embedding_preset": args.embedding_preset if not args.embeddings.strip() else "custom",
                "n_embeddings_ok": len(results),
                "silhouette_subsample": None,
                "subsample_repeats": int(args.subsample_repeats),
                "subsample_frac": float(args.subsample_frac) if args.subsample_repeats > 0 else None,
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    all_summaries = []
    for dataset in selected_presets(args):
        all_summaries.extend(run_dataset(dataset, args))

    summary_name = (
        f"avgBIO_run_summary_{run_timestamp(args)}.json"
        if args.timestamped_save_dir
        else "avgBIO_run_summary.json"
    )
    out = ROOT / "results" / "evaluation" / summary_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summaries": all_summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n[avgBIO] wrote {out}")


if __name__ == "__main__":
    main()
