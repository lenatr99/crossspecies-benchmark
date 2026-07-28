"""Compute pseudotime trajectory-conservation scores for embedding keys."""

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

import scanpy as sc

from util.pseudotime_presets import (
    DEFAULT_PSEUDOTIME_EMBEDDINGS,
    PSEUDOTIME_EMBEDDING_PRESETS,
)

PRESETS = {
    "dicty": {
        "data": ROOT / "data" / "dicty.h5ad",
        "time_key": "time",
        "save_dir": ROOT / "results" / "pseudotime_dicty",
    },
    "elegans": {
        "data": ROOT / "data" / "elegans.h5ad",
        "time_key": "embryo.time",
        "save_dir": ROOT / "results" / "pseudotime_elegans",
    },
}


def parse_csv(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "dicty", "elegans"], default="all")
    parser.add_argument("--data", type=Path, default=None, help="Override input H5AD for a single dataset run.")
    parser.add_argument("--time-key", default="", help="Override ground-truth time column for a single dataset run.")
    parser.add_argument(
        "--embedding-preset",
        choices=sorted(PSEUDOTIME_EMBEDDING_PRESETS),
        default="main",
        help="Named embedding-key preset. Ignored when --embeddings is provided.",
    )
    parser.add_argument(
        "--embeddings",
        default="",
        help="Comma-separated obsm keys to score. Empty uses --embedding-preset and skips preset keys missing from the current H5AD.",
    )
    parser.add_argument("--save-dir", type=Path, default=None, help="Override output directory for a single dataset run.")
    parser.add_argument(
        "--timestamped-save-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write results under <save-dir>/runs/<timestamp> to avoid "
            "overwriting previous pseudotime JSONs. Use --no-timestamped-save-dir "
            "to restore legacy overwrite behavior."
        ),
    )
    parser.add_argument(
        "--inference",
        choices=["subsample", "bootstrap"],
        default="subsample",
        help=(
            "Uncertainty method. 'subsample' (default): subsample cells without "
            "replacement and recompute the neighbor graph + DPT per repeat "
            "(Design A; field-standard trajectory robustness). 'bootstrap': "
            "legacy paired bootstrap over a precomputed pseudotime matrix."
        ),
    )
    parser.add_argument("--n-candidates", type=int, default=500, help="Number of candidate root cells to sample.")
    parser.add_argument("--root-method", choices=["uniform", "stratified"], default="uniform")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap iterations (--inference bootstrap only).")
    parser.add_argument("--n-repeats", type=int, default=30, help="Subsampling repeats (--inference subsample only).")
    parser.add_argument("--subsample-frac", type=float, default=0.9, help="Cell fraction retained per subsample repeat.")
    parser.add_argument("--k-roots", type=int, default=20, help="Roots per replicate/repeat.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Interval alpha level.")
    parser.add_argument(
        "--bootstrap-consensus-roots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resample roots when computing consensus-metric uncertainty.",
    )
    parser.add_argument(
        "--write-h5ad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the per-embedding DPT pseudotime columns back into the input H5AD.",
    )
    parser.add_argument("--max-cells", type=int, default=0, help="Optional smoke-test subset using the first N cells.")
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


def run_dataset(dataset: str, args: argparse.Namespace) -> dict[str, object]:
    preset = PRESETS[dataset]
    data_path = args.data if args.data is not None else preset["data"]
    base_save_dir = args.save_dir if args.save_dir is not None else preset["save_dir"]
    save_dir = (
        base_save_dir / "runs" / run_timestamp(args)
        if args.timestamped_save_dir
        else base_save_dir
    )
    time_key = args.time_key.strip() or str(preset["time_key"])

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    print(f"\n# {dataset}: reading {data_path}")
    if args.embeddings.strip():
        embeddings = parse_csv(args.embeddings)
    else:
        preset_embeddings = PSEUDOTIME_EMBEDDING_PRESETS.get(
            args.embedding_preset,
            DEFAULT_PSEUDOTIME_EMBEDDINGS,
        )
        adata = sc.read_h5ad(data_path)
        available = set(adata.obsm.keys())
        embeddings = [key for key in preset_embeddings if key in available]
        skipped = [key for key in preset_embeddings if key not in available]
        if skipped:
            print(
                "[pseudotime] skipping preset embedding(s) absent from this H5AD: "
                + ", ".join(skipped)
            )
        if not embeddings:
            raise KeyError(
                "None of the default pseudotime embeddings are present in adata.obsm. "
                f"Available keys include: {list(adata.obsm.keys())[:50]}"
            )
        print(f"[pseudotime] embedding preset={args.embedding_preset} ({len(embeddings)} present)")
        del adata

    if args.max_cells > 0:
        raise NotImplementedError(
            "--max-cells subsetting requires writing a temporary H5AD; not yet supported."
        )

    start = time.time()
    print(f"\n[pseudotime] dataset={dataset} time_key={time_key} inference={args.inference} "
          f"n_candidates={args.n_candidates}")
    if args.inference == "subsample":
        from util.benchmark_pseudotime import benchmark_pseudotime_subsampling

        results, _distinguishability = benchmark_pseudotime_subsampling(
            data=data_path,
            embeddings=embeddings,
            time_key=time_key,
            n_repeats=args.n_repeats,
            subsample_frac=args.subsample_frac,
            k_roots=args.k_roots,
            n_candidates=args.n_candidates,
            alpha=args.alpha,
            verbose=not args.quiet,
            save_dir=save_dir,
        )
    else:
        from util.benchmark_pseudotime import benchmark_pseudotime

        results, _significance = benchmark_pseudotime(
            data=data_path,
            embeddings=embeddings,
            time_key=time_key,
            n_candidates=args.n_candidates,
            root_method=args.root_method,
            n_boot=args.n_boot,
            k_roots=args.k_roots,
            alpha=args.alpha,
            verbose=not args.quiet,
            save_dir=save_dir,
            write_h5ad=args.write_h5ad,
            bootstrap_consensus_roots=args.bootstrap_consensus_roots,
        )
    elapsed = time.time() - start
    print(f"[pseudotime] {dataset} finished in {elapsed:.1f}s")
    return {
        "dataset": dataset,
        "time_key": time_key,
        "inference": args.inference,
        "base_save_dir": str(base_save_dir),
        "save_dir": str(save_dir),
        "timestamped_save_dir": bool(args.timestamped_save_dir),
        "elapsed_seconds": elapsed,
        "embedding_preset": args.embedding_preset if not args.embeddings.strip() else "custom",
        "n_embeddings_ok": len(results),
        "n_candidates": args.n_candidates,
        "root_method": args.root_method,
        "n_repeats": args.n_repeats,
        "subsample_frac": args.subsample_frac,
        "k_roots": args.k_roots,
    }


def main() -> None:
    args = parse_args()
    all_summaries = []
    for dataset in selected_presets(args):
        all_summaries.append(run_dataset(dataset, args))

    summary_name = (
        f"pseudotime_run_summary_{run_timestamp(args)}.json"
        if args.timestamped_save_dir
        else "pseudotime_run_summary.json"
    )
    out = ROOT / "results" / "evaluation" / summary_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summaries": all_summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n[pseudotime] wrote {out}")


if __name__ == "__main__":
    main()
