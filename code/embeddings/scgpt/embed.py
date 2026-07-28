"""Reproducible scGPT cell embeddings for full and ortholog-tier gene sets.

This script replaces the exploratory scGPT notebooks with a rerunnable
pipeline for Dictyostelium discoideum and C. elegans. It prepares scGPT-ready
AnnData views, runs ``scgpt.tasks.embed_data`` against the pretrained
checkpoint, PCA-reduces the cell embeddings, and writes coexistable arrays back
to the base H5AD ``obsm``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT / "models" / "scGPT"
DEFAULT_WORK_DIR = ROOT / "data" / "scgpt" / "work"
DEFAULT_RESULTS_DIR = ROOT / "results" / "embeddings"
SCGPT_MODEL_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing"
)

_EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent
if str(_EMBEDDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(_EMBEDDINGS_DIR))

DEFAULT_CACHE_DIR = ROOT / "data" / "scgpt" / ".cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR / "xdg"))

# scGPT's data loader expects this Linux API in some dependency paths. The
# notebook used the same patch for macOS.
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda _pid: list(range(os.cpu_count() or 1))  # type: ignore[attr-defined]

try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from _utils import collapse_duplicate_columns, selected_symbols, valid_symbol

try:
    from anndata.io import write_elem
except ImportError:  # pragma: no cover - older anndata fallback
    from anndata.experimental import write_elem


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    h5ad_path: Path
    label: str


DATASETS = {
    "dicty": DatasetConfig(
        name="dicty",
        h5ad_path=ROOT / "data" / "dicty.h5ad",
        label="dicty",
    ),
    "elegans": DatasetConfig(
        name="elegans",
        h5ad_path=ROOT / "data" / "elegans.h5ad",
        label="elegans",
    ),
}

VARIANTS = {
    "full": (),
    "tier1": ("ortholog_tier_1",),
    "tier12": ("ortholog_tier_1", "ortholog_tier_2"),
    "tier123": ("ortholog_tier_1", "ortholog_tier_2", "ortholog_tier_3"),
}


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def write_obsm_array(h5ad_path: Path, key: str, array: np.ndarray) -> None:
    with h5py.File(h5ad_path, "a") as handle:
        if "obsm" not in handle:
            handle.create_group("obsm")
        obsm = handle["obsm"]
        if key in obsm:
            del obsm[key]
        write_elem(obsm, key, np.asarray(array))


def load_scgpt_vocab(model_dir: Path) -> set[str]:
    vocab_path = model_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(vocab_path)
    with vocab_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        return {str(k).strip() for k in raw}
    if isinstance(raw, list):
        return {str(x).strip() for x in raw}
    raise ValueError(f"Unsupported scGPT vocab format in {vocab_path}")


def map_symbol_to_vocab(symbol: object, vocab: set[str]) -> str | None:
    text = str(symbol).strip()
    if not valid_symbol(text):
        return None
    upper = text.upper()
    if upper in vocab:
        return upper
    if text in vocab:
        return text
    return None


def ensure_model_assets(model_dir: Path, *, download: bool) -> list[str]:
    required_names = ("args.json", "best_model.pt", "vocab.json")
    required = [model_dir / name for name in required_names]
    missing = [p for p in required if not p.exists()]
    downloaded: list[str] = []
    if missing and download:
        try:
            import gdown
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                f"Missing scGPT assets under {model_dir}. Install gdown or place "
                "args.json, best_model.pt, and vocab.json there."
            ) from exc
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading scGPT checkpoint into {model_dir}", flush=True)
        gdown.download_folder(SCGPT_MODEL_FOLDER_URL, output=str(model_dir), quiet=False)
        nested_model_dirs = [
            path
            for path in model_dir.rglob("*")
            if path.is_dir() and all((path / name).exists() for name in required_names)
        ]
        if nested_model_dirs:
            asset_dir = nested_model_dirs[0]
            if asset_dir != model_dir:
                print(f"Found scGPT checkpoint files under {asset_dir}; moving them into {model_dir}", flush=True)
                for name in required_names:
                    target = model_dir / name
                    if not target.exists():
                        shutil.move(str(asset_dir / name), str(target))
        downloaded.append(str(model_dir))

    still_missing = [str(p) for p in required if not p.exists()]
    if still_missing:
        raise FileNotFoundError("Missing required scGPT model file(s): " + ", ".join(still_missing))
    return downloaded


def prepare_full_h5ad(
    *,
    source_path: Path,
    output_path: Path,
    vocab: set[str],
    max_cells: int | None,
) -> dict[str, int]:
    adata = ad.read_h5ad(source_path)
    if max_cells is not None:
        adata = adata[:max_cells, :].copy()

    if "Gene Symbol" in adata.var.columns:
        symbols = adata.var["Gene Symbol"].astype(str)
    elif "gene_symbols" in adata.var.columns:
        symbols = adata.var["gene_symbols"].astype(str)
    elif "gene_name" in adata.var.columns:
        symbols = adata.var["gene_name"].astype(str)
    else:
        symbols = pd.Series(adata.var_names.astype(str), index=adata.var.index)

    vocab_symbols = symbols.map(lambda x: map_symbol_to_vocab(x, vocab))
    usable = vocab_symbols.notna()
    if int(usable.sum()) == 0:
        raise ValueError(f"No full-dataset genes from {source_path} matched the scGPT vocabulary")

    tmp = adata[:, usable.to_numpy()].copy()
    tmp.var["scgpt_gene_symbol"] = vocab_symbols.loc[usable].astype(str).to_numpy()
    tmp, n_unique_ids, n_collapsed = collapse_duplicate_columns(tmp, "scgpt_gene_symbol")
    tmp.var["Gene Symbol"] = tmp.var["scgpt_gene_symbol"].astype(str)
    tmp.var.index.name = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    remove_path(output_path)
    tmp.write_h5ad(output_path)

    return {
        "n_cells": int(tmp.n_obs),
        "n_source_genes": int(adata.n_vars),
        "n_selected_ortholog_symbols": 0,
        "n_mapped_ids": int(usable.sum()),
        "n_unique_ids": int(n_unique_ids),
        "n_collapsed_paralog_rows": int(n_collapsed),
    }


def prepare_ortholog_h5ad(
    *,
    source_path: Path,
    output_path: Path,
    tier_columns: tuple[str, ...],
    vocab: set[str],
    max_cells: int | None,
) -> dict[str, int]:
    adata = ad.read_h5ad(source_path)
    if max_cells is not None:
        adata = adata[:max_cells, :].copy()

    symbols = selected_symbols(adata.var, tier_columns)
    vocab_symbols = symbols.map(lambda x: map_symbol_to_vocab(x, vocab))
    usable = vocab_symbols.notna()
    if int(usable.sum()) == 0:
        raise ValueError(f"No scGPT-vocabulary genes for columns {tier_columns}")

    tmp = adata[:, usable.to_numpy()].copy()
    tmp.var["scgpt_gene_symbol"] = vocab_symbols.loc[usable].astype(str).to_numpy()
    n_pre_collapse = int(tmp.n_vars)
    tmp, n_unique_ids, n_collapsed = collapse_duplicate_columns(tmp, "scgpt_gene_symbol")
    tmp.var["Gene Symbol"] = tmp.var["scgpt_gene_symbol"].astype(str)
    tmp.var.index.name = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    remove_path(output_path)
    tmp.write_h5ad(output_path)

    return {
        "n_cells": int(tmp.n_obs),
        "n_source_genes": int(adata.n_vars),
        "n_selected_ortholog_symbols": int(symbols.map(valid_symbol).sum()),
        "n_mapped_ids": n_pre_collapse,
        "n_unique_ids": int(n_unique_ids),
        "n_collapsed_paralog_rows": int(n_collapsed),
    }


def run_variant(
    *,
    cfg: DatasetConfig,
    variant: str,
    model_dir: Path,
    work_dir: Path,
    vocab: set[str],
    batch_size: int,
    max_cells: int | None,
    overwrite: bool,
) -> dict[str, object]:
    import scgpt as scg

    variant_dir = work_dir / cfg.label / variant
    input_h5ad = variant_dir / f"{cfg.label}_{variant}.h5ad"
    if overwrite:
        remove_path(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)

    if variant == "full":
        prep = prepare_full_h5ad(
            source_path=cfg.h5ad_path,
            output_path=input_h5ad,
            vocab=vocab,
            max_cells=max_cells,
        )
        emb_key = "X_scGPT_full"
    else:
        prep = prepare_ortholog_h5ad(
            source_path=cfg.h5ad_path,
            output_path=input_h5ad,
            tier_columns=VARIANTS[variant],
            vocab=vocab,
            max_cells=max_cells,
        )
        emb_key = f"X_scGPT_ortholog_{variant}"

    try:
        adata_in = ad.read_h5ad(input_h5ad)
        embedded = scg.tasks.embed_data(
            adata_in,
            model_dir,
            gene_col="Gene Symbol",
            batch_size=batch_size,
        )
    finally:
        remove_path(input_h5ad)
    matrix = np.asarray(embedded.obsm["X_scGPT"], dtype=np.float32)
    pca_n = min(50, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=pca_n, random_state=42).fit_transform(matrix).astype(np.float32)

    pca_key = f"{emb_key}_50"
    smoke_outputs: list[str] = []
    if max_cells is None:
        write_obsm_array(cfg.h5ad_path, emb_key, matrix)
        write_obsm_array(cfg.h5ad_path, pca_key, pca)
    else:
        emb_npy = variant_dir / f"{cfg.label}_{variant}_embedding.npy"
        pca_npy = variant_dir / f"{cfg.label}_{variant}_pca.npy"
        np.save(emb_npy, matrix)
        np.save(pca_npy, pca)
        smoke_outputs.extend([str(emb_npy), str(pca_npy)])

    return {
        "variant": variant,
        "tier_columns": list(VARIANTS[variant]),
        **prep,
        "n_scgpt_vocab_size": int(len(vocab)),
        "embedding_shape": list(matrix.shape),
        "pca_shape": list(pca.shape),
        "obsm_keys_written": [emb_key, pca_key] if max_cells is None else [],
        "smoke_embedding_arrays": smoke_outputs,
        "temporary_h5ad_removed": str(input_h5ad),
    }


def run_dataset(
    *,
    cfg: DatasetConfig,
    variants: list[str],
    model_dir: Path,
    work_dir: Path,
    results_dir: Path,
    batch_size: int,
    max_cells: int | None,
    overwrite: bool,
    download_model: bool,
    seed: int,
) -> dict[str, object]:
    if not cfg.h5ad_path.exists():
        raise FileNotFoundError(cfg.h5ad_path)
    set_global_seeds(seed)
    downloaded = ensure_model_assets(model_dir, download=download_model)
    vocab = load_scgpt_vocab(model_dir)
    h5ad_sha256_before = sha256_file(cfg.h5ad_path)

    variant_summaries = []
    for variant in variants:
        print(f"=== scGPT {cfg.name}: {variant} ===", flush=True)
        variant_summaries.append(
            run_variant(
                cfg=cfg,
                variant=variant,
                model_dir=model_dir,
                work_dir=work_dir,
                vocab=vocab,
                batch_size=batch_size,
                max_cells=max_cells,
                overwrite=overwrite,
            )
        )

    manifest = {
        "step": "scgpt_embed",
        "dataset": cfg.name,
        "h5ad_path": str(cfg.h5ad_path),
        "h5ad_sha256_before": h5ad_sha256_before,
        "model_dir": str(model_dir),
        "model_files_downloaded": downloaded,
        "work_dir": str(work_dir),
        "variants": variant_summaries,
        "runtime": {
            "batch_size": batch_size,
            "max_cells": max_cells,
            "overwrite": overwrite,
            "seed": seed,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if max_cells is not None else ""
    out = results_dir / f"scgpt_{cfg.name}{suffix}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {out}", flush=True)
    return manifest


def parse_variants(raw: str) -> list[str]:
    variants = [piece.strip() for piece in raw.split(",") if piece.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {sorted(VARIANTS)}")
    return variants


def require_known_paths(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scGPT embeddings for full or ortholog tier variants.")
    parser.add_argument("--dataset", choices=[*sorted(DATASETS), "all"], default="all")
    parser.add_argument("--variants", default="tier1,tier12,tier123")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Delete and rebuild per-variant work dirs.")
    parser.add_argument("--download-model", action="store_true", help="Download checkpoint files with gdown if missing.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = parse_variants(args.variants)
    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    require_known_paths([DATASETS[name].h5ad_path for name in datasets])
    for name in datasets:
        run_dataset(
            cfg=DATASETS[name],
            variants=variants,
            model_dir=args.model_dir,
            work_dir=args.work_dir,
            results_dir=args.results_dir,
            batch_size=args.batch_size,
            max_cells=args.max_cells,
            overwrite=args.overwrite,
            download_model=args.download_model,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
