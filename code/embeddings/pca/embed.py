"""Reproducible PCA cell embeddings for full and ortholog-tier gene sets.

The default recipe intentionally does no preprocessing: PCA is fit directly on
``adata.X`` for the selected gene set. Optional recipes mirror the exploratory
Scanpy notebook patterns for normalize/log1p, HVG+scale, and HVG+scale with
ComBat batch correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORK_DIR = ROOT / "data" / "pca" / "work"
DEFAULT_RESULTS_DIR = ROOT / "results" / "embeddings"

_EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent
if str(_EMBEDDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(_EMBEDDINGS_DIR))

DEFAULT_CACHE_DIR = ROOT / "data" / "pca" / ".cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR / "xdg"))

import anndata as ad
import h5py
import numpy as np

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

RECIPES = {
    "raw": "No preprocessing; fit PCA directly on the selected expression matrix.",
    "normalized_log1p": "Normalize total counts per cell and log1p before PCA.",
    "hvg_scaled": "Select HVGs, normalize/log1p, scale, then PCA.",
    "hvg_scaled_batchcorrected": "Select HVGs by batch, normalize/log1p, scale, ComBat-correct, then PCA.",
}
COUNTS_LAYER = "counts"


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


def write_obsm_array(h5ad_path: Path, key: str, array: np.ndarray) -> None:
    with h5py.File(h5ad_path, "a") as handle:
        if "obsm" not in handle:
            handle.create_group("obsm")
        obsm = handle["obsm"]
        if key in obsm:
            del obsm[key]
        write_elem(obsm, key, np.asarray(array))


def h5ad_has_obsm_keys(h5ad_path: Path, keys: Iterable[str]) -> bool:
    with h5py.File(h5ad_path, "r") as handle:
        obsm = handle.get("obsm")
        return obsm is not None and all(key in obsm for key in keys)


def embedding_key_for(variant: str, recipe: str, suffix: str = "") -> str:
    recipe_part = "" if recipe == "raw" else f"_{recipe}"
    if variant == "full":
        return f"X_pca{recipe_part}{suffix}"
    return f"X_pca{recipe_part}_ortholog_{variant}{suffix}"


def prepare_full_anndata(source_path: Path, max_cells: int | None) -> tuple[ad.AnnData, dict[str, int]]:
    adata = ad.read_h5ad(source_path)
    if max_cells is not None:
        adata = adata[:max_cells, :].copy()
    else:
        adata = adata.copy()
    return adata, {
        "n_cells": int(adata.n_obs),
        "n_source_genes": int(adata.n_vars),
        "n_selected_ortholog_symbols": 0,
        "n_unique_ids": int(adata.n_vars),
        "n_collapsed_paralog_rows": 0,
    }


def prepare_ortholog_anndata(
    source_path: Path,
    tier_columns: tuple[str, ...],
    max_cells: int | None,
) -> tuple[ad.AnnData, dict[str, int]]:
    adata = ad.read_h5ad(source_path)
    if max_cells is not None:
        adata = adata[:max_cells, :].copy()

    symbols = selected_symbols(adata.var, tier_columns)
    usable = symbols.map(valid_symbol)
    if int(usable.sum()) == 0:
        raise ValueError(f"No usable ortholog symbols for columns {tier_columns}")

    tmp = adata[:, usable.to_numpy()].copy()
    tmp.var["pca_gene_symbol"] = symbols.loc[usable].astype(str).str.strip().to_numpy()
    n_pre_collapse = int(tmp.n_vars)
    tmp, n_unique_ids, n_collapsed = collapse_duplicate_columns(tmp, "pca_gene_symbol")
    tmp.var.index.name = None

    return tmp, {
        "n_cells": int(tmp.n_obs),
        "n_source_genes": int(adata.n_vars),
        "n_selected_ortholog_symbols": int(usable.sum()),
        "n_mapped_ids": n_pre_collapse,
        "n_unique_ids": int(n_unique_ids),
        "n_collapsed_paralog_rows": int(n_collapsed),
    }


def effective_n_comps(adata: ad.AnnData, requested: int) -> int:
    n_comps = min(int(requested), int(adata.n_obs) - 1, int(adata.n_vars) - 1)
    if n_comps < 1:
        raise ValueError(f"Need at least 2 cells and 2 genes for PCA; got shape {adata.shape}")
    return n_comps


def preprocess_for_recipe(
    adata: ad.AnnData,
    *,
    recipe: str,
    n_top_genes: int,
    hvg_flavor: str,
    batch_key: str,
) -> tuple[ad.AnnData, dict[str, object]]:
    import scanpy as sc

    work = adata.copy()
    meta: dict[str, object] = {
        "recipe": recipe,
        "n_input_cells": int(work.n_obs),
        "n_input_genes": int(work.n_vars),
    }
    if recipe == "raw":
        meta["preprocessing"] = []
        return work, meta

    if recipe == "normalized_log1p":
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
        meta["preprocessing"] = ["normalize_total_target_sum_1e4", "log1p"]
        return work, meta

    work.layers[COUNTS_LAYER] = work.X.copy()
    hvg_kwargs: dict[str, object] = {
        "n_top_genes": min(int(n_top_genes), int(work.n_vars)),
        "flavor": hvg_flavor,
        "layer": COUNTS_LAYER,
    }
    if recipe == "hvg_scaled_batchcorrected":
        if batch_key not in work.obs.columns:
            raise ValueError(f"Missing obs batch_key '{batch_key}'")
        hvg_kwargs["batch_key"] = batch_key

    sc.pp.highly_variable_genes(work, **hvg_kwargs)
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    work = work[:, work.var["highly_variable"]].copy()
    sc.pp.scale(work, max_value=10)
    steps = ["highly_variable_genes", "normalize_total_target_sum_1e4", "log1p", "scale_max_value_10"]
    if recipe == "hvg_scaled_batchcorrected":
        sc.pp.combat(work, key=batch_key)
        steps.append(f"combat_{batch_key}")

    meta.update(
        {
            "preprocessing": steps,
            "hvg_flavor": hvg_flavor,
            "n_top_genes": int(hvg_kwargs["n_top_genes"]),
            "n_hvg_genes": int(work.n_vars),
            "batch_key": batch_key if recipe == "hvg_scaled_batchcorrected" else None,
        }
    )
    return work, meta


def compute_pca(
    adata: ad.AnnData,
    *,
    requested_n_comps: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    import scanpy as sc

    n_comps = effective_n_comps(adata, requested_n_comps)
    sc.tl.pca(adata, n_comps=n_comps, random_state=seed)
    return np.asarray(adata.obsm["X_pca"], dtype=np.float32), n_comps


def run_variant(
    *,
    cfg: DatasetConfig,
    variant: str,
    recipes: list[str],
    work_dir: Path,
    n_comps: int,
    n_top_genes: int,
    hvg_flavor: str,
    batch_key: str,
    max_cells: int | None,
    seed: int,
    overwrite: bool,
) -> dict[str, object]:
    if variant == "full":
        adata, prep = prepare_full_anndata(cfg.h5ad_path, max_cells)
    else:
        adata, prep = prepare_ortholog_anndata(cfg.h5ad_path, VARIANTS[variant], max_cells)

    variant_dir = work_dir / cfg.label / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    recipe_summaries: list[dict[str, object]] = []
    for recipe in recipes:
        pca_key = embedding_key_for(variant, recipe)
        pca50_key = (
            embedding_key_for(variant, recipe, "_50")
            if min(n_comps, adata.n_obs - 1, adata.n_vars - 1) >= 50
            else None
        )
        expected_keys = [pca_key, *([pca50_key] if pca50_key is not None else [])]
        stem = f"{cfg.label}_{variant}_{recipe}_{n_comps}"
        pca_npy = variant_dir / f"{stem}_pca.npy"
        pca50_npy = variant_dir / f"{stem}_pca50.npy"
        if max_cells is None:
            outputs_exist = h5ad_has_obsm_keys(cfg.h5ad_path, expected_keys)
        else:
            outputs_exist = pca_npy.exists() and (pca50_key is None or pca50_npy.exists())

        if outputs_exist and not overwrite:
            print(f"=== PCA {cfg.name}: {variant} / {recipe} exists; skipping (use --overwrite) ===", flush=True)
            recipe_summaries.append(
                {
                    "recipe": recipe,
                    "skipped_existing": True,
                    "obsm_keys_written": [],
                    "smoke_embedding_arrays": (
                        [str(pca_npy), str(pca50_npy)]
                        if max_cells is not None and pca50_key is not None
                        else [str(pca_npy)]
                        if max_cells is not None
                        else []
                    ),
                }
            )
            continue

        print(f"=== PCA {cfg.name}: {variant} / {recipe} ===", flush=True)
        work, recipe_meta = preprocess_for_recipe(
            adata,
            recipe=recipe,
            n_top_genes=n_top_genes,
            hvg_flavor=hvg_flavor,
            batch_key=batch_key,
        )
        pca, effective_components = compute_pca(work, requested_n_comps=n_comps, seed=seed)
        if effective_components < 50:
            pca50_key = None

        smoke_outputs: list[str] = []
        written_keys: list[str] = []
        if max_cells is None:
            write_obsm_array(cfg.h5ad_path, pca_key, pca)
            written_keys.append(pca_key)
            if pca50_key is not None:
                write_obsm_array(cfg.h5ad_path, pca50_key, pca[:, :50])
                written_keys.append(pca50_key)
        else:
            np.save(pca_npy, pca)
            smoke_outputs.append(str(pca_npy))
            if pca50_key is not None:
                np.save(pca50_npy, pca[:, :50])
                smoke_outputs.append(str(pca50_npy))

        recipe_summaries.append(
            {
                **recipe_meta,
                "skipped_existing": False,
                "requested_n_components": int(n_comps),
                "effective_n_components": int(effective_components),
                "embedding_shape": list(pca.shape),
                "pca50_shape": [int(pca.shape[0]), 50] if pca50_key is not None else None,
                "obsm_keys_written": written_keys,
                "smoke_embedding_arrays": smoke_outputs,
            }
        )

    return {
        "variant": variant,
        "tier_columns": list(VARIANTS[variant]),
        **prep,
        "recipes": recipe_summaries,
    }


def run_dataset(
    *,
    cfg: DatasetConfig,
    variants: list[str],
    recipes: list[str],
    work_dir: Path,
    results_dir: Path,
    n_comps: int,
    n_top_genes: int,
    hvg_flavor: str,
    batch_key: str,
    max_cells: int | None,
    seed: int,
    overwrite: bool,
    manifest_tag: str,
) -> dict[str, object]:
    if not cfg.h5ad_path.exists():
        raise FileNotFoundError(cfg.h5ad_path)
    set_global_seeds(seed)
    h5ad_sha256_before = sha256_file(cfg.h5ad_path) if max_cells is None else None

    variant_summaries = [
        run_variant(
            cfg=cfg,
            variant=variant,
            recipes=recipes,
            work_dir=work_dir,
            n_comps=n_comps,
            n_top_genes=n_top_genes,
            hvg_flavor=hvg_flavor,
            batch_key=batch_key,
            max_cells=max_cells,
            seed=seed,
            overwrite=overwrite,
        )
        for variant in variants
    ]

    h5ad_sha256_after = sha256_file(cfg.h5ad_path) if max_cells is None else None
    manifest = {
        "step": "pca_embed",
        "dataset": cfg.name,
        "h5ad_path": str(cfg.h5ad_path),
        "h5ad_sha256_before": h5ad_sha256_before,
        "h5ad_sha256_after": h5ad_sha256_after,
        "work_dir": str(work_dir),
        "variants": variant_summaries,
        "runtime": {
            "variants": variants,
            "recipes": recipes,
            "n_components": int(n_comps),
            "n_top_genes": int(n_top_genes),
            "hvg_flavor": hvg_flavor,
            "batch_key": batch_key,
            "max_cells": max_cells,
            "seed": int(seed),
            "overwrite": overwrite,
            "manifest_tag": manifest_tag,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{manifest_tag}" if manifest_tag else ""
    suffix = "_smoke" if max_cells is not None else ""
    out = results_dir / f"pca_{cfg.name}{tag}{suffix}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {out}", flush=True)
    return manifest


def parse_csv(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def parse_allowed_csv(raw: str, allowed: set[str], label: str) -> list[str]:
    values = parse_csv(raw)
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"Unknown {label} {unknown}; choose from {sorted(allowed)}")
    return values


def require_known_paths(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA embeddings for full and ortholog-tier variants.")
    parser.add_argument("--dataset", choices=[*sorted(DATASETS), "all"], default="all")
    parser.add_argument("--variants", default="full,tier1,tier12,tier123")
    parser.add_argument(
        "--recipes",
        default="raw",
        help=f"Comma-separated recipes. Choices: {','.join(RECIPES)}. Default is no preprocessing.",
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n-comps", type=int, default=100)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--hvg-flavor", default="seurat_v3")
    parser.add_argument("--batch-key", default="time")
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke-test subset; writes .npy files instead of H5AD.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing obsm/smoke outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest-tag",
        default="",
        help=(
            "Optional filesystem-safe tag inserted in the manifest filename. "
            "Use this for sensitivity runs so the primary raw-PCA manifest is preserved."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest_tag and not args.manifest_tag.replace("-", "").replace("_", "").isalnum():
        raise ValueError("--manifest-tag may contain only letters, numbers, hyphens and underscores")
    variants = parse_allowed_csv(args.variants, set(VARIANTS), "variants")
    recipes = parse_allowed_csv(args.recipes, set(RECIPES), "recipes")
    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    require_known_paths([DATASETS[name].h5ad_path for name in datasets])
    for name in datasets:
        run_dataset(
            cfg=DATASETS[name],
            variants=variants,
            recipes=recipes,
            work_dir=args.work_dir,
            results_dir=args.results_dir,
            n_comps=args.n_comps,
            n_top_genes=args.n_top_genes,
            hvg_flavor=args.hvg_flavor,
            batch_key=args.batch_key,
            max_cells=args.max_cells,
            seed=args.seed,
            overwrite=args.overwrite,
            manifest_tag=args.manifest_tag,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
