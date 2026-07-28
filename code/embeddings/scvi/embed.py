"""Reproducible scVI cell embeddings for full and ortholog-tier gene sets.

This script follows the exploratory scVI notebook pattern: copy ``X`` into a
``counts`` layer, train an ``SCVI`` model with a negative-binomial likelihood,
store the latent representation in ``obsm``, and optionally repeat with a
batch key such as ``time``. It keeps training in memory and disables Lightning
checkpoint/log outputs so runs do not leave large temporary artifacts behind.
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
DEFAULT_WORK_DIR = ROOT / "data" / "scvi" / "work"
DEFAULT_RESULTS_DIR = ROOT / "results" / "embeddings"

_EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent
if str(_EMBEDDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(_EMBEDDINGS_DIR))

DEFAULT_CACHE_DIR = ROOT / "data" / "scvi" / ".cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR / "xdg"))
os.environ.setdefault("SCVI_DATA_DIR", str(DEFAULT_CACHE_DIR / "scvi"))

import anndata as ad
import h5py
import numpy as np
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

MODES = ("embedding", "batchcorrected")
COUNTS_LAYER = "counts"


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import scvi

    scvi.settings.seed = seed


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


def embedding_key_for(variant: str, mode: str) -> str:
    if variant == "full":
        return "X_scVI" if mode == "embedding" else "X_scVI_batchcorrected"
    if mode == "embedding":
        return f"X_scVI_ortholog_{variant}"
    return f"X_scVI_batchcorrected_ortholog_{variant}"


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
    tmp.var["scvi_gene_symbol"] = symbols.loc[usable].astype(str).str.strip().to_numpy()
    n_pre_collapse = int(tmp.n_vars)
    tmp, n_unique_ids, n_collapsed = collapse_duplicate_columns(tmp, "scvi_gene_symbol")
    tmp.var.index.name = None

    return tmp, {
        "n_cells": int(tmp.n_obs),
        "n_source_genes": int(adata.n_vars),
        "n_selected_ortholog_symbols": int(usable.sum()),
        "n_mapped_ids": n_pre_collapse,
        "n_unique_ids": int(n_unique_ids),
        "n_collapsed_paralog_rows": int(n_collapsed),
    }


def add_counts_layer(adata: ad.AnnData) -> None:
    adata.layers[COUNTS_LAYER] = adata.X.copy()


def train_scvi_latent(
    adata: ad.AnnData,
    *,
    n_latent: int,
    n_layers: int,
    gene_likelihood: str,
    batch_key: str | None,
    max_epochs: int | None,
    accelerator: str,
    devices: str,
) -> np.ndarray:
    import scvi

    setup_kwargs = {"layer": COUNTS_LAYER}
    if batch_key is not None:
        if batch_key not in adata.obs.columns:
            raise ValueError(f"Missing obs batch_key '{batch_key}'")
        setup_kwargs["batch_key"] = batch_key

    scvi.model.SCVI.setup_anndata(adata, **setup_kwargs)
    model = scvi.model.SCVI(
        adata,
        n_layers=n_layers,
        n_latent=n_latent,
        gene_likelihood=gene_likelihood,
    )

    train_kwargs: dict[str, object] = {
        "accelerator": accelerator,
        "devices": int(devices) if devices.isdigit() else devices,
        "enable_checkpointing": False,
        "logger": False,
    }
    if max_epochs is not None:
        train_kwargs["max_epochs"] = max_epochs
    model.train(**train_kwargs)
    return np.asarray(model.get_latent_representation(), dtype=np.float32)


def pca50(matrix: np.ndarray, seed: int) -> np.ndarray:
    n_components = min(50, matrix.shape[0], matrix.shape[1])
    return PCA(n_components=n_components, random_state=seed).fit_transform(matrix).astype(np.float32)


def run_variant(
    *,
    cfg: DatasetConfig,
    variant: str,
    modes: list[str],
    work_dir: Path,
    n_latent: int,
    n_layers: int,
    gene_likelihood: str,
    batch_key: str,
    max_epochs: int | None,
    max_cells: int | None,
    accelerator: str,
    devices: str,
    seed: int,
    overwrite: bool,
) -> dict[str, object]:
    if variant == "full":
        adata, prep = prepare_full_anndata(cfg.h5ad_path, max_cells)
    else:
        adata, prep = prepare_ortholog_anndata(cfg.h5ad_path, VARIANTS[variant], max_cells)
    add_counts_layer(adata)

    variant_dir = work_dir / cfg.label / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    mode_summaries: list[dict[str, object]] = []
    for mode in modes:
        emb_key = embedding_key_for(variant, mode)
        pca_key = f"{emb_key}_50"
        emb_npy = variant_dir / f"{cfg.label}_{variant}_{mode}_embedding.npy"
        pca_npy = variant_dir / f"{cfg.label}_{variant}_{mode}_pca.npy"

        if max_cells is None:
            outputs_exist = h5ad_has_obsm_keys(cfg.h5ad_path, [emb_key, pca_key])
        else:
            outputs_exist = emb_npy.exists() and pca_npy.exists()

        if outputs_exist and not overwrite:
            print(f"=== scVI {cfg.name}: {variant} / {mode} exists; skipping (use --overwrite) ===", flush=True)
            mode_summaries.append(
                {
                    "mode": mode,
                    "batch_key": batch_key if mode == "batchcorrected" else None,
                    "skipped_existing": True,
                    "obsm_keys_written": [],
                    "smoke_embedding_arrays": [str(emb_npy), str(pca_npy)] if max_cells is not None else [],
                }
            )
            continue

        print(f"=== scVI {cfg.name}: {variant} / {mode} ===", flush=True)
        latent = train_scvi_latent(
            adata.copy(),
            n_latent=n_latent,
            n_layers=n_layers,
            gene_likelihood=gene_likelihood,
            batch_key=batch_key if mode == "batchcorrected" else None,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
        )
        pca = pca50(latent, seed)

        smoke_outputs: list[str] = []
        if max_cells is None:
            write_obsm_array(cfg.h5ad_path, emb_key, latent)
            write_obsm_array(cfg.h5ad_path, pca_key, pca)
        else:
            np.save(emb_npy, latent)
            np.save(pca_npy, pca)
            smoke_outputs.extend([str(emb_npy), str(pca_npy)])

        mode_summaries.append(
            {
                "mode": mode,
                "batch_key": batch_key if mode == "batchcorrected" else None,
                "skipped_existing": False,
                "embedding_shape": list(latent.shape),
                "pca_shape": list(pca.shape),
                "obsm_keys_written": [emb_key, pca_key] if max_cells is None else [],
                "smoke_embedding_arrays": smoke_outputs,
            }
        )

    return {
        "variant": variant,
        "tier_columns": list(VARIANTS[variant]),
        **prep,
        "modes": mode_summaries,
    }


def run_dataset(
    *,
    cfg: DatasetConfig,
    variants: list[str],
    modes: list[str],
    work_dir: Path,
    results_dir: Path,
    n_latent: int,
    n_layers: int,
    gene_likelihood: str,
    batch_key: str,
    max_epochs: int | None,
    max_cells: int | None,
    accelerator: str,
    devices: str,
    seed: int,
    overwrite: bool,
) -> dict[str, object]:
    if not cfg.h5ad_path.exists():
        raise FileNotFoundError(cfg.h5ad_path)
    set_global_seeds(seed)
    h5ad_sha256_before = sha256_file(cfg.h5ad_path)

    variant_summaries = [
        run_variant(
            cfg=cfg,
            variant=variant,
            modes=modes,
            work_dir=work_dir,
            n_latent=n_latent,
            n_layers=n_layers,
            gene_likelihood=gene_likelihood,
            batch_key=batch_key,
            max_epochs=max_epochs,
            max_cells=max_cells,
            accelerator=accelerator,
            devices=devices,
            seed=seed,
            overwrite=overwrite,
        )
        for variant in variants
    ]

    manifest = {
        "step": "scvi_embed",
        "dataset": cfg.name,
        "h5ad_path": str(cfg.h5ad_path),
        "h5ad_sha256_before": h5ad_sha256_before,
        "work_dir": str(work_dir),
        "variants": variant_summaries,
        "runtime": {
            "n_latent": n_latent,
            "n_layers": n_layers,
            "gene_likelihood": gene_likelihood,
            "batch_key": batch_key,
            "max_epochs": max_epochs,
            "max_cells": max_cells,
            "accelerator": accelerator,
            "devices": devices,
            "seed": seed,
            "overwrite": overwrite,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if max_cells is not None else ""
    out = results_dir / f"scvi_{cfg.name}{suffix}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {out}", flush=True)
    return manifest


def parse_csv(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def parse_variants(raw: str) -> list[str]:
    variants = parse_csv(raw)
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {sorted(VARIANTS)}")
    return variants


def parse_modes(raw: str) -> list[str]:
    modes = parse_csv(raw)
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"Unknown modes {unknown}; choose from {list(MODES)}")
    return modes


def require_known_paths(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train scVI cell embeddings and store the latent representation "
            "(plus a 50-dim PCA of it) for one or more gene-set variants. Full "
            "runs patch the source H5AD's obsm in place; --max-cells smoke runs "
            "write .npy files instead."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default: all variants, both datasets, no batch correction\n"
            "  python embed.py\n\n"
            "  # Quick smoke test on 256 cells (writes .npy, not H5AD)\n"
            "  python embed.py --dataset dicty --variants full --max-cells 256 --overwrite\n\n"
            "  # Batch-corrected embedding keyed on 'time'\n"
            "  python embed.py --dataset elegans --modes batchcorrected --batch-key time"
        ),
    )

    data = parser.add_argument_group("data selection")
    data.add_argument(
        "--dataset",
        choices=[*sorted(DATASETS), "all"],
        default="all",
        help="Which dataset(s) to embed.",
    )
    data.add_argument(
        "--variants",
        default="full,tier1,tier12,tier123",
        help=(
            "Comma-separated gene-set variants: full (no ortholog mapping, all "
            "measured genes), tier1 (ortholog_tier_1), tier12 (tier 1 + tier 2 "
            "fallback), tier123 (tiers 1->3)."
        ),
    )
    data.add_argument(
        "--modes",
        default="embedding",
        help=(
            "Comma-separated modes: embedding (no batch correction) and/or "
            "batchcorrected (conditions on --batch-key)."
        ),
    )

    model = parser.add_argument_group("scVI model / training")
    model.add_argument("--n-latent", type=int, default=100, help="Latent-space dimensionality.")
    model.add_argument("--n-layers", type=int, default=2, help="Number of hidden layers in the encoder/decoder.")
    model.add_argument(
        "--gene-likelihood",
        choices=["nb", "zinb", "poisson"],
        default="nb",
        help="Reconstruction likelihood for the decoder.",
    )
    model.add_argument(
        "--batch-key",
        default="time",
        help="obs column used as the batch covariate in batchcorrected mode.",
    )
    model.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Cap on training epochs (default: scVI's own heuristic).",
    )
    model.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Subsample to the first N cells for a smoke test; writes .npy instead of patching H5AD.",
    )
    model.add_argument("--seed", type=int, default=42, help="Global RNG seed (Python, NumPy, scVI, PCA).")

    hardware = parser.add_argument_group("hardware")
    hardware.add_argument(
        "--accelerator",
        default="mps",
        help="Lightning accelerator, e.g. mps, gpu, cpu.",
    )
    hardware.add_argument(
        "--devices",
        default="1",
        help="Devices passed to Lightning (integer count or 'auto').",
    )

    io = parser.add_argument_group("output")
    io.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="Scratch dir for per-variant smoke arrays.")
    io.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Where run manifests are written.")
    io.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and replace existing obsm keys / smoke outputs instead of skipping them.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = parse_variants(args.variants)
    modes = parse_modes(args.modes)
    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    require_known_paths([DATASETS[name].h5ad_path for name in datasets])
    for name in datasets:
        run_dataset(
            cfg=DATASETS[name],
            variants=variants,
            modes=modes,
            work_dir=args.work_dir,
            results_dir=args.results_dir,
            n_latent=args.n_latent,
            n_layers=args.n_layers,
            gene_likelihood=args.gene_likelihood,
            batch_key=args.batch_key,
            max_epochs=args.max_epochs,
            max_cells=args.max_cells,
            accelerator=args.accelerator,
            devices=args.devices,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
