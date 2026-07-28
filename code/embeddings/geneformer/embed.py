"""Reproducible Geneformer cell embeddings for ortholog-tier gene sets.

This script replaces the exploratory Geneformer notebooks with a rerunnable
pipeline for dicty (Dictyostelium discoideum) and C. elegans. It reads ortholog tier columns from
``adata.var``, maps the selected human symbols to Geneformer Ensembl IDs using
Geneformer's native mapping dictionary, extracts CLS cell embeddings, and
writes the resulting arrays back to the base H5AD ``obsm``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GENEFORMER_DIR = ROOT / "models" / "Geneformer"
DEFAULT_MODEL_DIR = DEFAULT_GENEFORMER_DIR / "gf-12L-95M-i4096"
DEFAULT_WORK_DIR = ROOT / "data" / "geneformer" / "work"
DEFAULT_RESULTS_DIR = ROOT / "results" / "embeddings"

# Make ``code/embeddings/_utils.py`` importable.
_EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent
if str(_EMBEDDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(_EMBEDDINGS_DIR))

# Geneformer imports Scanpy/Matplotlib/Numba. Keep their runtime caches out of
# user homes and read-only package directories, especially inside Docker.
DEFAULT_CACHE_DIR = ROOT / "data" / "geneformer" / ".cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR / "xdg"))

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.decomposition import PCA

from _utils import prepare_ortholog_h5ad

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
    "tier1": ("ortholog_tier_1",),
    "tier12": ("ortholog_tier_1", "ortholog_tier_2"),
    "tier123": ("ortholog_tier_1", "ortholog_tier_2", "ortholog_tier_3"),
}

COMPATIBILITY_ALIASES = {
    "tier12": {
        "embedding": "X_geneformer_orto",
        "pca50": "X_geneformer_orto_50",
    }
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def ensure_geneformer_import(geneformer_dir: Path) -> None:
    geneformer_dir = geneformer_dir.resolve()
    if not (geneformer_dir / "geneformer").exists():
        raise FileNotFoundError(f"Geneformer checkout missing package dir: {geneformer_dir}")
    if str(geneformer_dir) not in sys.path:
        sys.path.insert(0, str(geneformer_dir))


def apply_cpu_patch() -> None:
    import torch

    if getattr(torch, "_geneformer_cpu_patch_done", False):
        return

    orig_tensor = torch.tensor

    def tensor_cpu(*args, **kwargs):
        if isinstance(kwargs.get("device"), str) and kwargs["device"].startswith("cuda"):
            kwargs["device"] = "cpu"
        return orig_tensor(*args, **kwargs)

    orig_to = torch.Tensor.to

    def to_cpu(self, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("cuda"):
            args = ("cpu",) + args[1:]
        if isinstance(kwargs.get("device"), str) and kwargs["device"].startswith("cuda"):
            kwargs["device"] = "cpu"
        return orig_to(self, *args, **kwargs)

    torch.tensor = tensor_cpu
    torch.Tensor.to = to_cpu
    torch.cuda.is_available = lambda: False
    torch.cuda.empty_cache = lambda: None
    torch._geneformer_cpu_patch_done = True


def load_symbol_to_ensembl(geneformer_dir: Path) -> dict[str, str]:
    mapping_path = geneformer_dir / "geneformer" / "ensembl_mapping_dict_gc95M.pkl"
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)
    with mapping_path.open("rb") as handle:
        raw = pickle.load(handle)
    return {str(k).strip(): str(v).strip() for k, v in raw.items() if str(v).strip()}


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def add_cell_ids_to_dataset(dataset_path: Path, cell_ids: list[str], output_path: Path) -> None:
    ds = load_from_disk(str(dataset_path))
    if ds.num_rows != len(cell_ids):
        raise ValueError(
            f"Tokenized dataset row count {ds.num_rows} != H5AD cell count {len(cell_ids)}"
        )
    if "cell_id" in ds.column_names:
        ds = ds.remove_columns(["cell_id"])
    ds = ds.add_column("cell_id", cell_ids)
    remove_path(output_path)
    ds.save_to_disk(str(output_path))


def write_obsm_array(h5ad_path: Path, key: str, array: np.ndarray) -> None:
    with h5py.File(h5ad_path, "a") as handle:
        if "obsm" not in handle:
            handle.create_group("obsm")
        obsm = handle["obsm"]
        if key in obsm:
            del obsm[key]
        write_elem(obsm, key, np.asarray(array))


def extract_embedding_matrix(embs_df: pd.DataFrame, cell_order: list[str]) -> np.ndarray:
    if "cell_id" not in embs_df.columns:
        raise ValueError("Geneformer embeddings did not include a cell_id column")
    embs = embs_df.set_index("cell_id")
    missing = [c for c in cell_order if c not in embs.index]
    if missing:
        raise ValueError(f"Missing {len(missing)} cell embeddings; first missing: {missing[:3]}")
    numeric = embs.loc[cell_order].drop(columns=[c for c in embs.columns if c == "cell_id"], errors="ignore")
    numeric = numeric.apply(pd.to_numeric)
    return numeric.to_numpy(dtype=np.float32)


def run_variant(
    *,
    cfg: DatasetConfig,
    variant: str,
    geneformer_dir: Path,
    model_dir: Path,
    work_dir: Path,
    symbol_to_ensembl: dict[str, str],
    nproc: int,
    chunk_size: int,
    forward_batch_size: int,
    max_cells: int | None,
    overwrite: bool,
) -> dict[str, object]:
    from datasets import load_from_disk  # noqa: F401 - validates dependency in worker env
    from geneformer import EmbExtractor, TranscriptomeTokenizer

    variant_dir = work_dir / cfg.label / variant
    input_dir = variant_dir / "input"
    token_dir = variant_dir / "tokens"
    emb_dir = variant_dir / "embeddings"
    temp_h5ad = input_dir / f"{cfg.label}_{variant}.h5ad"
    token_prefix = f"{cfg.label}_{variant}"
    token_dataset = token_dir / f"{token_prefix}.dataset"
    token_dataset_ids = token_dir / f"{token_prefix}_ids.dataset"

    if overwrite:
        remove_path(variant_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    prep = prepare_ortholog_h5ad(
        source_path=cfg.h5ad_path,
        output_path=temp_h5ad,
        tier_columns=VARIANTS[variant],
        id_mapper=symbol_to_ensembl.get,
        id_column="ensembl_id",
        max_cells=max_cells,
    )

    remove_path(token_dataset)
    remove_path(token_dataset_ids)
    tokenizer = TranscriptomeTokenizer(
        {"cell_id": "cell_id"},
        nproc=nproc,
        chunk_size=chunk_size,
        model_input_size=4096,
        special_token=True,
    )
    tokenizer.tokenize_data(input_dir, token_dir, token_prefix, file_format="h5ad")

    tmp = ad.read_h5ad(temp_h5ad, backed="r")
    try:
        cell_order = list(tmp.obs_names.astype(str))
    finally:
        tmp.file.close()

    add_cell_ids_to_dataset(token_dataset, cell_order, token_dataset_ids)
    ds = load_from_disk(str(token_dataset_ids))
    unique_tokens = {tok for row in ds["input_ids"] for tok in row}

    embex = EmbExtractor(
        emb_mode="cls",
        emb_label=["cell_id"],
        max_ncells=None,
        forward_batch_size=forward_batch_size,
        nproc=nproc,
    )
    output_prefix = f"{cfg.label}_{variant}_embeddings"
    embs_df = embex.extract_embs(model_dir, token_dataset_ids, emb_dir, output_prefix)
    matrix = extract_embedding_matrix(embs_df, cell_order)
    pca_n = min(50, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=pca_n, random_state=42).fit_transform(matrix).astype(np.float32)

    emb_key = f"X_geneformer_ortholog_{variant}"
    pca_key = f"{emb_key}_50"
    alias_keys: list[str] = []
    smoke_outputs: list[str] = []

    if max_cells is None:
        write_obsm_array(cfg.h5ad_path, emb_key, matrix)
        write_obsm_array(cfg.h5ad_path, pca_key, pca)

        aliases = COMPATIBILITY_ALIASES.get(variant, {})
        if aliases:
            write_obsm_array(cfg.h5ad_path, aliases["embedding"], matrix)
            write_obsm_array(cfg.h5ad_path, aliases["pca50"], pca)
            alias_keys.extend([aliases["embedding"], aliases["pca50"]])
    else:
        emb_npy = emb_dir / f"{cfg.label}_{variant}_embedding.npy"
        pca_npy = emb_dir / f"{cfg.label}_{variant}_pca.npy"
        np.save(emb_npy, matrix)
        np.save(pca_npy, pca)
        smoke_outputs.extend([str(emb_npy), str(pca_npy)])

    tmp_for_vocab = ad.read_h5ad(temp_h5ad, backed="r")
    try:
        matched_vocab = sum(
            1 for ens in tmp_for_vocab.var["ensembl_id"].astype(str)
            if ens in tokenizer.gene_token_dict
        )
    finally:
        tmp_for_vocab.file.close()
    remove_path(temp_h5ad)

    return {
        "variant": variant,
        "tier_columns": list(VARIANTS[variant]),
        **prep,
        "n_geneformer_vocab_matched_genes": int(matched_vocab),
        "n_unique_tokens": int(len(unique_tokens)),
        "embedding_shape": list(matrix.shape),
        "pca_shape": list(pca.shape),
        "obsm_keys_written": [emb_key, pca_key, *alias_keys] if max_cells is None else [],
        "smoke_embedding_arrays": smoke_outputs,
        "temporary_h5ad_removed": str(temp_h5ad),
        "tokenized_dataset": str(token_dataset_ids),
        "embedding_csv": str((emb_dir / output_prefix).with_suffix(".csv")),
    }


def run_dataset(
    *,
    cfg: DatasetConfig,
    variants: list[str],
    geneformer_dir: Path,
    model_dir: Path,
    work_dir: Path,
    results_dir: Path,
    nproc: int,
    chunk_size: int,
    forward_batch_size: int,
    max_cells: int | None,
    overwrite: bool,
    force_cpu: bool,
) -> dict[str, object]:
    if force_cpu:
        apply_cpu_patch()
    ensure_geneformer_import(geneformer_dir)
    symbol_to_ensembl = load_symbol_to_ensembl(geneformer_dir)

    if not cfg.h5ad_path.exists():
        raise FileNotFoundError(cfg.h5ad_path)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    h5ad_sha256_before = sha256_file(cfg.h5ad_path)
    variant_summaries = []
    for variant in variants:
        print(f"=== Geneformer {cfg.name}: {variant} ===", flush=True)
        variant_summaries.append(
            run_variant(
                cfg=cfg,
                variant=variant,
                geneformer_dir=geneformer_dir,
                model_dir=model_dir,
                work_dir=work_dir,
                symbol_to_ensembl=symbol_to_ensembl,
                nproc=nproc,
                chunk_size=chunk_size,
                forward_batch_size=forward_batch_size,
                max_cells=max_cells,
                overwrite=overwrite,
            )
        )

    manifest = {
        "step": "geneformer_embed",
        "dataset": cfg.name,
        "h5ad_path": str(cfg.h5ad_path),
        "h5ad_sha256_before": h5ad_sha256_before,
        "geneformer_dir": str(geneformer_dir),
        "geneformer_commit": git_commit(geneformer_dir),
        "model_dir": str(model_dir),
        "work_dir": str(work_dir),
        "variants": variant_summaries,
        "runtime": {
            "nproc": nproc,
            "chunk_size": chunk_size,
            "forward_batch_size": forward_batch_size,
            "force_cpu": force_cpu,
            "max_cells": max_cells,
            "overwrite": overwrite,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if max_cells is not None else ""
    out = results_dir / f"geneformer_{cfg.name}{suffix}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {out}", flush=True)
    return manifest


def parse_variants(raw: str) -> list[str]:
    variants = [piece.strip() for piece in raw.split(",") if piece.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {sorted(VARIANTS)}")
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Geneformer embeddings for ortholog tier variants.")
    parser.add_argument("--dataset", choices=[*sorted(DATASETS), "all"], default="all")
    parser.add_argument("--variants", default="tier1,tier12,tier123")
    parser.add_argument("--geneformer-dir", type=Path, default=DEFAULT_GENEFORMER_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--nproc", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--forward-batch-size", type=int, default=32)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Delete and rebuild per-variant work dirs.")
    parser.add_argument("--force-cpu", dest="force_cpu", action="store_true", default=True)
    parser.add_argument("--no-force-cpu", dest="force_cpu", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_geneformer_import(args.geneformer_dir)
    variants = parse_variants(args.variants)
    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in datasets:
        run_dataset(
            cfg=DATASETS[name],
            variants=variants,
            geneformer_dir=args.geneformer_dir,
            model_dir=args.model_dir,
            work_dir=args.work_dir,
            results_dir=args.results_dir,
            nproc=args.nproc,
            chunk_size=args.chunk_size,
            forward_batch_size=args.forward_batch_size,
            max_cells=args.max_cells,
            overwrite=args.overwrite,
            force_cpu=args.force_cpu,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
