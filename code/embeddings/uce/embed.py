"""Reproducible UCE cell embeddings for full and ortholog-tier gene sets.

This script wraps ``models/UCE/eval_single_anndata.py`` for Dictyostelium and
C. elegans, runs both 4-layer and 33-layer checkpoints, and writes coexistable
raw/PCA50 embeddings into the cleaned target H5AD files. It can also patch the
local UCE checkout to build protein-only cell sentences without chromosome
tokens for distant-species sensitivity analyses.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import random
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UCE_DIR = ROOT / "models" / "UCE"
DEFAULT_DATA_UCE_DIR = ROOT / "data" / "UCE"
DEFAULT_WORK_DIR = DEFAULT_DATA_UCE_DIR / "work"
DEFAULT_RESULTS_DIR = ROOT / "results" / "embeddings"
DEFAULT_ADAPTER_DIR = ROOT / "results" / "embeddings" / "uce" / "esm2_adapter"
DRIVE_PROTEIN_EMBEDDINGS_FOLDER = "https://drive.google.com/drive/folders/1_Dz7HS5N3GoOAG6MdhsXWY1nwLoN13DJ"
HUMAN_PROTEIN_EMBEDDING = "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
DIRECT_DRIVE_PROTEIN_EMBEDDING_IDS = {
    "Caenorhabditis_elegans.WBcel235.pep.all.gene_symbol_to_embedding_ESM2.pt": (
        "12vcUIj_NYrdsAzo3aynk97qeNsEbzYG2"
    ),
    "Dictyostelium_discoideum.dicty_2.7.pep.all.gene_symbol_to_embedding_ESM2.pt": (
        "170BaVmOUYxgjq3QlvnExbXjiDIg97u_L"
    ),
}

_EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent
if str(_EMBEDDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(_EMBEDDINGS_DIR))

DEFAULT_CACHE_DIR = ROOT / "data" / "uce" / ".cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR / "xdg"))

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from _utils import selected_symbols, valid_symbol

try:
    from anndata.io import write_elem
except ImportError:  # pragma: no cover - older anndata fallback
    from anndata.experimental import write_elem


def set_global_seeds(seed: int) -> None:
    """Seed RNGs used by this wrapper process."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    target_h5ad: Path
    uce_input: Path
    species: str
    chrom_csv_name: str
    token_name: str
    offsets_name: str
    fasta_name: str
    fasta_url: str
    protein_embedding_name: str
    assembly_name: str
    taxonomy_id: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    depth: int
    model_loc: Path
    default_batch_size: int


DATASETS = {
    "dicty": DatasetConfig(
        name="dicty",
        target_h5ad=ROOT / "data" / "dicty.h5ad",
        uce_input=DEFAULT_DATA_UCE_DIR / "dicty.h5ad",
        species="dicty",
        chrom_csv_name="dicty_to_chrom_pos.csv",
        token_name="dicty_pe_tokens.torch",
        offsets_name="dicty_offsets.pkl",
        fasta_name="Dictyostelium_discoideum.dicty_2.7.pep.all.fa",
        fasta_url=(
            "https://ftp.ensemblgenomes.ebi.ac.uk/pub/protists/release-61/"
            "fasta/dictyostelium_discoideum/pep/"
            "Dictyostelium_discoideum.dicty_2.7.pep.all.fa.gz"
        ),
        protein_embedding_name=(
            "Dictyostelium_discoideum.dicty_2.7.pep.all."
            "gene_symbol_to_embedding_ESM2.pt"
        ),
        assembly_name="dicty_2.7",
        taxonomy_id=44689,
    ),
    "elegans": DatasetConfig(
        name="elegans",
        target_h5ad=ROOT / "data" / "elegans.h5ad",
        uce_input=DEFAULT_DATA_UCE_DIR / "elegans.h5ad",
        species="elegans",
        chrom_csv_name="elegans_to_chrom_pos.csv",
        token_name="elegans_pe_tokens.torch",
        offsets_name="elegans_offsets.pkl",
        fasta_name="Caenorhabditis_elegans.WBcel235.pep.all.fa",
        fasta_url=(
            "https://ftp.ensembl.org/pub/release-115/fasta/"
            "caenorhabditis_elegans/pep/"
            "Caenorhabditis_elegans.WBcel235.pep.all.fa.gz"
        ),
        protein_embedding_name=(
            "Caenorhabditis_elegans.WBcel235.pep.all."
            "gene_symbol_to_embedding_ESM2.pt"
        ),
        assembly_name="WBcel235",
        taxonomy_id=6239,
    ),
}

VARIANTS = {
    "tier1": ("ortholog_tier_1",),
    "tier12": ("ortholog_tier_1", "ortholog_tier_2"),
    "tier123": ("ortholog_tier_1", "ortholog_tier_2", "ortholog_tier_3"),
}

FIGSHARE_FILES = {
    "species_chrom.csv": "https://ndownloader.figshare.com/files/42706558",
    "species_offsets.pkl": "https://ndownloader.figshare.com/files/42706555",
    "protein_embeddings.tar.gz": "https://ndownloader.figshare.com/files/42715213",
    "all_tokens.torch": "https://ndownloader.figshare.com/files/42706585",
    "4layer_model.torch": "https://ndownloader.figshare.com/files/42706576",
    "33l_8ep_1024t_1280.torch": "https://ndownloader.figshare.com/files/43423236",
}
FIGSHARE_ARTICLE_URLS = {
    "protein_embeddings.tar.gz": "https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806?file=42715213",
}

ADAPTER_METHODS = {
    "none": None,
    "esm2_macrogene": "macrogene",
    "esm2_whitening": "whitening",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_manifest(path: Path) -> dict[str, object]:
    path = Path(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": int(path.stat().st_size) if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }


def files_manifest(paths: Iterable[Path]) -> dict[str, dict[str, object]]:
    return {str(path): file_manifest(path) for path in paths}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in [
                "accelerate",
                "anndata",
                "gdown",
                "h5py",
                "numpy",
                "pandas",
                "requests",
                "scanpy",
                "scikit-learn",
                "scipy",
                "torch",
                "tqdm",
                "urllib3",
            ]
        },
    }
    try:
        import torch

        metadata["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
    except Exception as exc:  # noqa: BLE001
        metadata["torch_error"] = repr(exc)
    return metadata


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


def git_output(path: Path, args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.stdout
    except Exception:  # noqa: BLE001
        return None


def git_repro_metadata(path: Path) -> dict[str, object]:
    status = git_output(path, ["status", "--short"])
    diff = git_output(path, ["diff", "--", "eval_single_anndata.py", "evaluate.py", "eval_data.py"])
    return {
        "commit": git_commit(path),
        "status_short": status.strip().splitlines() if status is not None else None,
        "dirty": bool(status.strip()) if status is not None else None,
        "patched_files_diff_sha256": sha256_text(diff) if diff else None,
        "patched_files_diff": diff or "",
    }


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def remove_work_files(path: Path, suffixes: tuple[str, ...]) -> list[str]:
    """Remove temporary work files with matching suffixes and report them."""
    if path.is_file():
        candidates = [path] if path.suffix in suffixes else []
    elif path.exists():
        candidates = sorted(
            candidate
            for suffix in suffixes
            for candidate in path.rglob(f"*{suffix}")
        )
    else:
        candidates = []

    removed: list[str] = []
    for candidate in candidates:
        if candidate.exists():
            candidate.unlink()
            removed.append(str(candidate))
    return removed


def remove_work_h5ads(path: Path) -> list[str]:
    """Remove temporary H5AD files under a UCE work path and report them."""
    return remove_work_files(path, (".h5ad",))


def remove_work_npzs(path: Path) -> list[str]:
    """Remove temporary NPZ files under a UCE work path and report them."""
    return remove_work_files(path, (".npz",))


def write_obsm_array(h5ad_path: Path, key: str, array: np.ndarray) -> None:
    with h5py.File(h5ad_path, "a") as handle:
        if "obsm" not in handle:
            handle.create_group("obsm")
        obsm = handle["obsm"]
        if key in obsm:
            del obsm[key]
        write_elem(obsm, key, np.asarray(array))


def existing_obsm_keys(h5ad_path: Path) -> set[str]:
    with h5py.File(h5ad_path, "r") as handle:
        if "obsm" not in handle:
            return set()
        return set(handle["obsm"].keys())


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))


def download_url(url: str, output_path: Path, retries: int = 3) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        output_path.unlink(missing_ok=True)
        suffix = f" (attempt {attempt}/{retries})" if retries > 1 else ""
        print(f"Downloading {output_path} from {url}{suffix}", flush=True)
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                if response.status_code == 202 and response.headers.get("x-amzn-waf-action"):
                    raise RuntimeError(
                        "Figshare returned an AWS WAF challenge instead of the file. "
                        f"Open {url} in a browser or place the asset at {output_path}."
                    )
                response.raise_for_status()
                with output_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if output_path.exists() and output_path.stat().st_size > 0:
                return
            raise RuntimeError(f"Downloaded empty file from {url}: {output_path}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            output_path.unlink(missing_ok=True)
            if attempt < retries:
                print(f"Download failed; retrying: {exc}", flush=True)

    if last_error is not None:
        output_path.unlink(missing_ok=True)
        article_url = next(
            (page_url for name, page_url in FIGSHARE_ARTICLE_URLS.items() if FIGSHARE_FILES.get(name) == url),
            url,
        )
        raise RuntimeError(
            f"Failed to download non-empty file from {url}: {output_path}. "
            f"If Figshare is blocking scripted downloads, download it from {article_url} "
            f"and save it to {output_path}."
        ) from last_error


def ensure_protein_embedding_assets(uce_dir: Path) -> list[str]:
    model_files = uce_dir / "model_files"
    model_files.mkdir(parents=True, exist_ok=True)
    downloaded = []
    archive = model_files / "protein_embeddings.tar.gz"
    protein_dir = model_files / "protein_embeddings"
    human_embedding = protein_dir / HUMAN_PROTEIN_EMBEDDING
    if human_embedding.exists() and human_embedding.stat().st_size > 0:
        return downloaded

    if not archive.exists() or archive.stat().st_size == 0:
        download_url(FIGSHARE_FILES["protein_embeddings.tar.gz"], archive)
        downloaded.append(str(archive))

    for attempt in range(2):
        if archive.stat().st_size > 0 and tarfile.is_tarfile(archive):
            break
        archive.unlink(missing_ok=True)
        if attempt == 1:
            raise RuntimeError(f"Downloaded archive was empty or invalid after retry: {archive}")
        download_url(FIGSHARE_FILES["protein_embeddings.tar.gz"], archive)
        downloaded.append(str(archive))

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=model_files)
    downloaded.append(str(protein_dir))
    return downloaded


def ensure_core_assets(uce_dir: Path) -> list[str]:
    model_files = uce_dir / "model_files"
    model_files.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename, url in FIGSHARE_FILES.items():
        path = model_files / filename
        if not path.exists() or path.stat().st_size == 0:
            download_url(url, path)
            downloaded.append(str(path))

    downloaded.extend(ensure_protein_embedding_assets(uce_dir))
    return downloaded


def ensure_drive_protein_embedding(uce_dir: Path, filename: str) -> Path:
    protein_dir = uce_dir / "model_files" / "protein_embeddings"
    protein_dir.mkdir(parents=True, exist_ok=True)
    target = protein_dir / filename
    if target.exists() and target.stat().st_size > 0:
        return target

    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            f"Missing {target}. Install gdown or place the file from "
            f"{DRIVE_PROTEIN_EMBEDDINGS_FOLDER} into {protein_dir}."
        ) from exc

    file_id = DIRECT_DRIVE_PROTEIN_EMBEDDING_IDS.get(filename)
    if file_id is None:
        print(f"Listing Drive folder metadata to find {filename}", flush=True)
        try:
            files = gdown.download_folder(
                DRIVE_PROTEIN_EMBEDDINGS_FOLDER,
                quiet=True,
                skip_download=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                f"Cannot locate {filename} without downloading the whole Drive folder. "
                "Upgrade gdown or add this file to DIRECT_DRIVE_PROTEIN_EMBEDDING_IDS."
            ) from exc
        for drive_file in files or []:
            if Path(drive_file.path).name == filename:
                file_id = drive_file.id
                break

    if file_id is None:
        raise FileNotFoundError(
            f"{filename} was not found in Drive metadata. Place it manually at {target} "
            "or add a direct Google Drive file ID to DIRECT_DRIVE_PROTEIN_EMBEDDING_IDS."
        )

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {filename} from Google Drive file id {file_id}", flush=True)
    result = gdown.download(url, str(target), quiet=False)
    if result is None or not target.exists() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download non-empty Google Drive file: {filename}")
    return target


def ensure_new_species_csv(uce_dir: Path, species_paths: dict[str, Path]) -> Path:
    out = uce_dir / "model_files" / "new_species_protein_embeddings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["species", "path"])
        for species, path in sorted(species_paths.items()):
            writer.writerow([species, str(path.resolve())])
    return out


def ensure_fasta(cfg: DatasetConfig, uce_dir: Path) -> Path:
    fasta_dir = uce_dir / "model_files" / "fasta_files"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = fasta_dir / cfg.fasta_name
    if fasta_path.exists() and fasta_path.stat().st_size > 0:
        return fasta_path

    gz_path = fasta_path.with_suffix(fasta_path.suffix + ".gz")
    download_url(cfg.fasta_url, gz_path)
    with gzip.open(gz_path, "rb") as src, fasta_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return fasta_path


def parse_fasta_positions(fasta_path: Path, species: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            tokens = line.strip().split()
            gene_tokens = [token for token in tokens if token.startswith("gene:")]
            if not gene_tokens:
                continue
            gene = gene_tokens[0].split(":", 1)[1]
            chrom_token = next(
                (
                    token
                    for token in tokens
                    if token.startswith(("chromosome:", "primary_assembly:", "supercontig:", "scaffold:"))
                ),
                None,
            )
            if chrom_token is None:
                continue
            parts = chrom_token.split(":")
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "gene_symbol": gene.upper(),
                    "chromosome": parts[2],
                    "start": int(parts[3]),
                    "species": species,
                }
            )
    if not rows:
        raise ValueError(f"No gene/chromosome positions parsed from {fasta_path}")
    frame = pd.DataFrame(rows)
    return frame.sort_values(["chromosome", "start"], kind="stable")


def ensure_species_token_files(
    cfg: DatasetConfig,
    uce_dir: Path,
    protein_embedding: Path,
    token_suffix: str = "",
) -> dict[str, object]:
    import torch

    data_proc = uce_dir / "data_proc"
    data_proc.mkdir(parents=True, exist_ok=True)
    chrom_csv = data_proc / cfg.chrom_csv_name
    if token_suffix:
        token_file = data_proc / f"{Path(cfg.token_name).stem}_{token_suffix}.torch"
        offsets_pkl = data_proc / f"{Path(cfg.offsets_name).stem}_{token_suffix}.pkl"
    else:
        token_file = data_proc / cfg.token_name
        offsets_pkl = data_proc / cfg.offsets_name

    fasta_path = ensure_fasta(cfg, uce_dir)
    if not chrom_csv.exists() or chrom_csv.stat().st_size == 0:
        positions = parse_fasta_positions(fasta_path, cfg.species)
        positions.to_csv(chrom_csv, index=False)
    else:
        positions = pd.read_csv(chrom_csv)

    pe = torch.load(protein_embedding, map_location="cpu")
    chrom_token_offset = 4 + len(pe)

    if token_suffix or not token_file.exists() or not offsets_pkl.exists():
        all_tokens = torch.load(uce_dir / "model_files" / "all_tokens.torch", map_location="cpu")[0:4]
        pe_stacked = torch.stack(list(pe.values()))
        all_tokens = torch.vstack((all_tokens, pe_stacked))
        torch.manual_seed(cfg.taxonomy_id)
        n_chrom = int(positions["chromosome"].nunique())
        chrom_tensors = torch.normal(mean=0, std=1, size=(n_chrom, all_tokens.shape[1]))
        all_tokens = torch.vstack((all_tokens, chrom_tensors))
        all_tokens.requires_grad = False
        torch.save(all_tokens, token_file)
        with offsets_pkl.open("wb") as handle:
            pickle.dump({cfg.species: 4}, handle)

    return {
        "fasta_path": str(fasta_path),
        "protein_embedding_path": str(protein_embedding),
        "chrom_csv_path": str(chrom_csv),
        "token_file": str(token_file),
        "offset_pkl_path": str(offsets_pkl),
        "fasta_sha256": sha256_file(fasta_path),
        "protein_embedding_sha256": sha256_file(protein_embedding),
        "chrom_csv_sha256": sha256_file(chrom_csv),
        "token_file_sha256": sha256_file(token_file),
        "offset_pkl_sha256": sha256_file(offsets_pkl),
        "chrom_token_offset": int(chrom_token_offset),
        "n_chromosomes": int(positions["chromosome"].nunique()),
        "n_chrom_position_genes": int(positions.shape[0]),
        "n_protein_embedding_genes": int(len(pe)),
    }


def normalize_dicty_var_names(adata: ad.AnnData) -> ad.AnnData:
    if "gene_ids" in adata.var.columns:
        adata = adata.copy()
        adata.var_names = adata.var["gene_ids"].astype(str).to_numpy()
        adata.var_names_make_unique()
    return adata


def normalize_elegans_var_names(adata: ad.AnnData) -> ad.AnnData:
    if "id" in adata.var.columns:
        adata = adata.copy()
        adata.var_names = adata.var["id"].astype(str).to_numpy()
        adata.var_names_make_unique()
    return adata


def parse_exclude_genes(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(piece.strip() for piece in raw.split(",") if piece.strip())


def resolve_exclude_tokens_to_var_names(adata: ad.AnnData, tokens: tuple[str, ...]) -> tuple[set[str], list[str]]:
    """Map CLI tokens to ``var_names`` rows to drop.

    After dicty normalization, ``var_names`` are ``gene_ids`` (e.g. ``DDB_*``), so a
    symbol like ``rad51-1`` must match ``var['Gene Symbol']`` (case-insensitive).
    """
    names = pd.Index(adata.var_names.astype(str))
    name_set = set(names)
    resolved: set[str] = set()
    unresolved: list[str] = []
    for token in tokens:
        if token in name_set:
            resolved.add(token)
            continue
        if "gene_ids" in adata.var.columns:
            gid = adata.var["gene_ids"].astype(str)
            if token in set(gid):
                mask = gid == token
                resolved.update(names[mask].tolist())
                continue
        if "Gene Symbol" in adata.var.columns:
            sym = adata.var["Gene Symbol"].astype(str).str.strip()
            tu = token.strip().upper()
            mask = sym.str.upper() == tu
            if mask.any():
                resolved.update(names[mask].tolist())
                continue
        unresolved.append(token)
    return resolved, unresolved


def drop_vars_by_name(adata: ad.AnnData, exclude: tuple[str, ...]) -> tuple[ad.AnnData, dict[str, object]]:
    """Drop genes matching ``exclude`` tokens (resolved to ``var_names``)."""
    if not exclude:
        return adata, {}
    to_drop, unresolved = resolve_exclude_tokens_to_var_names(adata, exclude)
    names = pd.Index(adata.var_names.astype(str))
    if not to_drop:
        return adata, {
            "exclude_genes_requested": list(exclude),
            "exclude_genes_resolved_var_names": [],
            "exclude_genes_unresolved_tokens": unresolved,
            "n_genes_removed_by_exclude_genes": 0,
        }
    keep = ~names.isin(to_drop)
    adata = adata[:, keep].copy()
    return adata, {
        "exclude_genes_requested": list(exclude),
        "exclude_genes_resolved_var_names": sorted(to_drop),
        "exclude_genes_unresolved_tokens": unresolved,
        "n_genes_removed_by_exclude_genes": len(to_drop),
    }


def build_tier_symbols(cfg: DatasetConfig, uce_var: pd.DataFrame, uce_var_names: pd.Index, columns: tuple[str, ...]) -> pd.Series:
    target = ad.read_h5ad(cfg.target_h5ad, backed="r")
    try:
        target_var = target.var.copy()
        target_names = pd.Index(target.var_names.astype(str))
    finally:
        target.file.close()

    symbols = selected_symbols(target_var, columns)
    if cfg.name == "elegans":
        if "id" not in target_var.columns:
            raise ValueError(f"{cfg.target_h5ad} missing var['id']")
        index = target_var["id"].astype(str)
    else:
        index = target_names

    primary = pd.Series(symbols.to_numpy(), index=index.astype(str), dtype=object)
    maps = [primary]
    if cfg.name == "dicty" and "gene_ids" in target_var.columns:
        maps.append(pd.Series(symbols.to_numpy(), index=target_var["gene_ids"].astype(str), dtype=object))

    out = pd.Series("", index=uce_var_names.astype(str), dtype=object)
    for mapping in maps:
        fill = out.eq("") & out.index.isin(mapping.index)
        if fill.any():
            out.loc[fill] = mapping.reindex(out.index[fill]).fillna("").astype(str)
    return out


def prepare_input_h5ad(
    *,
    cfg: DatasetConfig,
    variant: str,
    output_path: Path,
    max_cells: int | None,
    max_cells_per_time: int | None,
    seed: int,
    exclude_genes: tuple[str, ...],
) -> dict[str, object]:
    adata = ad.read_h5ad(cfg.uce_input)
    if cfg.name == "dicty":
        adata = normalize_dicty_var_names(adata)
    elif cfg.name == "elegans":
        adata = normalize_elegans_var_names(adata)
    stratified_counts: dict[str, int] | None = None
    if max_cells_per_time is not None:
        if "time" not in adata.obs.columns:
            raise ValueError("--max-cells-per-time requires obs['time']")
        rng = np.random.default_rng(seed)
        selected_positions: list[int] = []
        time_values = adata.obs["time"].astype(str)
        for time_value in sorted(time_values.unique()):
            positions = np.flatnonzero((time_values == time_value).to_numpy())
            if len(positions) > max_cells_per_time:
                positions = np.sort(rng.choice(positions, size=max_cells_per_time, replace=False))
            selected_positions.extend(positions.tolist())
        selected_positions = sorted(selected_positions)
        adata = adata[selected_positions, :].copy()
        stratified_counts = adata.obs["time"].astype(str).value_counts().sort_index().astype(int).to_dict()
    if max_cells is not None:
        adata = adata[:max_cells, :].copy()

    summary: dict[str, object] = {
        "source_input": str(cfg.uce_input),
        "prepared_input": str(output_path),
        "variant": variant,
        "n_cells": int(adata.n_obs),
        "n_source_genes": int(adata.n_vars),
    }
    if stratified_counts is not None:
        summary["max_cells_per_time"] = int(max_cells_per_time)
        summary["time_stratified_cell_counts"] = stratified_counts

    if exclude_genes:
        adata, drop_meta = drop_vars_by_name(adata, exclude_genes)
        summary.update(drop_meta)
        summary["n_vars_after_exclude_genes"] = int(adata.n_vars)

    if variant != "full":
        columns = VARIANTS[variant]
        tier_symbols = build_tier_symbols(cfg, adata.var, pd.Index(adata.var_names.astype(str)), columns)
        keep = tier_symbols.map(valid_symbol).to_numpy()
        if int(keep.sum()) == 0:
            raise ValueError(f"No selected UCE genes for {cfg.name} {variant}")
        adata = adata[:, keep].copy()
        summary.update(
            {
                "tier_columns": list(columns),
                "n_selected_ortholog_genes": int(keep.sum()),
            }
        )

    # UCE writes its full AnnData output after adding one embedding. Keep the
    # temporary input lean so previous benchmark embeddings in obsm/layers are
    # not copied into large intermediate H5AD files.
    stripped_obsm_keys = list(adata.obsm.keys())
    stripped_layer_keys = list(adata.layers.keys())
    stripped_obsp_keys = list(adata.obsp.keys())
    adata = ad.AnnData(X=adata.X, obs=adata.obs.copy(), var=adata.var.copy())
    summary.update(
        {
            "stripped_obsm_keys": stripped_obsm_keys,
            "stripped_layer_keys": stripped_layer_keys,
            "stripped_obsp_keys": stripped_obsp_keys,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    adata.write_h5ad(output_path)
    summary["n_output_genes"] = int(adata.n_vars)
    return summary


def ensure_uce_input(cfg: DatasetConfig) -> None:
    if cfg.uce_input.exists():
        return

    if cfg.uce_input.is_symlink():
        cfg.uce_input.unlink()

    cfg.uce_input.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(cfg.target_h5ad, start=cfg.uce_input.parent)
    cfg.uce_input.symlink_to(relative_target)
    print(f"Linked missing UCE input: {cfg.uce_input} -> {relative_target}")


def protein_gene_keys(path: Path) -> set[str]:
    import torch

    pe = torch.load(path, map_location="cpu")
    return {str(k).upper() for k in pe.keys()}


def filter_input_to_valid_uce_genes(
    input_h5ad: Path,
    chrom_csv: Path,
    protein_embedding: Path,
    species: str,
    *,
    chromosome_aware: bool,
) -> dict[str, object]:
    adata = ad.read_h5ad(input_h5ad)
    pe_keys = protein_gene_keys(protein_embedding)
    chrom_keys: set[str] = set()
    if chromosome_aware:
        chrom = pd.read_csv(chrom_csv)
        chrom_keys = set(chrom.loc[chrom["species"].astype(str) == species, "gene_symbol"].astype(str).str.upper())
        valid = [str(g).upper() in pe_keys and str(g).upper() in chrom_keys for g in adata.var_names]
    else:
        valid = [str(g).upper() in pe_keys for g in adata.var_names]
    n_before = int(adata.n_vars)
    n_after = int(sum(valid))
    if n_after == 0:
        raise ValueError(f"No UCE-compatible genes remain after filtering {input_h5ad}")
    if n_after < n_before:
        adata = adata[:, np.asarray(valid)].copy()
        adata.write_h5ad(input_h5ad)
    return {
        "n_genes_before_uce_asset_filter": n_before,
        "n_genes_after_uce_asset_filter": n_after,
        "n_genes_removed_by_uce_asset_filter": n_before - n_after,
        "n_genes_considered_by_uce": n_after,
        "uce_input_shape": [int(adata.n_obs), n_after],
    }


def validate_uce_genes(
    input_h5ad: Path,
    chrom_csv: Path,
    protein_embedding: Path,
    species: str,
    *,
    chromosome_aware: bool,
) -> dict[str, object]:
    tmp = ad.read_h5ad(input_h5ad, backed="r")
    try:
        genes = [str(g).upper() for g in tmp.var_names]
    finally:
        tmp.file.close()

    pe_keys = protein_gene_keys(protein_embedding)
    chrom_keys: set[str] = set()
    if chromosome_aware:
        chrom = pd.read_csv(chrom_csv)
        chrom_keys = set(chrom.loc[chrom["species"].astype(str) == species, "gene_symbol"].astype(str).str.upper())
    missing_pe = [g for g in genes if g not in pe_keys]
    missing_chrom = [g for g in genes if chromosome_aware and g not in chrom_keys]
    if missing_pe or missing_chrom:
        raise ValueError(
            f"UCE gene validation failed for {input_h5ad}: "
            f"{len(missing_pe)} missing protein embeddings (first {missing_pe[:5]}), "
            f"{len(missing_chrom)} missing chromosome rows (first {missing_chrom[:5]})"
        )
    return {
        "n_validated_genes": len(genes),
        "n_missing_protein_embedding_genes": 0,
        "n_missing_chromosome_genes": 0,
        "chromosome_aware_validation": bool(chromosome_aware),
    }


def model_output_keys(
    model_name: str,
    variant: str,
    chromosome_aware: bool,
    adapter_suffix: str = "",
) -> tuple[str, str, list[str]]:
    model_part = {"4layer": "4L", "33layer": "33L"}.get(model_name, model_name)
    variant_part = {
        "tier1": "orthoT1",
        "tier12": "orthoT12",
        "tier123": "orthoT123",
    }.get(variant, variant)
    adapter_part = f"_{adapter_suffix}" if adapter_suffix else ""
    mode_suffix = "" if chromosome_aware else "_no_chrom"
    if variant == "full":
        emb_key = f"X_uce_{model_part}{adapter_part}{mode_suffix}"
        pca_key = f"{emb_key}_50"
    else:
        emb_key = f"X_uce_{model_part}_{variant_part}{adapter_part}{mode_suffix}"
        pca_key = f"{emb_key}_50"
    return emb_key, pca_key, []


def replace_once(path: Path, old: str, new: str, marker: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    if old not in text:
        raise RuntimeError(f"Could not patch {path}; expected source block not found.")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def replace_between_once(path: Path, start_marker: str, end_marker: str, new: str, marker: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    try:
        start = text.index(start_marker)
        end = text.index(end_marker, start)
    except ValueError as exc:
        raise RuntimeError(f"Could not patch {path}; expected source block not found.") from exc
    path.write_text(text[:start] + new + text[end:], encoding="utf-8")
    return True


def ensure_uce_chromosome_flag_support(uce_dir: Path) -> list[str]:
    """Patch the local UCE checkout to support protein-only cell sentences."""
    patched: list[str] = []

    eval_single = uce_dir / "eval_single_anndata.py"
    eval_file = uce_dir / "evaluate.py"
    eval_data = uce_dir / "eval_data.py"
    require_paths([eval_single, eval_file, eval_data])

    old = '''    parser.add_argument('--sample_size', type=int, default=1024,
                        help='Number of genes sampled for cell sentence')
'''
    new = '''    parser.add_argument("--chromosome_aware", action=argparse.BooleanOptionalAction, default=True,
                        help="If false, build cell sentences from CLS plus sampled protein-gene tokens only.")
    parser.add_argument('--sample_size', type=int, default=1024,
                        help='Number of genes sampled for cell sentence')
'''
    if replace_once(eval_single, old, new, "--chromosome_aware"):
        patched.append(str(eval_single))

    old = '''                gene_to_chrom_pos = get_spec_chrom_csv(
                    self.args.spec_chrom_csv_path)
                dataset_species = self.args.species
                spec_pe_genes = list(species_to_pe[dataset_species].keys())
                offset = species_to_offsets[dataset_species]
                pe_row_idxs, dataset_chroms, dataset_pos = adata_path_to_prot_chrom_starts(
                    self.adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset)
'''
    new = '''                dataset_species = self.args.species
                spec_pe_genes = list(species_to_pe[dataset_species].keys())
                offset = species_to_offsets[dataset_species]
                if self.args.chromosome_aware:
                    gene_to_chrom_pos = get_spec_chrom_csv(
                        self.args.spec_chrom_csv_path)
                    pe_row_idxs, dataset_chroms, dataset_pos = adata_path_to_prot_chrom_starts(
                        self.adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset)
                else:
                    pe_row_idxs = torch.tensor([spec_pe_genes.index(k.upper()) + offset for k in self.adata.var_names]).long()
                    dataset_chroms = np.zeros(len(pe_row_idxs), dtype=int)
                    dataset_pos = np.arange(len(pe_row_idxs), dtype=int)
'''
    if replace_once(eval_file, old, new, "self.args.chromosome_aware"):
        patched.append(str(eval_file))

    old = '''        choosen_chrom = chroms[choice_idx] # get the sampled genes chromosomes
        # order the genes by chromosome
        chrom_sort = np.argsort(choosen_chrom)  
        choice_idx = choice_idx[chrom_sort]

        # sort the genes by start
        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]

        ordered_choice_idx = np.full((args.pad_length),
                                     args.cls_token_idx)  # start with cls
        # i= 0 first token is CLS
        i = 1  # continue on to the rest of the sequence with left bracket being assumed.
        # Shuffle the chroms now, there's no natural order to chromosomes
        uq_chroms = np.unique(new_chrom)
        np.random.shuffle(uq_chroms) # shuffle
 
        # This loop is actually just over one cell
        for chrom in uq_chroms:
            # Open Chrom token
            ordered_choice_idx[i] = int(chrom) + args.CHROM_TOKEN_OFFSET # token of this chromosome # i = 1 next token is a chrom open
            i += 1
            # now sort the genes by start order within the chroms
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(
                choosen_starts[loc])  # start locations for this chromsome

            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
            i += len(to_add)
            ordered_choice_idx[i] = args.chrom_token_right_idx # add the chrom sep again
            i += 1  # add the closing token again
'''
    new = '''        ordered_choice_idx = np.full((args.pad_length),
                                     args.cls_token_idx)  # start with cls
        # i= 0 first token is CLS
        i = 1

        if args.chromosome_aware:
            choosen_chrom = chroms[choice_idx] # get the sampled genes chromosomes
            # order the genes by chromosome
            chrom_sort = np.argsort(choosen_chrom)
            choice_idx = choice_idx[chrom_sort]

            # sort the genes by start
            new_chrom = chroms[choice_idx]
            choosen_starts = starts[choice_idx]

            # Shuffle the chroms now, there's no natural order to chromosomes
            uq_chroms = np.unique(new_chrom)
            np.random.shuffle(uq_chroms) # shuffle

            # This loop is actually just over one cell
            for chrom in uq_chroms:
                # Open Chrom token
                ordered_choice_idx[i] = int(chrom) + args.CHROM_TOKEN_OFFSET # token of this chromosome
                i += 1
                # now sort the genes by start order within the chroms
                loc = np.where(new_chrom == chrom)[0]
                sort_by_start = np.argsort(
                    choosen_starts[loc])  # start locations for this chromsome

                to_add = choice_idx[loc[sort_by_start]]
                ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
                i += len(to_add)
                ordered_choice_idx[i] = args.chrom_token_right_idx # add the chrom sep again
                i += 1  # add the closing token again
        else:
            to_add = choice_idx
            ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
            i += len(to_add)
'''
    if replace_between_once(
        eval_data,
        "        choosen_chrom = chroms[choice_idx] # get the sampled genes chromosomes",
        "        longest_seq_len = max(longest_seq_len, i)",
        new,
        "args.chromosome_aware",
    ):
        patched.append(str(eval_data))

    return patched


def ensure_uce_seed_support(uce_dir: Path) -> list[str]:
    """Patch the local UCE checkout to seed stochastic gene sampling."""
    patched: list[str] = []
    eval_single = uce_dir / "eval_single_anndata.py"
    require_paths([eval_single])

    old = '''import argparse
from evaluate import AnndataProcessor
from accelerate import Accelerator
'''
    new = '''import argparse
import os
import random

import numpy as np
import torch
from evaluate import AnndataProcessor
from accelerate import Accelerator

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

'''
    if replace_once(eval_single, old, new, "def seed_everything(seed):"):
        patched.append(str(eval_single))

    old = '''    parser.add_argument('--sample_size', type=int, default=1024,
                        help='Number of genes sampled for cell sentence')
'''
    new = '''    parser.add_argument('--sample_size', type=int, default=1024,
                        help='Number of genes sampled for cell sentence')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for stochastic UCE cell sentence sampling.')
'''
    if replace_once(eval_single, old, new, "--seed"):
        patched.append(str(eval_single))

    old = '''    args = parser.parse_args()
    accelerator = Accelerator(project_dir=args.dir)
'''
    new = '''    args = parser.parse_args()
    seed_everything(args.seed)
    accelerator = Accelerator(project_dir=args.dir)
'''
    if replace_once(eval_single, old, new, "seed_everything(args.seed)"):
        patched.append(str(eval_single))

    return patched


def ensure_uce_pooling_support(uce_dir: Path) -> list[str]:
    """Patch the local UCE checkout to support non-CLS inference readouts."""
    patched: list[str] = []
    eval_single = uce_dir / "eval_single_anndata.py"
    eval_file = uce_dir / "evaluate.py"
    require_paths([eval_single, eval_file])

    old = '''    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for stochastic UCE cell sentence sampling.')
'''
    new = '''    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for stochastic UCE cell sentence sampling.')
    parser.add_argument('--pooling', choices=['cls', 'mean_gene'], default='cls',
                        help='Cell readout from UCE token outputs. cls is the original UCE behavior; mean_gene averages contextualized gene-token outputs.')
'''
    if replace_once(eval_single, old, new, "--pooling"):
        patched.append(str(eval_single))

    old = '''            batch_sentences = batch_sentences.permute(1, 0)
            if args.multi_gpu:
                batch_sentences = model.module.pe_embedding(batch_sentences.long())
            else:
                batch_sentences = model.pe_embedding(batch_sentences.long())
            batch_sentences = nn.functional.normalize(batch_sentences,
                                                      dim=2)  # Normalize token outputs now
            _, embedding = model.forward(batch_sentences, mask=mask)
            # Fix for duplicates in last batch
'''
    new = '''            batch_sentences = batch_sentences.permute(1, 0)
            token_ids = batch_sentences
            if args.multi_gpu:
                batch_sentences = model.module.pe_embedding(batch_sentences.long())
            else:
                batch_sentences = model.pe_embedding(batch_sentences.long())
            batch_sentences = nn.functional.normalize(batch_sentences,
                                                      dim=2)  # Normalize token outputs now
            gene_output, embedding = model.forward(batch_sentences, mask=mask)
            if args.pooling == "mean_gene":
                gene_mask = (
                    (token_ids >= 4)
                    & (token_ids < args.CHROM_TOKEN_OFFSET)
                    & (mask.t() > 0)
                )
                denom = gene_mask.sum(0).clamp(min=1).unsqueeze(1)
                embedding = (gene_output * gene_mask.unsqueeze(2)).sum(0) / denom
                embedding = nn.functional.normalize(embedding, dim=1)
            # Fix for duplicates in last batch
'''
    if replace_once(eval_file, old, new, "args.pooling == \"mean_gene\""):
        patched.append(str(eval_file))

    return patched


def ensure_can_write(h5ad_path: Path, keys: list[str], overwrite: bool) -> None:
    if overwrite:
        return
    existing = existing_obsm_keys(h5ad_path)
    collisions = [key for key in keys if key in existing]
    if collisions:
        raise ValueError(
            f"{h5ad_path} already has obsm key(s) {collisions}. "
            "Rerun with --overwrite to replace them."
        )


def pca50(matrix: np.ndarray) -> np.ndarray:
    n_components = min(50, matrix.shape[0], matrix.shape[1])
    return PCA(n_components=n_components, random_state=42).fit_transform(matrix).astype(np.float32)


def run_uce_command(
    *,
    uce_dir: Path,
    input_h5ad: Path,
    run_dir: Path,
    cfg: DatasetConfig,
    model_cfg: ModelConfig,
    species_assets: dict[str, object],
    batch_size: int,
    chromosome_aware: bool,
    seed: int,
    pooling: str,
) -> tuple[Path, list[str]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir_arg = str(run_dir.resolve()) + "/"
    command = [
        sys.executable,
        "eval_single_anndata.py",
        "--adata_path",
        str(input_h5ad.resolve()),
        "--dir",
        run_dir_arg,
        "--species",
        cfg.species,
        "--filter",
        "",
        "--skip",
        "",
        "--model_loc",
        str(model_cfg.model_loc),
        "--nlayers",
        str(model_cfg.depth),
        "--batch_size",
        str(batch_size),
        "--seed",
        str(seed),
        "--pooling",
        pooling,
        "--CHROM_TOKEN_OFFSET",
        str(species_assets["chrom_token_offset"]),
        "--spec_chrom_csv_path",
        str(species_assets["chrom_csv_path"]),
        "--token_file",
        str(species_assets["token_file"]),
        "--offset_pkl_path",
        str(species_assets["offset_pkl_path"]),
        "--protein_embeddings_dir",
        str((uce_dir / "model_files" / "protein_embeddings").resolve()),
    ]
    if not chromosome_aware:
        command.append("--no-chromosome_aware")
    print("Running UCE: " + " ".join(command), flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    subprocess.run(command, cwd=uce_dir, check=True, env=env)
    output = run_dir / f"{input_h5ad.stem}_uce_adata.h5ad"
    if not output.exists():
        raise FileNotFoundError(output)
    return output, command


def read_embedding_and_check_order(output_h5ad: Path, expected_h5ad: Path) -> np.ndarray:
    out = ad.read_h5ad(output_h5ad, backed="r")
    expected_adata = ad.read_h5ad(expected_h5ad, backed="r")
    try:
        if "X_uce" not in out.obsm:
            raise ValueError(f"{output_h5ad} missing obsm['X_uce']")
        expected = list(expected_adata.obs_names.astype(str))
        got = list(out.obs_names.astype(str))
        if got != expected:
            raise ValueError(
                f"Cell order mismatch for {output_h5ad}: "
                f"first output {got[:3]}, expected {expected[:3]}"
            )
        return np.asarray(out.obsm["X_uce"], dtype=np.float32)
    finally:
        out.file.close()
        expected_adata.file.close()


def run_variant(
    *,
    cfg: DatasetConfig,
    model_cfg: ModelConfig,
    variant: str,
    uce_dir: Path,
    work_dir: Path,
    species_assets: dict[str, object],
    batch_size: int,
    max_cells: int | None,
    max_cells_per_time: int | None,
    overwrite: bool,
    exclude_genes: tuple[str, ...],
    chromosome_aware: bool,
    seed: int,
    adapter_suffix: str,
    pooling: str,
) -> dict[str, object]:
    is_smoke = max_cells is not None or max_cells_per_time is not None
    mode_dir = "" if chromosome_aware else "no_chrom"
    variant_dir = work_dir / cfg.name / model_cfg.name / mode_dir / variant if mode_dir else work_dir / cfg.name / model_cfg.name / variant
    if overwrite:
        remove_path(variant_dir)
    input_path = variant_dir / "input" / f"{cfg.name}_{model_cfg.name}_{variant}.h5ad"
    prep_summary = prepare_input_h5ad(
        cfg=cfg,
        variant=variant,
        output_path=input_path,
        max_cells=max_cells,
        max_cells_per_time=max_cells_per_time,
        seed=seed,
        exclude_genes=exclude_genes,
    )
    prep_summary.update(
        filter_input_to_valid_uce_genes(
            input_path,
            Path(str(species_assets["chrom_csv_path"])),
            Path(str(species_assets["protein_embedding_path"])),
            cfg.species,
            chromosome_aware=chromosome_aware,
        )
    )
    validation = validate_uce_genes(
        input_path,
        Path(str(species_assets["chrom_csv_path"])),
        Path(str(species_assets["protein_embedding_path"])),
        cfg.species,
        chromosome_aware=chromosome_aware,
    )
    pooling_suffix = "" if pooling == "cls" else f"pool_{pooling}"
    full_suffix = "_".join(piece for piece in [adapter_suffix, pooling_suffix] if piece)
    emb_key, pca_key, aliases = model_output_keys(model_cfg.name, variant, chromosome_aware, full_suffix)
    keys_to_write = [emb_key, pca_key, *aliases]
    if not is_smoke:
        ensure_can_write(cfg.target_h5ad, keys_to_write, overwrite)

    command: list[str] = []
    uce_output: Path | None = None
    removed_work_h5ad: list[str] = []
    removed_work_npz: list[str] = []
    try:
        uce_output, command = run_uce_command(
            uce_dir=uce_dir,
            input_h5ad=input_path,
            run_dir=variant_dir / "uce",
            cfg=cfg,
            model_cfg=model_cfg,
            species_assets=species_assets,
            batch_size=batch_size,
            chromosome_aware=chromosome_aware,
            seed=seed,
            pooling=pooling,
        )
        matrix = read_embedding_and_check_order(uce_output, input_path)
    finally:
        removed_work_h5ad.extend(remove_work_h5ads(input_path))
        if uce_output is not None:
            removed_work_h5ad.extend(remove_work_h5ads(uce_output))
        removed_work_h5ad.extend(remove_work_h5ads(variant_dir / "input"))
        removed_work_h5ad.extend(remove_work_h5ads(variant_dir / "uce"))
        removed_work_npz.extend(remove_work_npzs(variant_dir / "uce"))

    pca = pca50(matrix)

    smoke_outputs: list[str] = []
    written_keys: list[str] = []
    if not is_smoke:
        write_obsm_array(cfg.target_h5ad, emb_key, matrix)
        write_obsm_array(cfg.target_h5ad, pca_key, pca)
        written_keys.extend([emb_key, pca_key])
        for alias in aliases:
            arr = matrix if not alias.endswith("_50") else pca
            write_obsm_array(cfg.target_h5ad, alias, arr)
            written_keys.append(alias)
    else:
        out_dir = variant_dir / "embeddings"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_stem = f"{cfg.name}_{model_cfg.name}_{variant}"
        if full_suffix:
            output_stem = f"{output_stem}_{full_suffix}"
        emb_npy = out_dir / f"{output_stem}_embedding.npy"
        pca_npy = out_dir / f"{output_stem}_pca50.npy"
        np.save(emb_npy, matrix)
        np.save(pca_npy, pca)
        smoke_outputs.extend([str(emb_npy), str(pca_npy)])

    return {
        **prep_summary,
        **validation,
        "model": model_cfg.name,
        "model_depth": model_cfg.depth,
        "model_loc": str(model_cfg.model_loc),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "chromosome_aware": bool(chromosome_aware),
        "pooling": pooling,
        "protein_adapter": adapter_suffix or "none",
        "work_h5ad_removed": removed_work_h5ad,
        "work_npz_removed": removed_work_npz,
        "uce_command": command,
        "embedding_shape": list(matrix.shape),
        "pca_shape": list(pca.shape),
        "obsm_keys_written": written_keys,
        "smoke_embedding_arrays": smoke_outputs,
    }


def run_dataset(
    *,
    cfg: DatasetConfig,
    models: list[str],
    variants: list[str],
    uce_dir: Path,
    work_dir: Path,
    results_dir: Path,
    batch_size_4layer: int,
    batch_size_33layer: int,
    max_cells: int | None,
    max_cells_per_time: int | None,
    overwrite: bool,
    exclude_genes: tuple[str, ...],
    chromosome_aware: bool,
    seed: int,
    protein_adapter: str,
    protein_adapter_tiers: str,
    adapter_dir: Path,
    pooling: str,
) -> dict[str, object]:
    if not cfg.target_h5ad.exists():
        raise FileNotFoundError(cfg.target_h5ad)
    ensure_uce_input(cfg)
    patched_uce_files = [
        *ensure_uce_chromosome_flag_support(uce_dir),
        *ensure_uce_seed_support(uce_dir),
        *ensure_uce_pooling_support(uce_dir),
    ]

    downloaded = ensure_core_assets(uce_dir)
    protein_paths = {
        name: ensure_drive_protein_embedding(uce_dir, ds.protein_embedding_name)
        for name, ds in DATASETS.items()
    }
    adapter_metadata: dict[str, object] | None = None
    adapter_suffix = ""
    token_suffix = ""
    if protein_adapter != "none":
        adapter_embedding, adapter_metadata = resolve_adapter_embedding(
            adapter_dir=adapter_dir,
            cfg=cfg,
            adapter_name=protein_adapter,
            tiers=protein_adapter_tiers,
        )
        protein_paths[cfg.name] = adapter_embedding
        adapter_suffix = adapter_output_slug(
            str(adapter_metadata["method_slug"]),
            str(adapter_metadata["tiers_label"]),
        )
        token_suffix = f"adapter_{adapter_metadata['method_slug']}_{adapter_metadata['tiers_label']}"

    new_species_csv = ensure_new_species_csv(
        uce_dir,
        {DATASETS[name].species: path for name, path in protein_paths.items()},
    )
    species_assets = ensure_species_token_files(cfg, uce_dir, protein_paths[cfg.name], token_suffix=token_suffix)

    model_configs = {
        "4layer": ModelConfig(
            name="4layer",
            depth=4,
            model_loc=uce_dir / "model_files" / "4layer_model.torch",
            default_batch_size=batch_size_4layer,
        ),
        "33layer": ModelConfig(
            name="33layer",
            depth=33,
            model_loc=uce_dir / "model_files" / "33l_8ep_1024t_1280.torch",
            default_batch_size=batch_size_33layer,
        ),
    }
    require_paths([model_configs[m].model_loc for m in models])

    h5ad_sha256_before = sha256_file(cfg.target_h5ad) if max_cells is None else None
    run_summaries = []
    final_work_h5ad_removed: list[str] = []
    final_work_npz_removed: list[str] = []
    try:
        for model_name in models:
            model_cfg = model_configs[model_name]
            for variant in variants:
                print(f"=== UCE {cfg.name}: {model_name} {variant} ===", flush=True)
                run_summaries.append(
                    run_variant(
                        cfg=cfg,
                        model_cfg=model_cfg,
                        variant=variant,
                        uce_dir=uce_dir,
                        work_dir=work_dir,
                        species_assets=species_assets,
                        batch_size=model_cfg.default_batch_size,
                        max_cells=max_cells,
                        max_cells_per_time=max_cells_per_time,
                        overwrite=overwrite,
                        exclude_genes=exclude_genes,
                        chromosome_aware=chromosome_aware,
                        seed=seed,
                        adapter_suffix=adapter_suffix,
                        pooling=pooling,
                    )
                )
    finally:
        final_work_h5ad_removed = remove_work_h5ads(work_dir / cfg.name)
        final_work_npz_removed = remove_work_npzs(work_dir / cfg.name)

    h5ad_sha256_after = sha256_file(cfg.target_h5ad) if max_cells is None else None
    model_asset_paths = [model_configs[m].model_loc for m in models]
    core_asset_paths = [
        uce_dir / "model_files" / "all_tokens.torch",
        uce_dir / "model_files" / "species_chrom.csv",
        uce_dir / "model_files" / "species_offsets.pkl",
        uce_dir / "model_files" / "protein_embeddings" / HUMAN_PROTEIN_EMBEDDING,
    ]
    species_asset_paths = [
        Path(str(species_assets[key]))
        for key in ["fasta_path", "protein_embedding_path", "chrom_csv_path", "token_file", "offset_pkl_path"]
    ]
    adapter_asset_paths = []
    if adapter_metadata is not None:
        adapter_asset_paths.append(Path(str(adapter_metadata["embedding_path"])))
        if adapter_metadata.get("manifest_path"):
            adapter_asset_paths.append(Path(str(adapter_metadata["manifest_path"])))

    manifest = {
        "step": "uce_embed",
        "dataset": cfg.name,
        "target_h5ad": str(cfg.target_h5ad),
        "target_h5ad_sha256_before": h5ad_sha256_before,
        "target_h5ad_sha256_after": h5ad_sha256_after,
        "uce_input": str(cfg.uce_input),
        "uce_dir": str(uce_dir),
        "uce_commit": git_commit(uce_dir),
        "work_dir": str(work_dir),
        "new_species_csv": str(new_species_csv),
        "downloaded_assets": downloaded,
        "patched_uce_files": patched_uce_files,
        "species_assets": species_assets,
        "protein_adapter": adapter_metadata,
        "runs": run_summaries,
        "work_h5ad_removed_after_run": final_work_h5ad_removed,
        "work_npz_removed_after_run": final_work_npz_removed,
        "reproducibility": {
            "environment": environment_metadata(),
            "uce_git": git_repro_metadata(uce_dir),
            "asset_files": files_manifest(
                [
                    cfg.target_h5ad,
                    cfg.uce_input,
                    new_species_csv,
                    *model_asset_paths,
                    *core_asset_paths,
                    *species_asset_paths,
                    *adapter_asset_paths,
                ]
            ),
        },
        "runtime": {
            "models": models,
            "variants": variants,
            "batch_size_4layer": batch_size_4layer,
            "batch_size_33layer": batch_size_33layer,
            "max_cells": max_cells,
            "max_cells_per_time": max_cells_per_time,
            "overwrite": overwrite,
            "exclude_genes": list(exclude_genes),
            "chromosome_aware": chromosome_aware,
            "seed": int(seed),
            "pooling": pooling,
            "protein_adapter": protein_adapter,
            "protein_adapter_tiers": protein_adapter_tiers,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if max_cells is not None or max_cells_per_time is not None else ""
    out = results_dir / f"uce_{cfg.name}{suffix}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {out}", flush=True)
    return manifest


def parse_csv(raw: str, allowed: set[str], label: str) -> list[str]:
    values = [piece.strip() for piece in raw.split(",") if piece.strip()]
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"Unknown {label} {unknown}; choose from {sorted(allowed)}")
    return values


def adapter_method_slug(adapter_name: str) -> str:
    method = ADAPTER_METHODS[adapter_name]
    if method is None:
        return ""
    return method


def adapter_output_slug(method_slug: str, tiers_label: str) -> str:
    if tiers_label == "global":
        return method_slug
    if tiers_label == "tier1":
        return f"{method_slug}T1"
    raise ValueError(f"Unsupported ESM2 whitening adapter tiers: {tiers_label}")


def adapter_tiers_label(raw: str) -> str:
    value = raw.strip().lower()
    if value not in {"global", "tier1"}:
        raise ValueError("Supported ESM2 whitening adapter tiers are: global, tier1")
    return value


def resolve_adapter_embedding(adapter_dir: Path, cfg: DatasetConfig, adapter_name: str, tiers: str) -> tuple[Path, dict[str, object]]:
    method_slug = adapter_method_slug(adapter_name)
    tiers_label = adapter_tiers_label(tiers)
    run_name = f"{cfg.name}_{method_slug}_{tiers_label}"
    run_dir = adapter_dir / run_name
    embedding = run_dir / f"{run_name}_gene_symbol_to_embedding_ESM2_adapted.pt"
    manifest = run_dir / f"{run_name}_manifest.json"
    if not embedding.exists():
        raise FileNotFoundError(
            f"Missing adapted protein embedding: {embedding}. "
            "Run code/embeddings/uce/fit_whitening.py first."
        )
    metadata: dict[str, object] = {
        "name": adapter_name,
        "method_slug": method_slug,
        "tiers_label": tiers_label,
        "run_dir": str(run_dir),
        "embedding_path": str(embedding),
        "manifest_path": str(manifest) if manifest.exists() else None,
    }
    if manifest.exists():
        metadata["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    return embedding, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UCE embeddings for full and ortholog-tier variants.")
    parser.add_argument("--dataset", choices=[*sorted(DATASETS), "all"], default="all")
    parser.add_argument("--models", default="4layer,33layer")
    parser.add_argument("--variants", default="full,tier1,tier12,tier123")
    parser.add_argument("--uce-dir", type=Path, default=DEFAULT_UCE_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--batch-size-4layer", type=int, default=25)
    parser.add_argument("--batch-size-33layer", type=int, default=4)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument(
        "--max-cells-per-time",
        type=int,
        default=None,
        help="Smoke-test subset: sample up to this many cells per obs['time'] group before UCE.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed UCE stochastic gene sampling and PCA setup.")
    parser.add_argument(
        "--protein-adapter",
        choices=sorted(ADAPTER_METHODS),
        default="none",
        help="Use an ESM2 protein dictionary adapter fitted by code/embeddings/uce/fit_whitening.py.",
    )
    parser.add_argument(
        "--protein-adapter-tiers",
        default="global",
        help="Adapter label. Whitening supports global or tier1; macrogene currently uses global.",
    )
    parser.add_argument(
        "--exclude-genes",
        default="",
        help=(
            "Comma-separated genes to drop before UCE: current var_names, "
            "var['gene_ids'], or var['Gene Symbol'] (case-insensitive). "
            "Example: rad51-1 or DDB_G0273139"
        ),
    )
    parser.add_argument(
        "--no-chromosome-aware",
        action="store_true",
        help=(
            "Disable UCE chromosome/start-position sentence construction. "
            "The model receives CLS plus sampled protein-gene tokens only, and "
            "outputs are written with a _no_chrom suffix."
        ),
    )
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean_gene"],
        default="cls",
        help="Cell readout from UCE token outputs. cls matches original UCE; mean_gene averages contextualized gene-token outputs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seeds(args.seed)
    uce_dir = args.uce_dir.resolve()
    if not (uce_dir / "eval_single_anndata.py").exists():
        raise FileNotFoundError(f"UCE checkout missing eval_single_anndata.py: {uce_dir}")

    models = parse_csv(args.models, {"4layer", "33layer"}, "models")
    variants = parse_csv(args.variants, {"full", *VARIANTS.keys()}, "variants")
    exclude_genes = parse_exclude_genes(args.exclude_genes)
    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in datasets:
        run_dataset(
            cfg=DATASETS[name],
            models=models,
            variants=variants,
            uce_dir=uce_dir,
            work_dir=args.work_dir,
            results_dir=args.results_dir,
            batch_size_4layer=args.batch_size_4layer,
            batch_size_33layer=args.batch_size_33layer,
            max_cells=args.max_cells,
            max_cells_per_time=args.max_cells_per_time,
            overwrite=args.overwrite,
            exclude_genes=exclude_genes,
            chromosome_aware=not args.no_chromosome_aware,
            seed=args.seed,
            protein_adapter=args.protein_adapter,
            protein_adapter_tiers=args.protein_adapter_tiers,
            adapter_dir=args.adapter_dir,
            pooling=args.pooling,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
