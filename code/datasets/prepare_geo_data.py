"""Download the Dicty and C. elegans benchmark sources and build clean H5ADs.

Sources
-------
Dictyostelium discoideum (GSE305468; Zenodo 10.5281/zenodo.18481907)
    The paper's filtered AX4 H5AD is downloaded from its versioned Zenodo
    dataset record. GEO remains the archive for the underlying sequencing
    data, while Zenodo preserves the exact analysis-ready cell selection.

Caenorhabditis elegans (GSE126954)
    A gzipped Matrix Market gene-by-cell count matrix plus cell and gene
    annotation CSVs. They are converted into ``data/elegans.h5ad``.

The default action is download + build. Existing output H5ADs are never
overwritten unless ``--overwrite`` is passed. Downloads are retained under
``data/sources/`` so interrupted or repeated builds do not fetch them again.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOWNLOAD_DIR = ROOT / "data" / "sources"
DEFAULT_OUTPUT_DIR = ROOT / "data"
DEFAULT_MANIFEST = ROOT / "data" / "manifest.json"
USER_AGENT = "crossspecies-benchmark GEO downloader/1.0"
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_INTERVAL = 256 * 1024 * 1024


@dataclass(frozen=True)
class SourceFile:
    accession: str
    filename: str
    url: str
    timepoint: str | None = None

    def local_path(self, root: Path) -> Path:
        return root / self.accession / self.filename


DICTY_ACCESSION = "GSE305468"
DICTY_SERIES_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305468"
)
DICTY_ZENODO_DOI = "10.5281/zenodo.18481907"
DICTY_ZENODO_URL = "https://zenodo.org/records/18481907"
DICTY_FILES = (
    SourceFile(
        accession="zenodo.18481907",
        filename="AX4.h5ad",
        url=(
            "https://zenodo.org/api/records/18481907/"
            "files/AX4.h5ad/content"
        ),
    ),
)
DICTY_EXPECTED_SHAPE = (48_983, 12_883)
DICTY_EXPECTED_TIME_COUNTS = {
    "00hr": 6_626,
    "04hr": 4_823,
    "08hr": 9_323,
    "12hr": 7_725,
    "16hr": 11_827,
    "20hr": 8_659,
}


ELEGANS_ACCESSION = "GSE126954"
ELEGANS_SERIES_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126954"
)
ELEGANS_BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE126nnn/"
    "GSE126954/suppl"
)
ELEGANS_FILES = (
    SourceFile(
        accession=ELEGANS_ACCESSION,
        filename="GSE126954_gene_by_cell_count_matrix.txt.gz",
        url=f"{ELEGANS_BASE_URL}/GSE126954_gene_by_cell_count_matrix.txt.gz",
    ),
    SourceFile(
        accession=ELEGANS_ACCESSION,
        filename="GSE126954_cell_annotation.csv.gz",
        url=f"{ELEGANS_BASE_URL}/GSE126954_cell_annotation.csv.gz",
    ),
    SourceFile(
        accession=ELEGANS_ACCESSION,
        filename="GSE126954_gene_annotation.csv.gz",
        url=f"{ELEGANS_BASE_URL}/GSE126954_gene_annotation.csv.gz",
    ),
)
ELEGANS_EXPECTED_SHAPE = (89_701, 20_222)


DATASET_FILES = {
    "dicty": DICTY_FILES,
    "elegans": ELEGANS_FILES,
}
OUTPUT_FILENAMES = {
    "dicty": "dicty.h5ad",
    "elegans": "elegans.h5ad",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def download_file(source: SourceFile, root: Path, *, refresh: bool) -> Path:
    target = source.local_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not refresh:
        print(f"[download] present: {target}", flush=True)
        return target

    partial = target.with_name(f"{target.name}.part")
    if refresh and partial.exists():
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0

    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(
            f"[download] resuming {source.filename} at {human_bytes(offset)}",
            flush=True,
        )
    else:
        print(f"[download] {source.url}", flush=True)

    request = urllib.request.Request(source.url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if offset and exc.code == 416:
            print("[download] invalid partial download; restarting", flush=True)
            partial.unlink()
            return download_file(source, root, refresh=False)
        raise

    status = getattr(response, "status", None)
    append = offset > 0 and status == 206
    if offset and not append:
        print("[download] server did not honor resume; restarting", flush=True)
        offset = 0

    downloaded = offset
    next_progress = downloaded + PROGRESS_INTERVAL
    with response, partial.open("ab" if append else "wb") as handle:
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_progress:
                print(
                    f"[download]   {source.filename}: {human_bytes(downloaded)}",
                    flush=True,
                )
                next_progress = downloaded + PROGRESS_INTERVAL

    partial.replace(target)
    print(
        f"[download] saved {target} ({human_bytes(target.stat().st_size)})",
        flush=True,
    )
    return target


def source_file_records(
    sources: tuple[SourceFile, ...],
    root: Path,
    *,
    checksums: bool,
) -> list[dict[str, Any]]:
    records = []
    for source in sources:
        path = source.local_path(root)
        record: dict[str, Any] = {
            "accession": source.accession,
            "filename": source.filename,
            "url": source.url,
            "bytes": path.stat().st_size,
        }
        if source.timepoint is not None:
            record["timepoint"] = source.timepoint
        if checksums:
            print(f"[manifest] hashing {path}", flush=True)
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def ensure_sources(dataset: str, download_dir: Path) -> None:
    missing = [
        str(source.local_path(download_dir))
        for source in DATASET_FILES[dataset]
        if not source.local_path(download_dir).is_file()
    ]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Missing downloaded files for {dataset}:\n  {joined}\n"
            "Run without --build-only first."
        )


def atomic_write_h5ad(adata: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.",
        suffix=".h5ad",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        adata.write_h5ad(temporary, compression="gzip")
        temporary.chmod(0o644)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _drop_source_derived_data(adata: Any) -> None:
    """Keep expression/metadata but remove GEO-supplied derived representations."""
    adata.raw = None
    for mapping in (adata.obsm, adata.obsp, adata.varm, adata.varp):
        for key in list(mapping.keys()):
            del mapping[key]
    adata.uns.clear()


def build_dicty(download_dir: Path, output: Path, *, keep_source_derived: bool) -> Any:
    import anndata as ad
    import pandas as pd

    source = DICTY_FILES[0]
    path = source.local_path(download_dir)
    print(f"[dicty] reading {path}", flush=True)
    combined = ad.read_h5ad(path)
    if not keep_source_derived:
        _drop_source_derived_data(combined)

    if not combined.obs_names.is_unique:
        raise ValueError("Dicty cell names are not unique")
    if not combined.var_names.is_unique:
        raise ValueError("Dicty gene names are not unique")
    combined.obs["strain"] = "AX4"
    combined.obs["cell_id"] = combined.obs_names.astype(str)

    combined.obs["time"] = pd.Categorical(
        combined.obs["time"],
        categories=list(DICTY_EXPECTED_TIME_COUNTS),
        ordered=True,
    )
    actual_counts = (
        combined.obs["time"].value_counts(sort=False).astype(int).to_dict()
    )
    if actual_counts != DICTY_EXPECTED_TIME_COUNTS:
        raise ValueError(
            f"Unexpected Dicty time-point counts: {actual_counts}; "
            f"expected {DICTY_EXPECTED_TIME_COUNTS}"
        )
    if combined.shape != DICTY_EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected Dicty shape {combined.shape}; "
            f"expected {DICTY_EXPECTED_SHAPE}"
        )
    if "counts" not in combined.layers:
        combined.layers["counts"] = combined.X.copy()

    combined.uns["geo_source"] = {
        "accession": DICTY_ACCESSION,
        "series_url": DICTY_SERIES_URL,
        "processed_dataset_doi": DICTY_ZENODO_DOI,
        "processed_dataset_url": DICTY_ZENODO_URL,
        "processed_file": source.filename,
        "processing": "Copied the paper's filtered AX4 object; source-derived "
        + ("arrays retained." if keep_source_derived else "arrays removed."),
    }
    print(f"[dicty] writing {output} with shape {combined.shape}", flush=True)
    atomic_write_h5ad(combined, output)
    return combined


def build_elegans(download_dir: Path, output: Path) -> Any:
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from scipy.io import mmread

    source_dir = download_dir / ELEGANS_ACCESSION
    matrix_path = source_dir / "GSE126954_gene_by_cell_count_matrix.txt.gz"
    cell_path = source_dir / "GSE126954_cell_annotation.csv.gz"
    gene_path = source_dir / "GSE126954_gene_annotation.csv.gz"

    print(f"[elegans] reading cell annotations: {cell_path}", flush=True)
    obs = pd.read_csv(cell_path, index_col=0)
    print(f"[elegans] reading gene annotations: {gene_path}", flush=True)
    var = pd.read_csv(gene_path, index_col=0)

    required_obs = {"cell", "embryo.time"}
    required_var = {"id", "gene_short_name"}
    if missing := required_obs - set(obs.columns):
        raise ValueError(f"Elegans cell annotations missing columns: {sorted(missing)}")
    if missing := required_var - set(var.columns):
        raise ValueError(f"Elegans gene annotations missing columns: {sorted(missing)}")
    if obs["cell"].isna().any() or not obs["cell"].is_unique:
        raise ValueError("Elegans cell IDs are missing or duplicated")
    if var["id"].isna().any() or not var["id"].is_unique:
        raise ValueError("Elegans gene IDs are missing or duplicated")
    if var["gene_short_name"].isna().any() or not var["gene_short_name"].is_unique:
        raise ValueError("Elegans gene short names are missing or duplicated")

    obs = obs.copy()
    obs.index = pd.Index(obs["cell"].astype(str), name="cell")
    obs["cell_id"] = obs.index.astype(str)

    var = var.copy()
    var.index = pd.Index(var["gene_short_name"].astype(str), name="gene_short_name")
    # Retain the stable WormBase ID used by the repository's ortholog workflow.
    var["id"] = var["id"].astype(str)
    var["gene_short_name"] = var["gene_short_name"].astype(str)

    print(f"[elegans] reading sparse count matrix: {matrix_path}", flush=True)
    with gzip.open(matrix_path, "rb") as handle:
        gene_by_cell = mmread(handle)
    if gene_by_cell.shape != (len(var), len(obs)):
        raise ValueError(
            f"Elegans matrix shape {gene_by_cell.shape} does not match "
            f"{len(var)} genes x {len(obs)} cells from the annotations"
        )
    if gene_by_cell.shape[::-1] != ELEGANS_EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected Elegans shape {gene_by_cell.shape[::-1]}; "
            f"expected {ELEGANS_EXPECTED_SHAPE}"
        )

    counts = sparse.csr_matrix(gene_by_cell.T, dtype=np.int32)
    del gene_by_cell
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    adata.uns["geo_source"] = {
        "accession": ELEGANS_ACCESSION,
        "series_url": ELEGANS_SERIES_URL,
        "matrix_file": matrix_path.name,
        "cell_annotation_file": cell_path.name,
        "gene_annotation_file": gene_path.name,
        "matrix_orientation_on_geo": "genes_by_cells",
    }
    print(f"[elegans] writing {output} with shape {adata.shape}", flush=True)
    atomic_write_h5ad(adata, output)
    return adata


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "description": "Provenance for locally built benchmark H5AD files.",
            "datasets": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot update invalid JSON manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest root must be a JSON object: {path}")
    payload.setdefault("schema_version", 1)
    payload.setdefault("datasets", {})
    if not isinstance(payload["datasets"], dict):
        raise ValueError(f"Manifest datasets must be a JSON object: {path}")
    return payload


def update_manifest(
    path: Path,
    *,
    dataset: str,
    output: Path,
    shape: tuple[int, int],
    source_records: list[dict[str, Any]],
    output_checksum: bool,
) -> None:
    payload = load_manifest(path)
    record: dict[str, Any] = {
        "geo_accession": (
            DICTY_ACCESSION if dataset == "dicty" else ELEGANS_ACCESSION
        ),
        "geo_series_url": (
            DICTY_SERIES_URL if dataset == "dicty" else ELEGANS_SERIES_URL
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "code/datasets/prepare_geo_data.py",
        "output_h5ad": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
        "shape": [int(shape[0]), int(shape[1])],
        "source_files": source_records,
    }
    if output_checksum:
        print(f"[manifest] hashing {output}", flush=True)
        record["output_h5ad_sha256"] = sha256_file(output)
    payload["datasets"][dataset] = record
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o644)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[manifest] wrote {path}", flush=True)


def selected_datasets(raw: str) -> list[str]:
    return ["dicty", "elegans"] if raw == "all" else [raw]


def output_path(args: argparse.Namespace, dataset: str) -> Path:
    if args.output is not None:
        return args.output
    return args.output_dir / OUTPUT_FILENAMES[dataset]


def print_dry_run(args: argparse.Namespace, datasets: list[str]) -> None:
    print(f"download directory: {args.download_dir}")
    print(f"output directory:   {args.output_dir}")
    print(f"manifest:           {args.manifest}")
    for dataset in datasets:
        print(f"\n{dataset}:")
        for source in DATASET_FILES[dataset]:
            print(f"  GET {source.url}")
            print(f"   -> {source.local_path(args.download_dir)}")
        if not args.download_only:
            print(f"  BUILD -> {output_path(args, dataset)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", "dicty", "elegans"],
        default="all",
        help="Dataset to prepare (default: both).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory for retained GEO downloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for dicty.h5ad and elegans.h5ad.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Exact output H5AD path; valid only with a single --dataset.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSON provenance manifest updated after each successful build.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download-only",
        action="store_true",
        help="Download the GEO supplements without building H5ADs.",
    )
    mode.add_argument(
        "--build-only",
        action="store_true",
        help="Build from files already present in --download-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output H5ADs. Never affects retained downloads.",
    )
    parser.add_argument(
        "--refresh-downloads",
        action="store_true",
        help="Download GEO files again even if their local paths already exist.",
    )
    parser.add_argument(
        "--keep-source-derived",
        action="store_true",
        help="Keep embeddings and other derived arrays in the Dicty source H5AD.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Do not SHA-256 hash source files or output H5ADs in the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs and destination paths without downloading or building.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.download_dir = args.download_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    datasets = selected_datasets(args.dataset)
    if args.output is not None and len(datasets) != 1:
        raise ValueError("--output requires --dataset dicty or --dataset elegans")

    if args.dry_run:
        print_dry_run(args, datasets)
        return 0

    if not args.download_only:
        existing = [
            output_path(args, dataset)
            for dataset in datasets
            if output_path(args, dataset).exists()
        ]
        if existing and not args.overwrite:
            joined = "\n  ".join(map(str, existing))
            raise FileExistsError(
                f"Refusing to overwrite existing output(s):\n  {joined}\n"
                "Move them elsewhere or rerun with --overwrite."
            )

    if not args.build_only:
        for dataset in datasets:
            for source in DATASET_FILES[dataset]:
                download_file(
                    source,
                    args.download_dir,
                    refresh=args.refresh_downloads,
                )

    if args.download_only:
        return 0

    for dataset in datasets:
        ensure_sources(dataset, args.download_dir)
        output = output_path(args, dataset)
        if dataset == "dicty":
            adata = build_dicty(
                args.download_dir,
                output,
                keep_source_derived=args.keep_source_derived,
            )
        else:
            adata = build_elegans(args.download_dir, output)

        source_records = source_file_records(
            DATASET_FILES[dataset],
            args.download_dir,
            checksums=not args.skip_checksums,
        )
        update_manifest(
            args.manifest,
            dataset=dataset,
            output=output,
            shape=adata.shape,
            source_records=source_records,
            output_checksum=not args.skip_checksums,
        )
        del adata

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
