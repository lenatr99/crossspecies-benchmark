"""Convert a processed matrix series plus cell metadata into an H5AD file.

This is intentionally organism-agnostic. It is for GEO-style processed
supplements where each matrix is a gene-by-cell CSV/CSV.GZ and an Excel/CSV
metadata file has one row per cell.

Expected metadata shape
-----------------------
Preferred required columns:
    matrix_file
        Matrix filename inside ``--matrix-tar`` or relative to ``--matrix-dir``.

    cell_id
        Exact column name in the expression matrix.

Optional convenience columns:
    matrix_key
        Human-readable sample key. If absent, it is copied from matrix_file.

    cell_barcode
        Barcode or suffix. Only needed if cell_id must be built from
        ``--cell-id-template``.

All other metadata columns are copied into ``adata.obs``. Recommended columns
for benchmark-like H5ADs are sample, batch, condition, timepoint, cell_type, and
cell_type_coarse.

Expected matrix shape
---------------------
Each expression matrix must be CSV or CSV.GZ with genes in rows and cells in
columns. The first column stores gene IDs, and all other columns are cell IDs.
The gene rows must have the same order in all selected matrices.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import tarfile
from array import array
from datetime import datetime, timezone
from io import TextIOWrapper
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _open_text_from_tar(tar: tarfile.TarFile, member_name: str):
    handle = tar.extractfile(member_name)
    if handle is None:
        raise FileNotFoundError(member_name)
    if member_name.endswith(".gz"):
        return TextIOWrapper(gzip.GzipFile(fileobj=handle), encoding="utf-8")
    return TextIOWrapper(handle, encoding="utf-8")


def _open_text_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _read_metadata(path: Path, *, sheet: str | int | None):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".xlsx") or suffixes.endswith(".xls"):
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    if suffixes.endswith(".tsv") or suffixes.endswith(".txt"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _split_values(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def _as_bool_series(series):
    clean = series.astype(str).str.strip().str.lower()
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n"}
    known = true_values | false_values
    unknown = sorted(set(clean.dropna()) - known)
    if unknown:
        raise ValueError(f"Cannot parse boolean values: {unknown[:5]}")
    return clean.isin(true_values)


def _apply_filters(meta, filters: Iterable[str], bool_filters: Iterable[str]):
    out = meta
    for spec in filters:
        if "=" not in spec:
            raise ValueError(f"Filter must be column=value[,value], got {spec!r}")
        column, raw_values = spec.split("=", 1)
        column = column.strip()
        if column not in out.columns:
            raise ValueError(f"Filter column {column!r} is not in metadata")
        keep = set(_split_values(raw_values))
        out = out[out[column].astype(str).isin(keep)]
    for spec in bool_filters:
        if "=" not in spec:
            raise ValueError(f"Boolean filter must be column=true|false, got {spec!r}")
        column, raw_value = spec.split("=", 1)
        column = column.strip()
        if column not in out.columns:
            raise ValueError(f"Filter column {column!r} is not in metadata")
        target = raw_value.strip().lower() in {"1", "true", "t", "yes", "y"}
        out = out[_as_bool_series(out[column]) == target]
    if out.empty:
        raise ValueError("No cells remain after metadata filtering.")
    return out.copy()


def _matrix_members_from_tar(raw_tar: Path, pattern: re.Pattern[str]) -> dict[str, str]:
    members: dict[str, str] = {}
    with tarfile.open(raw_tar, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            match = pattern.search(member.name)
            if match is None:
                continue
            key = match.group("key")
            if key in members:
                raise ValueError(f"Multiple matrix files matched key {key!r}")
            members[key] = member.name
    return members


def _matrix_members_from_dir(matrix_dir: Path, pattern: re.Pattern[str]) -> dict[str, Path]:
    members: dict[str, Path] = {}
    for path in matrix_dir.rglob("*"):
        if not path.is_file():
            continue
        match = pattern.search(str(path.relative_to(matrix_dir)))
        if match is None:
            continue
        key = match.group("key")
        if key in members:
            raise ValueError(f"Multiple matrix files matched key {key!r}")
        members[key] = path
    return members


def _read_matrix_stream(
    text,
    source_name: str,
    *,
    expected_genes: list[str] | None,
    keep_cells: set[str],
    gene_id_column: str,
    delimiter: str,
):
    import numpy as np
    from scipy import sparse

    reader = csv.reader(text, delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"{source_name}: empty matrix file") from exc
    if not header or header[0] != gene_id_column:
        raise ValueError(
            f"{source_name}: expected first header column {gene_id_column!r}, got {header[:1]}"
        )
    all_cells = header[1:]
    kept_positions = [(idx, cell) for idx, cell in enumerate(all_cells) if cell in keep_cells]
    if not kept_positions:
        raise ValueError(f"{source_name}: no matrix columns matched selected metadata cells")
    kept_cells = [cell for _, cell in kept_positions]

    rows = array("i")
    cols = array("i")
    data = array("i")
    genes: list[str] = []
    n_rows = 0
    for row_idx, parts in enumerate(reader):
        if not parts:
            continue
        gene = parts[0]
        if expected_genes is not None and row_idx >= len(expected_genes):
            raise ValueError(f"{source_name}: has more gene rows than the first matrix")
        if expected_genes is not None and expected_genes[row_idx] != gene:
            raise ValueError(
                f"{source_name}: gene order mismatch at row {row_idx}: "
                f"{gene!r} != {expected_genes[row_idx]!r}"
            )
        if expected_genes is None:
            genes.append(gene)
        for local_idx, (source_idx, _) in enumerate(kept_positions):
            matrix_col = source_idx + 1
            if matrix_col >= len(parts):
                raise ValueError(f"{source_name}: row {row_idx} has fewer columns than the header")
            value = parts[matrix_col]
            if value == "0" or value == "":
                continue
            rows.append(row_idx)
            cols.append(local_idx)
            data.append(int(float(value)))
        n_rows += 1
        if n_rows % 5000 == 0:
            print(f"[convert]   {source_name}: read {n_rows} genes", flush=True)

    gene_index = expected_genes if expected_genes is not None else genes
    if expected_genes is not None and n_rows != len(expected_genes):
        raise ValueError(f"{source_name}: has {n_rows} gene rows, expected {len(expected_genes)}")
    matrix = sparse.coo_matrix(
        (
            np.frombuffer(data, dtype=np.int32),
            (
                np.frombuffer(rows, dtype=np.int32),
                np.frombuffer(cols, dtype=np.int32),
            ),
        ),
        shape=(n_rows, len(kept_cells)),
        dtype=np.int32,
    ).tocsr()
    return gene_index, kept_cells, matrix


def _prepare_metadata(meta, *, template: str):
    meta = meta.copy()
    if "matrix_file" not in meta.columns and "matrix_key" not in meta.columns:
        raise ValueError("Metadata must contain matrix_file or matrix_key.")
    if "matrix_file" not in meta.columns:
        meta["matrix_file"] = meta["matrix_key"]
    if "matrix_key" not in meta.columns:
        meta["matrix_key"] = meta["matrix_file"]
    if "cell_id" not in meta.columns:
        if "cell_barcode" not in meta.columns:
            raise ValueError("Metadata must contain cell_id or cell_barcode.")
        meta["cell_id"] = [
            template.format(**{column: row[column] for column in meta.columns})
            for _, row in meta.iterrows()
        ]
    meta["matrix_file"] = meta["matrix_file"].astype(str)
    meta["matrix_key"] = meta["matrix_key"].astype(str)
    meta["cell_id"] = meta["cell_id"].astype(str)
    return meta


def _tar_member_names(raw_tar: Path) -> set[str]:
    with tarfile.open(raw_tar, "r") as tar:
        return {member.name for member in tar.getmembers() if member.isfile()}


def _resolve_matrix_files_from_tar(
    raw_tar: Path,
    wanted_files: list[str],
    pattern: re.Pattern[str] | None,
    wanted_keys: list[str],
) -> dict[str, str]:
    member_names = _tar_member_names(raw_tar)
    lookup = {name: name for name in wanted_files if name in member_names}
    missing_files = [name for name in wanted_files if name not in lookup]
    if missing_files and pattern is None:
        raise ValueError(f"Matrix files not found in tar: {missing_files[:5]}")
    if not missing_files:
        return lookup

    keyed = _matrix_members_from_tar(raw_tar, pattern)
    missing_keys = [key for key in wanted_keys if key not in keyed]
    if missing_keys:
        raise ValueError(
            "Some metadata matrix_file values were not in the tar and their "
            f"matrix_key values were not matched by --matrix-key-regex: {missing_keys[:5]}"
        )
    return {matrix_file: keyed[key] for matrix_file, key in zip(wanted_files, wanted_keys)}


def _resolve_matrix_files_from_dir(
    matrix_dir: Path,
    wanted_files: list[str],
    pattern: re.Pattern[str] | None,
    wanted_keys: list[str],
) -> dict[str, Path]:
    lookup = {}
    missing_files = []
    for name in wanted_files:
        path = matrix_dir / name
        if path.exists():
            lookup[name] = path
        else:
            missing_files.append(name)
    if missing_files and pattern is None:
        raise ValueError(f"Matrix files not found in directory: {missing_files[:5]}")
    if not missing_files:
        return lookup

    keyed = _matrix_members_from_dir(matrix_dir, pattern)
    missing_keys = [key for key in wanted_keys if key not in keyed]
    if missing_keys:
        raise ValueError(
            "Some metadata matrix_file values were not in the directory and their "
            f"matrix_key values were not matched by --matrix-key-regex: {missing_keys[:5]}"
        )
    return {matrix_file: keyed[key] for matrix_file, key in zip(wanted_files, wanted_keys)}


def convert(args: argparse.Namespace) -> None:
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    if bool(args.matrix_tar) == bool(args.matrix_dir):
        raise ValueError("Provide exactly one of --matrix-tar or --matrix-dir.")
    pattern = re.compile(args.matrix_key_regex) if args.matrix_key_regex else None
    if pattern is not None and "key" not in pattern.groupindex:
        raise ValueError("--matrix-key-regex must contain a named group called 'key'")

    meta = _read_metadata(args.metadata, sheet=args.metadata_sheet)
    meta = _apply_filters(meta, args.filter, args.bool_filter)
    meta = _prepare_metadata(meta, template=args.cell_id_template)

    if meta["cell_id"].duplicated().any():
        dupes = meta.loc[meta["cell_id"].duplicated(), "cell_id"].head(5).tolist()
        raise ValueError(f"Metadata has duplicate cell_id values, for example: {dupes}")

    wanted_files = list(dict.fromkeys(meta["matrix_file"].tolist()))
    wanted_keys_by_file = [
        meta.loc[meta["matrix_file"] == matrix_file, "matrix_key"].iloc[0]
        for matrix_file in wanted_files
    ]
    if args.matrix_tar:
        matrix_lookup = _resolve_matrix_files_from_tar(
            args.matrix_tar,
            wanted_files,
            pattern,
            wanted_keys_by_file,
        )
    else:
        matrix_lookup = _resolve_matrix_files_from_dir(
            args.matrix_dir,
            wanted_files,
            pattern,
            wanted_keys_by_file,
        )

    obs_frames = []
    matrices = []
    genes: list[str] | None = None
    if args.matrix_tar:
        with tarfile.open(args.matrix_tar, "r") as tar:
            for matrix_file in wanted_files:
                sample_meta = meta[meta["matrix_file"] == matrix_file].copy()
                keep_cells = set(sample_meta["cell_id"])
                member_name = matrix_lookup[matrix_file]
                print(f"[convert] reading {matrix_file}: {len(sample_meta)} selected cells", flush=True)
                with _open_text_from_tar(tar, member_name) as text:
                    genes, cells, matrix = _read_matrix_stream(
                        text,
                        member_name,
                        expected_genes=genes,
                        keep_cells=keep_cells,
                        gene_id_column=args.gene_id_column,
                        delimiter=args.delimiter,
                    )
                sample_meta = sample_meta.set_index("cell_id", drop=False).reindex(cells)
                if sample_meta["cell_id"].isna().any():
                    raise ValueError(f"{matrix_file}: metadata missing for at least one matrix cell")
                obs_frames.append(sample_meta)
                matrices.append(matrix.T.tocsr())
    else:
        for matrix_file in wanted_files:
            sample_meta = meta[meta["matrix_file"] == matrix_file].copy()
            keep_cells = set(sample_meta["cell_id"])
            matrix_path = matrix_lookup[matrix_file]
            print(f"[convert] reading {matrix_file}: {len(sample_meta)} selected cells", flush=True)
            with _open_text_file(matrix_path) as text:
                genes, cells, matrix = _read_matrix_stream(
                    text,
                    str(matrix_path),
                    expected_genes=genes,
                    keep_cells=keep_cells,
                    gene_id_column=args.gene_id_column,
                    delimiter=args.delimiter,
                )
            sample_meta = sample_meta.set_index("cell_id", drop=False).reindex(cells)
            if sample_meta["cell_id"].isna().any():
                raise ValueError(f"{matrix_file}: metadata missing for at least one matrix cell")
            obs_frames.append(sample_meta)
            matrices.append(matrix.T.tocsr())

    if genes is None:
        raise ValueError("No matrices were read.")

    X = sparse.vstack(matrices, format="csr")
    obs = pd.concat(obs_frames, axis=0)
    obs.index.name = "cell_id"
    var = pd.DataFrame(index=pd.Index(genes, name=args.var_index_name))
    var[args.var_index_name] = var.index.astype(str)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["source"] = {
        "dataset_id": args.dataset_id,
        "matrix_tar": str(args.matrix_tar) if args.matrix_tar else None,
        "matrix_dir": str(args.matrix_dir) if args.matrix_dir else None,
        "metadata": str(args.metadata),
        "metadata_sheet": args.metadata_sheet,
        "matrix_key_regex": args.matrix_key_regex,
        "gene_id_column": args.gene_id_column,
        "filters": args.filter,
        "bool_filters": args.bool_filter,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[convert] writing {args.output} with shape {adata.shape}", flush=True)
    adata.write_h5ad(args.output)

    if args.manifest:
        manifest = {
            "dataset_id": args.dataset_id,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "step": "convert_matrix_series_to_h5ad",
            "output_h5ad": str(args.output),
            "output_h5ad_sha256": sha256_file(args.output),
            "shape": [int(adata.n_obs), int(adata.n_vars)],
            "matrix_files": wanted_files,
            "n_cells_by_matrix_key": obs["matrix_key"].astype(str).value_counts().sort_index().to_dict(),
            "n_cells_by_matrix_file": obs["matrix_file"].astype(str).value_counts().sort_index().to_dict(),
            "filters": args.filter,
            "bool_filters": args.bool_filter,
        }
        write_json(args.manifest, manifest)
        print(f"[convert] wrote {args.manifest}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sheet", default=None)
    parser.add_argument("--matrix-tar", type=Path, default=None)
    parser.add_argument("--matrix-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dataset-id", default="")
    parser.add_argument(
        "--matrix-key-regex",
        default=None,
        help="Regex over matrix filenames with a named capture group (?P<key>...).",
    )
    parser.add_argument("--gene-id-column", default="rn")
    parser.add_argument("--var-index-name", default="gene_id")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument(
        "--cell-id-template",
        default="{matrix_key}_{cell_barcode}",
        help="Used only when metadata has no exact cell_id column.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Keep metadata rows matching column=value[,value]. Can be passed multiple times.",
    )
    parser.add_argument(
        "--bool-filter",
        action="append",
        default=[],
        help="Keep metadata rows matching column=true|false. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
