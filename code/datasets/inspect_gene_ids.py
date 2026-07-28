"""Inspect gene identifier columns in an H5AD file.

This is the first gate for adding a new species: before ortholog mapping,
protein embeddings, or model tokenization, we need to know which gene IDs the
dataset actually carries and whether they are unique enough to use.

The script reads only the HDF5 ``/var`` table, so it works even when the full
AnnData object cannot be loaded by the local anndata version.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_MISSING = {"", "NA", "N/A", "nan", "None", "none", "null", "NULL"}


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if value is None:
        return ""
    return str(value)


def _dataset_values(ds: h5py.Dataset, limit: int | None = None) -> list[str]:
    if limit is None:
        raw = ds[()]
    else:
        raw = ds[: min(limit, ds.shape[0])]
    arr = np.asarray(raw).reshape(-1)
    return [_decode(x) for x in arr]


def _read_column(var: h5py.Group, name: str, limit: int | None = None) -> list[str] | None:
    if name not in var:
        return None
    obj = var[name]
    if isinstance(obj, h5py.Dataset):
        return _dataset_values(obj, limit=limit)
    if isinstance(obj, h5py.Group) and {"categories", "codes"}.issubset(obj.keys()):
        categories = _dataset_values(obj["categories"], limit=None)
        codes = np.asarray(obj["codes"][: min(limit or obj["codes"].shape[0], obj["codes"].shape[0])])
        out: list[str] = []
        for code in codes.reshape(-1):
            idx = int(code)
            out.append(categories[idx] if 0 <= idx < len(categories) else "")
        return out
    return None


def _all_var_column_names(var: h5py.Group) -> list[str]:
    names = []
    for name, obj in var.items():
        if isinstance(obj, h5py.Dataset):
            names.append(name)
        elif isinstance(obj, h5py.Group) and {"categories", "codes"}.issubset(obj.keys()):
            names.append(name)
    return sorted(names)


def _first_column_length(group: h5py.Group) -> int | None:
    for name in _all_var_column_names(group):
        values = _read_column(group, name, limit=1)
        if values is None:
            continue
        obj = group[name]
        if isinstance(obj, h5py.Dataset):
            return int(obj.shape[0])
        if isinstance(obj, h5py.Group) and "codes" in obj:
            return int(obj["codes"].shape[0])
    return None


def _non_missing(values: Iterable[str], missing_tokens: set[str]) -> list[str]:
    out = []
    for value in values:
        clean = str(value).strip()
        if clean not in missing_tokens:
            out.append(clean)
    return out


def summarize_column(
    var: h5py.Group,
    name: str,
    *,
    n_genes: int,
    sample_size: int,
    missing_tokens: set[str],
) -> dict[str, object]:
    values = _read_column(var, name)
    if values is None:
        return {
            "column": name,
            "present": False,
        }

    clean = [v.strip() for v in values]
    present = _non_missing(clean, missing_tokens)
    unique_present = set(present)
    duplicate_non_missing = len(present) - len(unique_present)

    sample_values = _non_missing(_read_column(var, name, limit=n_genes) or [], missing_tokens)[:sample_size]
    return {
        "column": name,
        "present": True,
        "n_values": len(clean),
        "n_missing": len(clean) - len(present),
        "n_non_missing": len(present),
        "n_unique_non_missing": len(unique_present),
        "n_duplicate_non_missing": duplicate_non_missing,
        "non_missing_fraction": len(present) / n_genes if n_genes else 0.0,
        "unique_fraction_of_non_missing": len(unique_present) / len(present) if present else 0.0,
        "sample_values": sample_values,
    }


def inspect_h5ad(
    path: Path,
    *,
    columns: list[str] | None,
    sample_size: int,
    missing_tokens: set[str],
) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if "var" not in handle:
            raise KeyError(f"{path} has no /var group")
        var = handle["var"]
        var_names = _read_column(var, "_index")
        n_genes = len(var_names or []) or _first_column_length(var) or 0
        n_cells = None
        if "obs" in handle:
            obs = handle["obs"]
            obs_names = _read_column(obs, "_index")
            n_cells = len(obs_names or []) or _first_column_length(obs)

        available = _all_var_column_names(var)
        chosen = columns or available
        summaries = [
            summarize_column(
                var,
                name,
                n_genes=n_genes,
                sample_size=sample_size,
                missing_tokens=missing_tokens,
            )
            for name in chosen
        ]

    return {
        "path": str(path),
        "n_cells": n_cells,
        "n_genes": n_genes,
        "available_var_columns": available,
        "columns": summaries,
    }


def print_text(report: dict[str, object]) -> None:
    print(f"H5AD: {report['path']}")
    print(f"cells: {report['n_cells']}  genes: {report['n_genes']}")
    print("\n/var columns:")
    for name in report["available_var_columns"]:
        print(f"  - {name}")

    print("\nGene identifier audit:")
    for row in report["columns"]:
        if not row.get("present"):
            print(f"  {row['column']}: missing")
            continue
        print(
            "  {column}: non-missing {n_non_missing}/{n_values} "
            "({non_missing_fraction:.1%}), unique {n_unique_non_missing}, "
            "duplicates {n_duplicate_non_missing}".format(**row)
        )
        sample = row.get("sample_values") or []
        if sample:
            print("    examples: " + ", ".join(str(x) for x in sample[:8]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path, help="Input .h5ad file to inspect.")
    parser.add_argument(
        "--columns",
        default="",
        help=(
            "Comma-separated /var columns to inspect. Empty inspects all simple "
            "dataset/categorical columns."
        ),
    )
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--missing-token",
        action="append",
        default=[],
        help="Additional token to treat as missing. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    columns = [c.strip() for c in args.columns.split(",") if c.strip()] or None
    missing_tokens = set(DEFAULT_MISSING)
    missing_tokens.update(args.missing_token)
    report = inspect_h5ad(
        args.h5ad,
        columns=columns,
        sample_size=args.sample_size,
        missing_tokens=missing_tokens,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
